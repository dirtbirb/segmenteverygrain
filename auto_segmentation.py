# Setup ---

# Imports
import cv2
import glob
import keras.saving
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import segment_anything
import segmenteverygrain

# import tensorflow as tf
# from tensorflow.python.platform.build_info import build_info
# for k, v in build_info.items():
#     print(f'{k}:\t{v}')

# Plotting
DO_PLOT = True
FIGSIZE = (12, 8)
indir = './input/'
outdir = './output/'
n_of_units = 1000
units_per_pixel = n_of_units/1552.77 # length of scale bar in pixels

# Unet model
unet = keras.saving.load_model(
    'pretrained_model.keras',
    custom_objects={'weighted_crossentropy': segmenteverygrain.weighted_crossentropy})
# Segment Anything model
sam = segment_anything.sam_model_registry['default'](checkpoint='./sam_vit_h_4b8939.pth')
    # TODO: Figure out TensorRT to accelerate this w/GPU; currently it's actually slower on CUDA
    # sam.to(device='cuda')

# Main loop
fnames = sorted(glob(indir + '*.jpeg'))
print(f'\nFound {len(fnames)} image(s). Segmenting...\n')
for fname in fnames:
    # Load image
    print(f'\n--- {fname} ---')
    image = np.array(keras.utils.load_img(fname))
    outname = outdir + fname.split('/')[-1].split('.')[-2] + '_'
    
    # Unet ---
    # Generate prompts
    print('UNET ---')
    image_pred = segmenteverygrain.predict_image(image, unet, I=256)
    unet_labels, coords = segmenteverygrain.label_grains(image, image_pred, dbs_max_dist=20.0)
    # Save unet results for verification
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_aspect('equal')
    ax.imshow(image_pred)
    plt.scatter(np.array(coords)[:,0], np.array(coords)[:,1], c='k')
    plt.xticks([])
    plt.yticks([])
    fig.savefig(outname + 'unet.jpg')
    plt.close()

    # SAM ---
    print('SAM ---')
    # TODO: Separate this function into smaller chunks (plotting, mask, etc)
    # TODO: Choose min_area by image size? Do unit conversion from pixels first?
    all_grains, sam_labels, mask_all, grain_data, fig, ax = segmenteverygrain.sam_segmentation(
        sam, image, image_pred, coords, unet_labels,
        min_area=400.0, plot_image=True, remove_edge_grains=False, remove_large_objects=False)
    fig.savefig(outname + 'sam.jpg')
    plt.close()

    # Results ---
    print('Saving...')
    # TODO: Convert from pixels to real units
    for col in ['major_axis_length', 'minor_axis_length', 'perimeter', 'area']:
        grain_data[col] *= units_per_pixel
    # csv
    grain_data.to_csv(outname + 'sam.csv')
    # Histogram
    fig, ax = segmenteverygrain.plot_histogram_of_axis_lengths(grain_data['major_axis_length']/1000, grain_data['minor_axis_length']/1000)
    fig.savefig(outname + 'histogram.jpg')
    # Mask and image
    rasterized_image, mask = segmenteverygrain.create_labeled_image(all_grains, image)
    outname += '_sam_'
    # TODO: Remove opencv dependency?
    cv2.imwrite(outname + 'mask.png', mask)
    cv2.imwrite(outname + 'verify.png', mask*127)
    cv2.imwrite(outname + 'image.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

print(f'\nProcessed {len(fnames)} images!\n')