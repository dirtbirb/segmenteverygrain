# Matplotlib backend
import matplotlib
matplotlib.use('agg')

# Use non-interactive matplot backend, doesn't depend on installed GUI kits
import matplotlib
matplotlib.use('agg')

# Imports
# TODO: Silence torch/tensorflow/whatever warnings on import
import cv2
import glob
import keras.saving
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segment_anything
import segmenteverygrain

# import tensorflow as tf
# from tensorflow.python.platform.build_info import build_info
# for k, v in build_info.items():
#     print(f'{k}:\t{v}')

# Options
in_dir = './1_input/'
out_dir = './2_sam_results/'
DO_PLOT = True
FIGSIZE = (12, 8)
MIN_AREA = 400
n_of_units = 1000
units_per_pixel = n_of_units/1552.77 # length of scale bar in pixels

# Load Unet model
unet = keras.saving.load_model(
    'seg_model.keras',
    custom_objects={'weighted_crossentropy': segmenteverygrain.weighted_crossentropy})

# Load Segment Anything model
# TODO: Figure out TensorRT to accelerate this w/GPU; currently it's actually slower on CUDA
# sam.to(device='cuda')
sam = segment_anything.sam_model_registry['default'](checkpoint='./sam_vit_h_4b8939.pth')

# Main loop
fnames = sorted(glob.glob(in_dir + '*.jpg'))
print(f'\nFound {len(fnames)} image(s). Segmenting...\n')
for fn in fnames:
    # Load image
    print(f'\n--- {fn} ---')
    image = np.array(keras.utils.load_img(fn))
    outname = out_dir + fn.split('/')[-1].split('.')[-2]
    
    # Unet ---
    # Generate prompts
    print('UNET -')
    image_pred = segmenteverygrain.predict_image(image, unet, I=256)
    unet_labels, coords = segmenteverygrain.label_grains(image, image_pred, dbs_max_dist=20.0)
    # Save unet results for verification
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_aspect('equal')
    ax.imshow(image_pred)
    plt.scatter(np.array(coords)[:,0], np.array(coords)[:,1], c='k')
    ax.set(xticks=[], yticks=[])
    fig.savefig(outname + '_unet.jpg')
    plt.close(fig)

    # SAM ---
    print('SAM ---')
    outname += '_sam_'
    # TODO: Separate this function into smaller chunks (plotting, mask, etc)
    # TODO: Choose min_area by image size? Do unit conversion from pixels first?
    all_grains, sam_labels, mask_all, grain_data, fig, ax = segmenteverygrain.sam_segmentation(
        sam, image, image_pred, coords, unet_labels,
        min_area=MIN_AREA, plot_image=False, remove_edge_grains=False, remove_large_objects=False)
    plt.close(fig)

    # Results ---
    print('Saving...')
    # Labeled image
    cv2.imwrite(sam_labels, outname + 'labels.jpg')
    # TODO: Convert from pixels to real units
    for col in ['major_axis_length', 'minor_axis_length', 'perimeter', 'area']:
        grain_data[col] *= units_per_pixel
    # CSV
    grain_data.to_csv(outname + 'summary.csv')
    # TODO: Convert to lists before saving, not just shapely.Polygon
    pd.DataFrame(all_grains).to_csv(outname + 'grains.csv')
    # Histogram
    fig, ax = segmenteverygrain.plot_histogram_of_axis_lengths(grain_data['major_axis_length']/1000, grain_data['minor_axis_length']/1000)
    fig.savefig(outname + 'histogram.jpg')
    plt.close(fig)
    # Mask
    rasterized_image, mask = segmenteverygrain.create_labeled_image(all_grains, image)
    # TODO: Remove opencv dependency?
    cv2.imwrite(outname + 'mask.png', mask)
    cv2.imwrite(outname + 'mask_visible.png', mask*127)

print(f'\nProcessed {len(fnames)} images!\n')