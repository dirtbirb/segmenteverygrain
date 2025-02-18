import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segment_anything
import segmenteverygrain


DBS_MAX_DIST = 20.0
I = 256
MIN_AREA = 400


# Load test image
fn = 'torrey_pines_beach_image.jpeg'
image = np.array(keras.utils.load_img(fn))

# Load unet model
fn = './segmenteverygrain/seg_model.keras'
unet = keras.saving.load_model(
    fn, 
    custom_objects={
        'weighted_crossentropy': segmenteverygrain.weighted_crossentropy
    }
)

# Generate prompts with UNET model
unet_image = segmenteverygrain.predict_image(image, unet, I=I)
unet_labels, unet_coords = segmenteverygrain.label_grains(
    image, unet_image, dbs_max_dist=DBS_MAX_DIST)

# Save unet diagnostic image
fig, ax = plt.subplots(figsize=(15,10))
ax.imshow(unet_image)
plt.scatter(np.array(unet_coords)[:,0], np.array(unet_coords)[:,1], c='k')
plt.xticks([])
plt.yticks([])
fig.savefig('./output/test_unet.jpg', bbox_inches='tight', pad_inches=0)


# Load SAM
fn = 'sam_vit_h_4b8939.pth'
sam = segment_anything.sam_model_registry['default'](checkpoint=fn)
predictor = segment_anything.SamPredictor(sam)
predictor.set_image(image)

# Apply SAM for actual segmentation
# TODO: What is sam_labels?
grains, sam_labels, mask, summary, fig, ax = segmenteverygrain.sam_segmentation(
    sam, image, unet_image, unet_coords, unet_labels,
    min_area=MIN_AREA, plot_image=True, remove_edge_grains=True, remove_large_objects=False
)



# Save results
fn = './output/test'

# Grain shapes
print(f'grains: {type(grains)} {grains}')
pd.DataFrame(grains).to_csv(fn + '_grains.csv')

# Grain image
fig.savefig(fn + '_grains.jpg', bbox_inches='tight', pad_inches=0)

# Summary data
grain_data = summary
print(f'grain_data: {type(grain_data)} {grain_data}')
grain_data.to_csv(fn + '_summary.csv')

# Summary histogram
fig, ax = segmenteverygrain.plot_histogram_of_axis_lengths(
    grain_data['major_axis_length']/1000, 
    grain_data['minor_axis_length']/1000)
fig.savefig(fn + '_summary.jpg', bbox_inches='tight', pad_inches=0)

# Training mask
mask = keras.utils.img_to_array(mask)
keras.utils.save_img(fn + '_mask.png', mask, scale=False)
keras.utils.save_img(fn + '_mask.jpg', mask, scale=True)