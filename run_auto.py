import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import segment_anything
import segmenteverygrain
import segmenteverygrain.interactions as si


DBS_MAX_DIST = 20.0 # px
I = 256             # px?
MIN_AREA = 400      # px
PX_PER_M = 1        # px/m


# Load test image
fn = 'torrey_pines_beach_image.jpeg'
image = si.load_image(fn)

# Load unet model
fn = './segmenteverygrain/seg_model.keras'
unet = keras.saving.load_model(fn, custom_objects={
    'weighted_crossentropy': segmenteverygrain.weighted_crossentropy})

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
plt.close()


# Load SAM
fn = 'sam_vit_h_4b8939.pth'
sam = segment_anything.sam_model_registry['default'](checkpoint=fn)
predictor = segment_anything.SamPredictor(sam)
predictor.set_image(image)

# Apply SAM for actual segmentation
grains, sam_labels, mask, summary, fig, ax = segmenteverygrain.sam_segmentation(
    sam, image, unet_image, unet_coords, unet_labels,
    min_area=MIN_AREA,
    plot_image=True,
    remove_edge_grains=False,
    remove_large_objects=False)


# Extract results
grains = [si.Grain(np.array(g.exterior.xy)) for g in grains]
for g in grains:
    g.measure(image=image)
fn = './output/test_auto'
# Grain shapes
si.save_grains(fn + '_grains.geojson', grains)
# Grain image
fig.savefig(fn + '_grains.jpg', bbox_inches='tight', pad_inches=0)
# Summary data
si.save_summary(fn + '_summary.csv', grains, px_per_m=PX_PER_M)
# Summary histogram
si.save_histogram(fn + '_summary.jpg', grains, px_per_m=PX_PER_M)
# Training mask
si.save_mask(fn + '_mask.png', grains, image, scale=False)
si.save_mask(fn + '_mask.jpg', grains, image, scale=True)