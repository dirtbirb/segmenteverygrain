import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import segment_anything
import segmenteverygrain
import segmenteverygrain.interactions as si
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DBS_MAX_DIST = 20.0 # px
I = 256             # px?
MIN_AREA = 400      # px
PX_PER_M = 33.9     # px/m


# Load image and define output filename stem
fn = 'examples/torrey_pines.jpeg'
out_fn = 'examples/auto/torrey_pines'
logger.info(f'Loading image {fn}')
image = si.load_image(fn)

# Load U-Net model
fn = './segmenteverygrain/seg_model.keras'
logger.info(f'Loading U-Net model {fn}')
unet = keras.saving.load_model(fn, custom_objects={
    'weighted_crossentropy': segmenteverygrain.weighted_crossentropy})

# Generate prompts with U-Net model
logger.info(f'Generating prompts with U-Net model')
unet_image = segmenteverygrain.predict_image(image, unet, I=I)
unet_labels, unet_coords = segmenteverygrain.label_grains(
    image, unet_image, dbs_max_dist=DBS_MAX_DIST)

# Save U-Net diagnostic image
logger.info(f'Saving U-Net diagnostic image as {out_fn}')
fig, ax = plt.subplots(figsize=(15,10))
ax.imshow(unet_image)
plt.scatter(np.array(unet_coords)[:,0], np.array(unet_coords)[:,1], c='k')
plt.xticks([])
plt.yticks([])
fig.savefig(out_fn + '_unet.jpg', bbox_inches='tight', pad_inches=0)
plt.close()


# Load SAM
fn = 'sam_vit_h_4b8939.pth'
logger.info(f'Loading Segment Anything model with checkpoints from {fn}')
sam = segment_anything.sam_model_registry['default'](checkpoint=fn)
predictor = segment_anything.SamPredictor(sam)
predictor.set_image(image)

# Apply SAM for actual segmentation
logger.info(f'Segmenting image')
polygons, sam_labels, mask, summary, fig, ax = segmenteverygrain.sam_segmentation(
    sam, image, unet_image, unet_coords, unet_labels,
    min_area=MIN_AREA,
    plot_image=True,
    remove_edge_grains=False,
    remove_large_objects=False)


# Extract results
logger.info(f'Measuring grains')
grains = si.polygons_to_grains(polygons)
for g in tqdm(grains):
    g.measure(image=image)


# Save results
# NOTE: U-Net image was already saved, in case of error during SAM step
# NOTE: out_fn was defined at image import step
logger.info(f'Saving results as {out_fn}')
# Grain shapes
si.save_grains(out_fn + '_grains.geojson', grains)
# Grain image
fig.savefig(out_fn + '_grains.jpg', bbox_inches='tight', pad_inches=0)
# Summary data
si.save_summary(out_fn + '_summary.csv', grains, px_per_m=PX_PER_M)
# Summary histogram
si.save_histogram(out_fn + '_summary.jpg', grains, px_per_m=PX_PER_M)
# Training mask
si.save_mask(out_fn + '_mask.png', grains, image, scale=False)
si.save_mask(out_fn + '_mask2.jpg', grains, image, scale=True)


logger.info(f'Auto-segmenting complete!')