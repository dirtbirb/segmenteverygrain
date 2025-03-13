import matplotlib.pyplot as plt
import segment_anything
import segmenteverygrain.interactions as si
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIGSIZE = (12, 8)                   # in
PX_PER_M = 33.9                     # px/m
IMAGE_MAX_SIZE = (2160, 4096)       # (y: px, x: px)
IMAGE_ALPHA = 1.


# Load image
fn = 'examples/torrey_pines.jpeg'
logger.info(f'Loading image {fn}')
image = si.load_image(fn)

# Load grains
fn = 'examples/auto/torrey_pines_grains.geojson'
logger.info(f'Loading grains from {fn}')
grains = si.load_grains(fn)
# If not loading any grains, use this line instead:
# grains = []

# # Filter grains (in pixels). Options:
# #   area
# #   centroid
# #   major_axis_length
# #   minor_axis_length 
# #   orientation
# #   perimeter
# #   max_intensity
# #   mean_intensity
# #   min_intensity
#
# for g in grains:
#     g.measure(image=image)
# min_area = 1                              # m^2
# min_area_px = min_area * PX_PER_M ** 2    # px^2
# grains = [g for g in grains if g.data['area'] > min_area_px]


# Load SAM
fn = 'sam_vit_h_4b8939.pth'
logger.info(f'Loading SAM with checkpoint {fn}')
sam = segment_anything.sam_model_registry['default'](checkpoint=fn)
predictor = segment_anything.SamPredictor(sam)
logger.info('Setting image predictor')
predictor.set_image(image)


# Display interactive interface
plot = si.GrainPlot(
    grains, 
    image = image, 
    predictor = predictor,
    blit = True,
    figsize = FIGSIZE,
    px_per_m = PX_PER_M,
    image_max_size = IMAGE_MAX_SIZE,
    image_alpha = IMAGE_ALPHA
)
plot.activate()
plt.show(block=True)
plot.deactivate()


# Get updated grains (with proper scale) and unit conversion (if changed)
grains = plot.grains
px_per_m = plot.px_per_m

# Add grain axes to plot
logger.info('Plotting grain axes')
for grain in tqdm(grains):
    grain.draw_axes(plot.ax)

# Save results
fn = 'examples/interactive/torrey_pines'
logger.info(f'Saving results as {fn}')
# Grain shapes
si.save_grains(fn + '_grains.geojson', grains)
# Grain image
plot.savefig(fn + '_grains.jpg')
# Summary data
si.save_summary(fn + '_summary.csv', grains, px_per_m=px_per_m)
# Summary histogram
si.save_histogram(fn + '_summary.jpg', grains, px_per_m=px_per_m)
# Training mask
si.save_mask(fn + '_mask.png', grains, image, scale=False)
si.save_mask(fn + '_mask2.jpg', grains, image, scale=True)

logger.info(f'Saving complete!')