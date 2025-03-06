import matplotlib.pyplot as plt
import segment_anything
import segmenteverygrain.interactions as si

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


FIGSIZE = (12, 8)   # in
PX_PER_M = 1        # px/m; be sure not to convert units twice!


# Load test image
fn = 'torrey_pines_beach_image.jpeg'
# fn = 'input/Waimea_2m_611-617_largestjpng.png'
logger.info(f'Loading image {fn}')
image = si.load_image(fn)

# Load SAM
fn = 'sam_vit_h_4b8939.pth'
logger.info(f'Loading SAM with checkpoint {fn}')
sam = segment_anything.sam_model_registry['default'](checkpoint=fn)
predictor = segment_anything.SamPredictor(sam)
logger.info('Setting image predictor')
predictor.set_image(image)

# Load grains
fn = './output/test_edit_grains.geojson'
# fn = 'input/Waimea_2m_611-617_largestjpng_grains.geojson'
logger.info(f'Loading grains from {fn}')
grains = si.load_grains(fn)
# grains = []


# Display editing interface
plot = si.GrainPlot(
    grains, 
    image=image, 
    predictor=predictor,
    blit=True,
    figsize=FIGSIZE,
    image_max_size=(240, 320)
)
plot.activate()
plt.show(block=True)
plot.deactivate()


# Save results
fn = './output/test_edit'
logger.info(f'Saving results as {fn}')
# Convert units
pass
# Grain shapes
# for g in grains:
#     g.measure(image=image)
grains = plot.grains
si.save_grains(fn + '_grains.geojson', grains)
# Grain image
plot.savefig(fn + '_grains.jpg')
# Summary data
si.save_summary(fn + '_summary.csv', grains, px_per_m=PX_PER_M)
# Summary histogram
si.save_histogram(fn + '_summary.jpg', grains, px_per_m=PX_PER_M)
# Training mask
si.save_mask(fn + '_mask.png', grains, image, scale=False)
si.save_mask(fn + '_mask.jpg', grains, image, scale=True)

logger.info(f'Saving complete!')