import matplotlib.pyplot as plt
import numpy as np
import segmenteverygrain.interactions as si
import shapely
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIGURE_DPI = 72     # dots/inch
PX_PER_M = 1        # px/m
SPACING = 220       # px


# Load image
fn = 'torrey_pines_beach_image.jpeg'
logger.info(f'Loading image {fn}')
image = si.load_image(fn)
img_y, img_x = image.shape[:2]

# Load grains
fn = './output/test_edit_grains.geojson'
logger.info(f'Loading grains from {fn}')
grains = si.load_grains(fn)


# Find and measure grains according to a grid
logger.info('Picking grains at grid locations')
points, xs, ys = si.make_grid(image, SPACING)
grains, points_found = si.filter_grains_by_points(grains, points)
logger.info('Measuring picked grains')
for g in tqdm(grains):
    g.measure(image=image)


# Get GrainPlot as a static image
logger.info('Creating plot')
plot = si.GrainPlot(grains, image, 
    figsize=(img_x/FIGURE_DPI, img_y/FIGURE_DPI), 
    dpi=FIGURE_DPI,
    image_alpha=0.5)
plot_image = np.asarray(plot.canvas.buffer_rgba(), dtype=np.uint8)

# Make new plot using static GrainPlot image as background
fig = plt.figure(
    figsize=(img_x/FIGURE_DPI, img_y/FIGURE_DPI),
    dpi=FIGURE_DPI)
ax = fig.add_subplot()
ax.imshow(plot_image, aspect='equal', origin='lower')
ax.autoscale(enable=False)

# Plot grain axes
for grain in tqdm(grains):
    grain.rescale(1 / plot.scale)
    grain.draw_axes(ax)

# Plot grid
point_colors = ['lime' if p else 'red' for p in points_found]
ax.scatter(xs, ys,
    s=min(plot_image[:2].shape) * FIGURE_DPI,
    c=point_colors,
    edgecolors='black')


# Save results
fn = './output/test_count'
logger.info(f'Saving results as {fn}')
# Grain shapes
si.save_grains(fn + '_grains.geojson', grains)
# Grain image
fig.savefig(fn + '_grains.jpg', bbox_inches='tight', pad_inches=0.2)
# Summary data
si.save_summary(fn + '_summary.csv', grains, px_per_m=PX_PER_M)
# Summary histogram
si.save_histogram(fn + '_summary.jpg', grains, px_per_m=PX_PER_M)


logger.info(f'Grid count complete!')