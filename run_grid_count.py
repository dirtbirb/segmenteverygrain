import matplotlib.pyplot as plt
import numpy as np
import segmenteverygrain.interactions as si
import shapely
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIGURE_DPI = 72                     # dots/inch
PX_PER_M = 33.9                     # px/m
SPACING_M = 0.155                   # m
SPACING_PX = PX_PER_M / SPACING_M   # px


# Load image
fn = 'examples/torrey_pines.jpeg'
logger.info(f'Loading image {fn}')
image = si.load_image(fn)
img_y, img_x = image.shape[:2]

# Load grains
fn = 'examples/interactive/torrey_pines_grains.geojson'
logger.info(f'Loading grains from {fn}')
grains = si.load_grains(fn)


# Find and measure grains according to a grid
logger.info('Constructing grid')
points, xs, ys = si.make_grid(image, SPACING_PX)
logger.info(f'Generated {len(points)} measurement points.')
grains, points_found = si.filter_grains_by_points(grains, points)
logger.info('Measuring selected grains')
for g in tqdm(grains):
    g.measure(image=image)


# Get GrainPlot as a static image
logger.info('Creating plot')
plot = si.GrainPlot(grains, image, 
    figsize = (img_x / FIGURE_DPI, img_y / FIGURE_DPI), 
    dpi = FIGURE_DPI,
    px_per_m = PX_PER_M,
    image_alpha = 0.5)
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
    grain.draw_axes(ax)

# Plot grid
ax.scatter(xs, ys,
    s=min(plot_image[:2].shape) * FIGURE_DPI,
    c=['lime' if p else 'red' for p in points_found],
    edgecolors='black')


# Save results
fn = 'examples/grid_count/torrey_pines'
logger.info(f'Saving results as {fn}')
# Grain shapes
grains = plot.grains
si.save_grains(fn + '_grains.geojson', grains)
# Grain image
fig.savefig(fn + '_grains.jpg', bbox_inches='tight', pad_inches=0.2)
# Summary data
si.save_summary(fn + '_summary.csv', grains, px_per_m=PX_PER_M)
# Summary histogram
si.save_histogram(fn + '_summary.jpg', grains, px_per_m=PX_PER_M)


logger.info(f'Grid count complete!')