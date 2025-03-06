import matplotlib.pyplot as plt
import numpy as np
import segmenteverygrain.interactions as si
import shapely


FIGURE_DPI = 72     # dots/inch
PX_PER_M = 1        # px/m; be sure not to convert units twice!!
SPACING = 220       # px


# Load image
fn = 'torrey_pines_beach_image.jpeg'
image = si.load_image(fn)
img_y, img_x = image.shape[:2]

# Load grains
fn = './output/test_edit_grains.geojson'
grains = si.load_grains(fn)


# Find and measure grains according to a grid
points, xs, ys = si.make_grid(image, SPACING)
grains, points_found = si.filter_grains_by_points(grains, points)
for g in grains:
    g.measure(image=image)


# Get GrainPlot as a static image
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

# Plot axes
for grain in grains:
    grain.rescale(1 / plot.scale)
    grain.draw_axes(ax)

# Plot count grid
point_colors = ['lime' if p else 'red' for p in points_found]
ax.scatter(xs, ys,
    s=min(plot_image[:2].shape) * FIGURE_DPI,
    c=point_colors,
    edgecolors='black')


# Save results
fn = './output/test_count'
# Convert units
pass
# Grain shapes
si.save_grains(fn + '_grains.geojson', grains)
# Grain image
fig.savefig(fn + '_grains.jpg', bbox_inches='tight', pad_inches=0.2)
# Summary data
si.save_summary(fn + '_summary.csv', grains, px_per_m=PX_PER_M)
# Summary histogram
si.save_histogram(fn + '_summary.jpg', grains, px_per_m=PX_PER_M)


# plt.show(block=True)