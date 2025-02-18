import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segment_anything
import segmenteverygrain
import segmenteverygrain.interactions as si
import shapely


FIGSIZE = (12, 8)


# Load test image
fn = 'torrey_pines_beach_image.jpeg'
image = np.array(keras.utils.load_img(fn))

# Load previous results
fn = './output/test_grains.csv'
grains = []
for grain in pd.read_csv(fn).iterrows():
    out_coords = []
    for coord in grain[1].iloc[1][10:-2].split(', '):
        x, y = coord.split(' ')
        out_coords.append((float(x), float(y)))
    grains.append(shapely.Polygon(out_coords))
grains = [si.Grain(p.exterior.xy) for p in grains]

# Load SAM
fn = 'sam_vit_h_4b8939.pth'
sam = segment_anything.sam_model_registry['default'](checkpoint=fn)
predictor = segment_anything.SamPredictor(sam)
predictor.set_image(image)


# Display editing interface
plot = si.GrainPlot(
    grains, 
    image=image, 
    predictor=predictor,
    figsize=FIGSIZE
)
plot.activate()
with plt.ion():
    plt.show(block=True)
plot.deactivate()


# # Save results
# fn = './output/test'

# # Grain shapes
# pd.DataFrame(
#     [g.get_polygon() for g in plot.grains]
# ).to_csv(fn + '_grains.csv')

# # Grain image
# plot.fig.savefig(fn + '_grains.jpg', bbox_inches='tight', pad_inches=0)

# # Summary data
# grain_data = plot.get_data()
# grain_data.to_csv(fn + '_summary.csv')

# # Summary histogram
# fig, ax = segmenteverygrain.plot_histogram_of_axis_lengths(
#     grain_data['major_axis_length']/1000, 
#     grain_data['minor_axis_length']/1000)
# fig.savefig(fn + '_summary.jpg', bbox_inches='tight', pad_inches=0)

# # Training mask
# rasterized_image, mask = plot.get_mask()
# mask = keras.utils.img_to_array(mask)
# keras.utils.save_img(fn + '_mask.png', mask, scale=False)
# keras.utils.save_img(fn + '_mask.jpg', mask, scale=True)