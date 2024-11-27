import cv2
import keras.saving
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import segment_anything
import segmenteverygrain
import segmenteverygrain.interactions as si

# Plotting
FIGSIZE = (12, 8)

# Load image and masks
fname = ''
image = None
all_grains = None
grain_data = None
predictor = None

# Create and display interactive grain plot
grains = [si.Grain(p.exterior.xy, row[1]) for p, row in zip(all_grains, grain_data.iterrows())]
grain_plot = si.GrainPlot(grains, image=image, predictor=predictor, figsize=FIGSIZE)
grain_plot.activate()
plt.show()

# Save final state
grain_plot.deactivate()
plt.savefig('.jpg')

# Get grain data as pd.DataFrame
new_grain_data = grain_plot.get_data()
plt.close()
# print(new_grain_data.head())

# TODO: Convert from pixels to real units
n_of_units = 1000
units_per_pixel = n_of_units/1552.77 # length of scale bar in pixels
for col in ['major_axis_length', 'minor_axis_length', 'perimeter', 'area']:
    new_grain_data[col] *= units_per_pixel

# Save csv and histogram
new_grain_data.to_csv(fname[:-4] + '.csv')
fig, ax = segmenteverygrain.plot_histogram_of_axis_lengths(new_grain_data['major_axis_length']/1000, new_grain_data['minor_axis_length']/1000)
fig.savefig('_edited_histogram.jpg')

# Save mask and original image for training
rasterized_image, mask = grain_plot.get_mask()
# TODO: Remove opencv dependency?
outname += '_man_'
cv2.imwrite(outname + 'mask.png', mask)
cv2.imwrite(outname + 'verify.png', mask*127)
# cv2.imwrite(outname + 'image.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))