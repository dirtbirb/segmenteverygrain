# Setup ---

import matplotlib
# HACK: Can't use qt due to conflict between matplotlib and opencv/keras
matplotlib.use('gtk4agg')

import cv2
import glob
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import segment_anything
import segmenteverygrain
import segmenteverygrain.interactions as si

FIGSIZE = (12, 8)
img_dir = '1_input/'
in_dir = '2_sam_results/'
out_dir = '3_edited_results/'

sam = segment_anything.sam_model_registry["default"](checkpoint="./sam_vit_h_4b8939.pth")


# Processing ---

fnames = sorted(glob.glob(img_dir + '*.jpg'))
print(f'\nFound {len(fnames)} image(s). Editing...\n')
for fn in fnames:
    print(f'--- {fn} ---')
    # Load image
    print('Loading image...')
    image = np.array(keras.utils.load_img(fn))
    fn = in_dir + fn.split('/')[-1].split('.')[-2]
    # Rebuild all_grains
    print('Rebuilding all_grains...')
    # HACK: Parse accidental string output
    all_grains = []
    for grain in pd.read_csv(fn + '_sam_grains.csv').iterrows():
        out_coords = []
        for coord in grain[1].iloc[1][10:-2].split(', '):
            x, y = coord.split(' ')
            out_coords.append((float(x), float(y)))
        all_grains.append(shapely.Polygon(out_coords))
    # Rebuild grain_data
    print('Rebuilding grain_data...')
    grain_data = pd.read_csv(fn + '_sam_summary.csv').drop('Unnamed: 0', axis=1)
    # Load SAM predictor
    print('Preparing SAM predictor...')
    predictor = segment_anything.SamPredictor(sam)
    predictor.set_image(image)

    # Interactive plot
    print('Displaying plot...')
    grains = [si.Grain(p.exterior.xy, row[1]) for p, row in zip(all_grains, grain_data.iterrows())]
    grain_plot = si.GrainPlot(grains, image=image, predictor=predictor, figsize=FIGSIZE)
    grain_plot.activate()
    with plt.ion():
        plt.show(block=True)
    grain_plot.deactivate()

    # Save figure
    print('Saving...')
    fn = out_dir + fn.split('/')[-1] + '_edited'
    grain_plot.fig.savefig(fn + '.jpg')
    # Get new grain_data and all_grains
    new_all_grains = [g.get_polygon() for g in grain_plot.grains]
    new_grain_data = grain_plot.get_data()
    plt.close(grain_plot.fig)
    # Convert units
    # TODO: Convert from pixels to real units
    n_of_units = 1000
    units_per_pixel = n_of_units/1552.77 # length of scale bar in pixels
    for col in ['major_axis_length', 'minor_axis_length', 'perimeter', 'area']:
        new_grain_data[col] *= units_per_pixel
    new_grain_data.to_csv(fn + '_summary.csv')
    pd.DataFrame(new_all_grains).to_csv(fn + '_grains.csv')
    # Save CSVs and histogram
    fig, ax = segmenteverygrain.plot_histogram_of_axis_lengths(new_grain_data['major_axis_length']/1000, new_grain_data['minor_axis_length']/1000)
    fig.savefig(fn + '_histogram.jpg')
    plt.close(fig)
    # Save mask for Unet training
    rasterized_image, mask = grain_plot.get_mask()
    cv2.imwrite(fn + '_mask.png', mask)
    cv2.imwrite(fn + '_mask_visible.png', mask*127)

print('\nDone!\n')