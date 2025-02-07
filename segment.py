import cv2
import glob
import keras.saving
import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import shapely
import segment_anything
import segmenteverygrain
import segmenteverygrain.interactions as si

import kivy
kivy.require('2.3.1')
# TODO: fix matplot / kivy log levels
from kivy.logger import Logger, LOG_LEVELS
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.properties import ObjectProperty, StringProperty
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.app import App

# plt.set_loglevel('error')
FIGSIZE = (12, 8)
sam = segment_anything.sam_model_registry["default"](checkpoint="./sam_vit_h_4b8939.pth")
predictor = segment_anything.SamPredictor(sam)


class LoadDialog(BoxLayout):
    load = ObjectProperty()
    cancel = ObjectProperty()


class SaveDialog(BoxLayout):
    save = ObjectProperty()
    text_input = ObjectProperty()
    cancel = ObjectProperty()


class RootLayout(BoxLayout):
    figure = ObjectProperty()
    unet_model = ObjectProperty()
    unet_fn = StringProperty()
    image = ObjectProperty()
    image_fn = StringProperty()
    grains = ObjectProperty([])
    grains_fn = StringProperty()
    summary = ObjectProperty([])
    summary_fn = StringProperty()

    # Segmentation -----------------------------------------------------------
    def auto_segment(self):
        pass

    def manual_segment(self):
        # Display editing interface
        Logger.info('Displaying plot...')
        self.figure = si.GrainPlot(
            self.grains, 
            image=self.image, 
            predictor=predictor, 
            figsize=FIGSIZE
        )
        self.figure.activate()
        with plt.ion():
            plt.show(block=True)
        self.figure.deactivate()

        # Show save dialog
        Logger.info('Saving...')
        self.show_save()

    # Save/load --------------------------------------------------------------
    def load_data(self, path, filename):
        # Load grain data csv
        Logger.info('Loading grains...')
        # HACK: Parse accidental string output
        grains = []
        for grain in pd.read_csv(filename).iterrows():
            out_coords = []
            for coord in grain[1].iloc[1][10:-2].split(', '):
                x, y = coord.split(' ')
                out_coords.append((float(x), float(y)))
            grains.append(shapely.Polygon(out_coords))
        self.grains = grains
        self.grains_fn = os.path.basename(filename)
        Logger.info(f'Loaded {self.grains_fn}.')       
        self._popup.dismiss()
        # Trigger "load summary" dialog immediately afterward
        dialog = LoadDialog(load=self.load_summary_data, cancel=self.dismiss_popup)
        self.show_dialog(dialog, title='Load summary data', filters=['*.csv'])

    def load_summary_data(self, path, filename):
        ''' Triggered after loading grains '''
        Logger.info('Loading data...')
        self.summary = pd.read_csv(filename).drop('Unnamed: 0', axis=1)
        self.summary_fn = os.path.basename(filename)
        self.grains = [si.Grain(p.exterior.xy, row[1]) for p, row in zip(self.grains, self.summary.iterrows())]
        Logger.info(f'Loaded {self.summary_fn}.')  
        self._popup.dismiss()

    def load_image(self, path, filename):
        Logger.info('Loading image...')
        self.image = np.array(keras.utils.load_img(filename))
        self.image_fn = os.path.basename(filename)
        Logger.info('Preparing SAM predictor...')
        predictor.set_image(self.image)
        Logger.info(f'Loaded {self.image_fn}.')
        self._popup.dismiss()

    def load_unet(self, path, filename):
        weights = segmenteverygrain.weighted_crossentropy
        self.unet_model = keras.saving.load_model(
            filename, custom_objects={'weighted_crossentropy': weights}
        )
        self.unet_fn = os.path.basename(filename)
        self._popup.dismiss()

    def save(self):
        pass
        # # Save image with highlighted grains
        # fn = out_dir + fn.split('/')[-1] + '_edited'
        # grain_plot.fig.savefig(fn + '.jpg')
        
        # # Get new grain_data and all_grains
        # new_all_grains = [g.get_polygon() for g in grain_plot.grains]
        # new_grain_data = grain_plot.get_data()
        # plt.close(grain_plot.fig)
        
        # # Convert units
        # # TODO: Convert from pixels to real units
        # n_of_units = 1000
        # units_per_pixel = n_of_units/1552.77 # length of scale bar in pixels
        # for col in ['major_axis_length', 'minor_axis_length', 'perimeter', 'area']:
        #     new_grain_data[col] *= units_per_pixel
        # new_grain_data.to_csv(fn + '_summary.csv')
        # pd.DataFrame(new_all_grains).to_csv(fn + '_grains.csv')
        
        # # Save CSVs and histogram
        # fig, ax = segmenteverygrain.plot_histogram_of_axis_lengths(new_grain_data['major_axis_length']/1000, new_grain_data['minor_axis_length']/1000)
        # fig.savefig(fn + '_histogram.jpg')
        # plt.close(fig)
        
        # # Save mask for Unet training
        # rasterized_image, mask = grain_plot.get_mask()
        # cv2.imwrite(fn + '_mask.png', mask)
        # cv2.imwrite(fn + '_mask_visible.png', mask*127)

    # Popups -----------------------------------------------------------------
    def dismiss_popup(self):
        self._popup.dismiss()

    def show_dialog(self, dialog, title='', filters=[]):
        dialog.ids.filechooser.filters = filters
        self._popup = Popup(title=title, content=dialog)
        self._popup.open()

    def show_load_image(self):
        plt.close()
        dialog = LoadDialog(load=self.load_image, cancel=self.dismiss_popup)
        self.show_dialog(dialog, title='Load image', filters=['*.jpg', '*.jpeg', '*.png'])

    def show_load_unet(self):
        dialog = LoadDialog(load=self.load_unet, cancel=self.dismiss_popup)
        self.show_dialog(dialog, title='Load Unet model', filters=['*.keras'])

    def show_load_data(self):
        # Triggers both "load grains" and "load summary" in sequence
        dialog = LoadDialog(load=self.load_data, cancel=self.dismiss_popup)
        self.show_dialog(dialog, title='Load grain data', filters=['*.csv'])

    def show_save(self):
        dialog = SaveDialog(save=self.save, cancel=self.dismiss_popup)
        self.show_dialog(dialog, title='Save results')


class SegmentApp(App):
    ''' Layout defined in segment.kv '''
    pass


# Logger.setLevel(LOG_LEVELS['error'])
SegmentApp().run()
