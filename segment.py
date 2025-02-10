import cv2
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
from kivy.logger import Logger, LOG_LEVELS
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.properties import ListProperty, ObjectProperty, StringProperty
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.app import App

# TODO: fix matplot / kivy log levels
# plt.set_loglevel('error')
FIGSIZE = (12, 8)


class LoadDialog(BoxLayout):
    load = ObjectProperty()
    cancel = ObjectProperty()


class SaveDialog(BoxLayout):
    save = ObjectProperty()
    path = StringProperty()
    filename = StringProperty()
    text_input = ObjectProperty()
    cancel = ObjectProperty()


class RootLayout(BoxLayout):
    # Internal properties
    plot = ObjectProperty()
    figure = ObjectProperty()
    predictor = ObjectProperty()
    _popup = ObjectProperty()
    # User settables
    grains = ListProperty()
    grains_fn = StringProperty()
    image = ObjectProperty()
    image_fn = StringProperty()
    sam = ObjectProperty()
    sam_checkpoint_fn = StringProperty()
    summary = ObjectProperty()
    summary_fn = StringProperty()
    unet_model = ObjectProperty()
    unet_fn = StringProperty()

    # Load defaults ----------------------------------------------------_-----
    def on_kv_post(self, widget):
        self.load_sam_checkpoint('', './sam_vit_h_4b8939.pth')
        self.load_unet('', './segmenteverygrain/seg_model.keras')

    # Segmentation -----------------------------------------------------------
    def auto_segment(self):
        # # Unet ---
        # # Generate prompts
        # print('UNET -')
        # image_pred = segmenteverygrain.predict_image(image, unet, I=256)
        # unet_labels, coords = segmenteverygrain.label_grains(image, image_pred, dbs_max_dist=20.0)

        # # SAM ---
        # print('SAM ---')
        # outname += '_sam_'
        # # TODO: Separate this function into smaller chunks (plotting, mask, etc)
        # # TODO: Choose min_area by image size? Do unit conversion from pixels first?
        # all_grains, sam_labels, mask_all, grain_data, fig, ax = segmenteverygrain.sam_segmentation(
        #     sam, image, image_pred, coords, unet_labels,
        #     min_area=MIN_AREA, plot_image=False, remove_edge_grains=False, remove_large_objects=False)
        # plt.close(fig)

        Logger.info('Saving...')
        self.show_auto_save()

    def manual_segment(self):
        # Prepare SAM predictor
        Logger.info('Preparing SAM predictor...')
        self.predictor.set_image(self.image)
        
        # Display editing interface
        Logger.info('Displaying interactive interface...')
        self.plot = si.GrainPlot(
            self.grains, 
            image=self.image, 
            predictor=self.predictor, 
            figsize=FIGSIZE
        )
        self.plot.activate()
        with plt.ion():
            plt.show(block=True)
        self.plot.deactivate()

        # Show save dialog
        self.show_save()

    # Save/load --------------------------------------------------------------
    def load_sam_checkpoint(self, path, filename):
        Logger.info('Loading checkpoint...')
        self.sam = segment_anything.sam_model_registry["default"](checkpoint=filename)
        self.sam_checkpoint_fn = os.path.basename(filename)
        self.predictor = segment_anything.SamPredictor(self.sam)
        self.dismiss_popup()
        Logger.info(f'Loaded checkpoint {self.sam_checkpoint_fn}.')

    def load_data(self, path, filename):
        # Load grain data
        self.load_grain_data(path, filename)
        # Trigger "load summary" dialog immediately afterward
        dialog = LoadDialog(load=self.load_summary_data, cancel=self.dismiss_popup)
        self.show_dialog(dialog, title='Load summary data', filters=['*.csv'])

    def load_grain_data(self, path, filename):
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
        self.dismiss_popup()
        Logger.info(f'Loaded {self.grains_fn}.')

    def load_image(self, path, filename):
        Logger.info('Loading image...')
        self.image = np.array(keras.utils.load_img(filename))
        self.ids.image.source = filename
        self.image_fn = os.path.basename(filename)
        self.dismiss_popup()
        Logger.info(f'Loaded {self.image_fn}.')

    def load_summary_data(self, path, filename):
        ''' Triggered after loading grains '''
        Logger.info('Loading data...')
        self.summary = pd.read_csv(filename).drop('Unnamed: 0', axis=1)
        self.summary_fn = os.path.basename(filename)
        self.grains = [si.Grain(p.exterior.xy, row[1]) for p, row in zip(self.grains, self.summary.iterrows())]
        self.dismiss_popup()
        Logger.info(f'Loaded {self.summary_fn}.')  

    def load_unet(self, path, filename):
        Logger.info('Loading unet model...')
        weights = segmenteverygrain.weighted_crossentropy
        self.unet_model = keras.saving.load_model(
            filename, custom_objects={'weighted_crossentropy': weights}
        )
        self.unet_fn = os.path.basename(filename)
        self.dismiss_popup()
        Logger.info(f'Loaded {self.unet_fn}.')  

    def save_grain_image(self, filename):
        Logger.info('Saving grain image...')
        self.ids.image.source = filename
        self.dismiss_popup()
        Logger.info(f'Saved {filename}.')

    def save_grains(self, filename):
        Logger.info('Saving grain data...')
        grains = [g.get_polygon() for g in self.plot.grains]
        pd.DataFrame(grains).to_csv(filename)
        Logger.info(f'Saved {filename}.')

    def save_summary(self, filename):
        Logger.info('Saving summary data...')
        # Get measurements from plot as a pd.DataFrame
        grain_data = self.plot.get_data()
        # Convert units
        # TODO: Convert from pixels to real units
        n_of_units = 1000
        units_per_pixel = n_of_units/1552.77 # length of scale bar in pixels
        for col in ['major_axis_length', 'minor_axis_length', 'perimeter', 'area']:
            grain_data[col] *= units_per_pixel
        # Save CSV
        grain_data.to_csv(filename)
        Logger.info(f'Saved {filename}.')
        # Build and save histogram
        filename = filename.split('.')[0] + '.jpg'
        fig, ax = segmenteverygrain.plot_histogram_of_axis_lengths(
            grain_data['major_axis_length']/1000, 
            grain_data['minor_axis_length']/1000)
        fig.savefig(filename)
        plt.close(fig)
        Logger.info(f'Saved {filename}.')

    def save_mask(self, filename):
        Logger.info('Saving mask...')
        rasterized_image, mask = self.plot.get_mask()
        # keras.utils.save_img(filename, mask)
        # keras.utils.save_img(filename.split('.')[0] + '_visible.png', mask*127)
        cv2.imwrite(filename, mask)
        Logger.info(f'Saved {filename}.')
        cv2.imwrite(filename.split('.')[0] + '_visible.png', mask*127)
        Logger.info(f'Saved {filename}.')

    def save_unet_image(self, filename):
        # Save unet results for verification (auto segmenting!)
        Logger.info(f'Saving unet image...')
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_aspect('equal')
        ax.imshow(image_pred)
        plt.scatter(np.array(coords)[:,0], np.array(coords)[:,1], c='k')
        ax.set(xticks=[], yticks=[])
        fig.savefig(outname + '_unet.jpg')
        plt.close(fig)
        Logger.info(f'Saved {filename}.')

    def save(self, path, filename):
        ''' 
        Save all results via save_whatever method for each type of data.
        '''
        Logger.info('\n--- Results ---')
        # Parse filename -- include path, remove extension
        filename = os.path.join(path, filename.split('.')[0])
        # Save results
        self.save_grains(filename + '_grains.csv')
        self.save_summary(filename + '_summary.csv')
        if self.unet_image:
            self.save_unet_image(filename + '_unet.jpg')
        self.save_mask(filename + '_mask.png')
        self.save_grain_image(filename + '_highlighted.png')
        # Close plot
        plt.close(self.plot.fig)
        self.dismiss_popup()
        Logger.info('Save complete!')

    # Popups -----------------------------------------------------------------
    def dismiss_popup(self):
        if self._popup:
            self._popup.dismiss()

    def show_dialog(self, dialog, title='', filters=[]):
        dialog.ids.filechooser.filters = filters
        self._popup = Popup(title=title, content=dialog)
        self._popup.open()

    def show_load_checkpoint(self):
        dialog = LoadDialog(load=self.load_sam_checkpoint, cancel=self.dismiss_popup)
        self.show_dialog(dialog, title='Load checkpoint', filters=['*.pth'])

    def show_load_data(self):
        # Triggers both "load grains" and "load summary" in sequence
        dialog = LoadDialog(load=self.load_data, cancel=self.dismiss_popup)
        self.show_dialog(dialog, title='Load grain data', filters=['*.csv'])

    def show_load_image(self):
        plt.close()
        dialog = LoadDialog(load=self.load_image, cancel=self.dismiss_popup)
        self.show_dialog(dialog, title='Load image', filters=['*.jpg', '*.jpeg', '*.png'])

    def show_load_unet(self):
        dialog = LoadDialog(load=self.load_unet, cancel=self.dismiss_popup)
        self.show_dialog(dialog, title='Load Unet model', filters=['*.keras'])

    def show_save(self):
        dialog = SaveDialog(save=self.save, cancel=self.dismiss_popup, path='.', filename=self.image_fn)
        self.show_dialog(dialog, title='Save results')


class SegmentApp(App):
    ''' Layout defined in segment.kv '''
    title = 'GrainTrainer2025'


# Logger.setLevel(LOG_LEVELS['error'])
SegmentApp().run()
