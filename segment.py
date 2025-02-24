import keras.saving
import matplotlib.pyplot as plt
import numpy as np
import os.path
import segment_anything
import segmenteverygrain
import segmenteverygrain.interactions as si

import kivy
kivy.require('2.3.1')
from kivy.app import App
from kivy.logger import Logger, LOG_LEVELS
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy_garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg


# HACK: Bypass large image restriction
from PIL import Image
Image.MAX_IMAGE_PIXELS = None 

# HACK: Turn off crazy debug output
import logging
for logger in [logging.getLogger(name) for name in logging.root.manager.loggerDict]:
    if logger.level == 0:
        logger.setLevel(30)


FIGURE_DPI = 72
# FIGSIZE = (img_x/FIGURE_DPI, img_y/FIGURE_DPI)
FIGSIZE = (12, 8)
MIN_AREA = 400
OVERLAP = 200
PATCH_SIZE = 2000


class LoadDialog(BoxLayout):
    cancel = ObjectProperty()
    load = ObjectProperty()


class PointCountDialog(BoxLayout):
    cancel = ObjectProperty()
    count = ObjectProperty()


class SaveDialog(BoxLayout):
    cancel = ObjectProperty()
    filename = StringProperty()
    path = StringProperty()
    save = ObjectProperty()
    text_input = ObjectProperty()


class RootLayout(BoxLayout):
    # Internal properties
    _popup = ObjectProperty()
    predictor = ObjectProperty()
    predictor_stale = BooleanProperty(True)
    # User settables
    grains = ListProperty()
    grains_fig = ObjectProperty()
    grains_fn = StringProperty()
    image = ObjectProperty()
    image_fn = StringProperty()
    sam = ObjectProperty()
    sam_checkpoint_fn = StringProperty()
    unet_image = ObjectProperty()
    unet_model = ObjectProperty()
    unet_fn = StringProperty()

    # Load defaults ----------------------------------------------------------
    def on_kv_post(self, widget):
        self.load_sam_checkpoint('', './sam_vit_h_4b8939.pth')
        self.load_unet('', './segmenteverygrain/seg_model.keras')
        self.update_data_labels()

    def update_data_labels(self, text='Load'):
        self.grains_fn = text

    # Segmentation -----------------------------------------------------------
    def auto_segment(self):
        Logger.info('Auto-segmenting: Normal image ---')
        plt.close('all')

        # Generate prompts with UNET model
        Logger.info('UNET prediction')
        self.unet_image = segmenteverygrain.predict_image(
            self.image, self.unet_model, I=256)
        unet_labels, self.unet_coords = segmenteverygrain.label_grains(
            self.image, self.unet_image, dbs_max_dist=20.0)

        # Apply SAM for actual segmentation
        Logger.info('SAM segmenting')
        # TODO: Separate this function into smaller chunks (plotting, mask, etc)
        # TODO: Choose min_area by image size? Do unit conversion from pixels first?
        grains, sam_labels, mask_all, summary, self.grains_fig, ax = segmenteverygrain.sam_segmentation(
            self.sam, self.image, self.unet_image, self.unet_coords, unet_labels,
            min_area=MIN_AREA, plot_image=True, remove_edge_grains=False, remove_large_objects=False)

        # Process results
        self.grains = [si.Grain(np.array(g.exterior.xy)) for g in grains]
        for g in self.grains:
            g.measure(image=self.image)

        # Update GUI and show save dialog
        Logger.info('Auto-segmenting complete!')
        self.update_data_labels('Calculated!')
        self.show_save()
    
    def large_segment(self):
        Logger.info('Auto-segmenting: Large image ---')
        plt.close('all')

        # Use SEG large image prediction
        grains, self.unet_image, self.unet_coords = segmenteverygrain.predict_large_image(
            self.image_fn, self.unet_model, self.sam,
            min_area=MIN_AREA, patch_size=PATCH_SIZE, overlap=OVERLAP)

        # Process results -- pretty image
        self.grains_fig, ax = plt.subplots(
            figsize=FIGSIZE)
        ax.set_aspect('equal')
        segmenteverygrain.plot_image_w_colorful_grains(
            self.image, grains, ax, cmap='Paired')
        ax.set(xticks=[], yticks=[])
        plt.xlim([0, np.shape(self.image)[1]])
        plt.ylim([np.shape(self.image)[0], 0])

        # Process results -- everything else
        self.grains = [si.Grain(np.array(g.exterior.xy)) for g in grains]
        for g in self.grains:
            g.measure(image=self.image)

        # Update GUI and show save dialog
        Logger.info('Auto-segmenting complete!')
        self.update_data_labels('Calculated!')
        self.show_save()

    def manual_segment(self):
        Logger.info('Manual segmenting ---')
        plt.close('all')

        # Prepare SAM predictor
        if self.predictor_stale:
            Logger.info('Preparing SAM predictor')
            self.predictor.set_image(self.image)
            self.predictor_stale = False
        
        # Display editing interface
        Logger.info('Displaying interactive interface')
        plot = si.GrainPlot(
            self.grains, 
            image=self.image, 
            predictor=self.predictor,
            figsize=FIGSIZE
        )
        plot.activate()
        plt.show(block=True)
        plot.deactivate()

        # Process results
        self.grains_fig = plot.fig

        # Update GUI and show save dialog
        Logger.info('Manual editing complete!')
        self.update_data_labels('Edited!')
        self.show_save()

    def point_count(self, spacing: int):
        self.dismiss_popup()
        Logger.info('Point count ---')
        Logger.info(f'Spacing: {spacing} pixels')

        # Find and measure grains at grid locations
        Logger.info('Performing count')
        points, xs, ys = si.make_grid(self.image, spacing)
        grains, points_found = si.filter_grains_by_points(self.grains, points)
        for g in grains:
            if g.data is None:
                g.measure(image=self.image)
        
        # Make GrainPlot and save it as a static image
        Logger.info('Plotting results')
        img_y, img_x = self.image.shape[:2]
        plot = si.GrainPlot(grains, self.image, 
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
            grain.draw_axes(ax)

        # Plot grid
        point_colors = ['lime' if p else 'red' for p in points_found]
        ax.scatter(xs, ys,
            s=min(plot_image[:2].shape) * FIGURE_DPI,
            c=point_colors,
            edgecolors='black')
        self.grains_fig = fig

        # Update GUI and show save dialog
        Logger.info('Point count complete!')
        self.grains = grains
        self.show_save()


    # Save/load --------------------------------------------------------------
    def load_grains(self, path, filename):
        # Load grain data csv
        Logger.info('Loading grains...')
        self.grains = si.load_grains(filename)
        self.grains_fn = os.path.basename(filename)
        self.dismiss_popup()
        Logger.info(f'Loaded {self.grains_fn}.')

    def load_image(self, path, filename):
        Logger.info('Loading image...')
        self.image = si.load_image(filename)
        self.ids.image.source = filename
        self.image_fn = os.path.basename(filename)
        self.dismiss_popup()
        # Update image predictor next time manual editing is used
        self.predictor_stale = True
        # Clear grain information, probably applied to a different image
        self.grains = []
        self.update_data_labels()
        Logger.info(f'Loaded {self.image_fn}.')

    def load_sam_checkpoint(self, path, filename):
        Logger.info('Loading checkpoint...')
        self.sam = segment_anything.sam_model_registry["default"](checkpoint=filename)
        self.sam_checkpoint_fn = os.path.basename(filename)
        self.predictor = segment_anything.SamPredictor(self.sam)
        self.dismiss_popup()
        Logger.info(f'Loaded checkpoint {self.sam_checkpoint_fn}.')

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
        # TODO
        Logger.info('Saving grain image...')
        self.grains_fig.savefig(filename, bbox_inches='tight', pad_inches=0)
        self.ids.image.source = filename
        self.dismiss_popup()
        Logger.info(f'Saved {filename}.')

    def save_grains(self, filename):
        Logger.info('Saving grain data...')
        si.save_grains(filename, self.grains)
        Logger.info(f'Saved {filename}.')

    def save_mask(self, filename, readable=True):
        Logger.info('Saving mask...')
        # Computer-readable mask (pixel values are 0 or 1)
        si.save_mask(filename, self.grains, self.image, scale=False)
        Logger.info(f'Saved {filename}.')
        # Human-readable mask (pixel values are 0 or 127)
        if readable:
            filename = filename.split('.')[0] + '.jpg'
            si.save_mask(filename, self.grains, self.image, scale=True)
            Logger.info(f'Saved {filename}.')

    def save_summary(self, filename, histogram=True):
        Logger.info('Saving summary data...')
        # Get measurements from plot as a DataFrame
        si.save_summary(filename, self.grains)
        Logger.info(f'Saved {filename}.')
        # Build and save histogram
        if histogram:
            filename = filename.split('.')[0] + '.jpg'
            si.save_histogram(filename, self.grains)
            Logger.info(f'Saved {filename}.')

    def save_unet_image(self, filename):
        # Save unet results for verification (auto segmenting!)
        Logger.info(f'Saving unet image...')
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_aspect('equal')
        ax.imshow(self.unet_image)
        plt.scatter(
            np.array(self.unet_coords)[:,0],
            np.array(self.unet_coords)[:,1],
            c='k')
        ax.set(xticks=[], yticks=[])
        fig.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        Logger.info(f'Saved {filename}.')

    def save(self, path, filename, grains=None):
        ''' 
        Save all results via save_whatever method for each type of data.
        '''
        Logger.info('--- Results ---')
        # Parse input: include path and name, remove extension
        filename = os.path.join(path, filename.split('.')[0])
        # Save results
        self.save_grains(filename + '_grains.csv')
        self.save_summary(filename + '_summary.csv')
        if self.unet_image is not None:
            self.save_unet_image(filename + '_unet.jpg')
        self.save_mask(filename + '_mask.png')
        self.save_grain_image(filename + '_grains.jpg')
        # Close plot
        self.dismiss_popup()
        plt.close('all')
        Logger.info('Save complete!')

    # Popups -----------------------------------------------------------------
    def dismiss_popup(self):
        if self._popup:
            self._popup.dismiss()

    def show_dialog(self, dialog, title='', filters=[]):
        if 'filechooser' in dialog.ids:
            dialog.ids.filechooser.filters = filters
        self._popup = Popup(title=title, content=dialog)
        self._popup.open()

    def show_load_checkpoint(self):
        dialog = LoadDialog(
            load=self.load_sam_checkpoint, cancel=self.dismiss_popup)
        self.show_dialog(dialog, title='Load checkpoint', filters=['*.pth'])

    def show_load_grains(self):
        dialog = LoadDialog(load=self.load_grains, cancel=self.dismiss_popup)
        self.show_dialog(dialog, title='Load grain data', filters=['*.csv'])

    def show_load_image(self):
        plt.close()
        dialog = LoadDialog(load=self.load_image, cancel=self.dismiss_popup)
        self.show_dialog(
            dialog, title='Load image', filters=['*.jpg', '*.jpeg', '*.png'])

    def show_load_unet(self):
        dialog = LoadDialog(load=self.load_unet, cancel=self.dismiss_popup)
        self.show_dialog(dialog, title='Load Unet model', filters=['*.keras'])

    def show_point_count(self):
        dialog = PointCountDialog(
            count=self.point_count, cancel=self.dismiss_popup)
        self.show_dialog(dialog, title='Grid spacing')

    def show_save(self):
        dialog = SaveDialog(
            save=self.save, cancel=self.dismiss_popup, 
            path='.', filename=self.image_fn)
        self.show_dialog(dialog, title='Save results')


class SegmentApp(App):
    ''' Layout defined in segment.kv '''
    title = 'GrainTrainer2025'


# Logger.setLevel(LOG_LEVELS['error'])
SegmentApp().run()
