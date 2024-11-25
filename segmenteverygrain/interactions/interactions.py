import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio.features
import segmenteverygrain
import shapely
import skimage


class Grain(object):
    """ Stores data and plot representation for a single grain """
    
    def __init__(self, xy:list=[], data:pd.Series=None):
        # Data
        self.data = data
        self.xy = xy
        # Region info to calculate
        self.region_props = [
            'label',
            'area',
            'centroid',
            'major_axis_length',
            'minor_axis_length', 
            'orientation',
            'perimeter',
            'max_intensity',
            'mean_intensity',
            'min_intensity'
        ]

        # Display
        self.normal_props = {
            'alpha': 0.6
        }
        self.selected_props = {
            'alpha': 1.0,
            'facecolor': 'lime'
        }
        self.patch = None
        self.selected = False

    def get_polygon(self) -> shapely.Polygon:
        return shapely.Polygon(self.patch.get_path().vertices)

    def make_data(self, ax) -> pd.Series:
        # Get image and labeled region
        image = ax.get_images()[0].get_array()
        label = rasterio.features.rasterize([self.get_polygon()], out_shape=image.shape[:2])
        # Calculate region properties
        # TODO: Avoid going list -> DataFrame -> Series
        data = skimage.measure.regionprops_table(
            label, intensity_image=image, properties=self.region_props)
        self.data = pd.DataFrame(data).iloc[0]
        # print('New grain:', self.data)
        return self.data

    def make_patch(self, ax):
        # Create patch
        (self.patch,) = ax.fill(
            *self.xy,
            edgecolor='black',
            linewidth=2.0,
            picker=True,
            **self.normal_props
        )
        # Save original color (for select/unselect)
        self.normal_props['facecolor'] = self.patch.get_facecolor()
        # Compute grain data if not provided
        if self.data is None:
            self.data = self.make_data(ax)
        return self.patch

    def select(self):
        self.selected = ~self.selected
        props = self.selected_props if self.selected else self.normal_props
        self.patch.set(**props)
        # if self.selected:
        #     print(self.data)
        return self.selected


class GrainPlot(object):
    """ Interactive plot to create, delete, and merge grains """

    def __init__(self, grains, image=None, predictor=None, figsize=(6, 4)):
        # Input
        self.grains = grains
        self.image = image
        self.predictor = predictor
        # Interactions
        self.cids = []
        self.created_grains = []
        self.events = {
            'button_press_event': self.onclick,
            'pick_event': self.onpick,
            'key_press_event': self.onpress
        }
        self.last_pick = None
        self.left_cursor = patches.Circle((10, 10), radius=5, color='lime', visible=False)
        self.right_cursor = patches.Circle((10, 10), radius=5, color='red', visible=False)
        self.selected_grains = []
        # Plot
        self.fig = plt.figure(figsize=figsize)
        self.canvas = self.fig.canvas
        self.ax = self.fig.add_subplot(aspect='equal', xticks=[], yticks=[])
        if isinstance(image, np.ndarray):
            self.ax.imshow(image)
            self.ax.autoscale(enable=False)
        self.fig.tight_layout()
        # Draw elements on plot without updating after each one
        with plt.ioff():
            for grain in grains:
                grain.make_patch(self.ax)
            self.ax.add_patch(self.left_cursor)
            self.ax.add_patch(self.right_cursor)
        # Seems to help with occasional failure to draw updates
        plt.pause(0.1)

    # Helper functions ---
    def set_cursor(self, cursor, xy=False):
        """ Set left or right cursor to given location """
        if isinstance(xy, tuple):
            cursor.set_center(xy)
            cursor.set_visible(True)
        else:
            cursor.set_visible(False)

    def unset_cursors(self):
        """ Hide left an right cursors """
        self.set_cursor(self.left_cursor, False)
        self.set_cursor(self.right_cursor, False)

    def unselect_grains(self):
        """ Unselect all selected grains """
        for grain in self.selected_grains:
            grain.select()
        self.selected_grains = []

    def unselect_all(self):
        """ Hide both cursors and unselect all grains """
        self.unset_cursors()
        self.unselect_grains()

    # Manage grains ---
    def create_grain(self):
        """ Attempt to find and add grain at left cursor """
        # Verify that we've actually selected something
        if not self.left_cursor.get_visible():
            return
        xy1 = self.left_cursor.get_center()
        if self.right_cursor.get_visible():
            # Two-point prompt (grain and background)
            xy2 = self.right_cursor.get_center()
            x, y = segmenteverygrain.two_point_prompt(*xy1, *xy2, image=self.image, predictor=self.predictor)
        else:
            # One-point prompt (grain only)
            x, y, mask = segmenteverygrain.one_point_prompt(*xy1, image=self.image, predictor=self.predictor)
        # Record new grain (plot, data, and undo list)
        grain = Grain((x, y))
        grain.make_patch(self.ax)
        self.grains.append(grain)
        self.created_grains.append(grain)
        # Reset cursors
        self.unset_cursors()

    def delete_grains(self):
        """ Delete all selected grains """
        with plt.ioff():
            for grain in self.selected_grains:
                # Remove grain frmo plot, data, and undo list
                grain.patch.remove()
                self.grains.remove(grain)
                if grain in self.created_grains:
                    self.created_grains.remove(grain)
        self.selected_grains = []
        # Reset cursors
        self.unset_cursors()

    def merge_grains(self):
        """ Merge all selected grains """
        # Verify there are at least two grains to merge
        if len(self.selected_grains) < 2:
            return
        # Find vertices of merged grains using Shapely
        poly = shapely.unary_union([g.get_polygon() for g in self.selected_grains])
        # Verify grains actually overlap, otherwise reject selections
        if isinstance(poly, shapely.MultiPolygon):
            self.unselect_grains()
            return
        # Make new merged grain
        new_grain = Grain(poly.exterior.xy)
        new_grain.make_patch(self.ax)
        self.grains.append(new_grain)
        self.created_grains.append(new_grain)
        # Clear old constituent grains
        self.delete_grains()

    def undo_grain(self):
        """ Remove latest created grain """
        # Verify that there is a grain to undo
        if len(self.created_grains) < 1:
            return
        # Select and remove latest grain
        self.unselect_all()
        self.selected_grains = [self.created_grains[-1]]
        self.delete_grains()

    # Events ---
    def onclick(self, event):
        """ Handle clicking anywhere on plot """
        # Only individual clicks, only if not handled by onpick, only when no grains selected
        if event.dblclick is True or event is self.last_pick or len(self.selected_grains) > 0:
            return
        # Left click: set grain prompt
        if event.button == 1:
            self.set_cursor(self.left_cursor, (event.xdata, event.ydata))
        # Right click: set background prompt
        elif event.button == 3:
            self.set_cursor(self.right_cursor, (event.xdata, event.ydata))

    def onpick(self, event):
        """ Handle clicking on an existing grain """
        # Only individual left-clicks
        print("picked!")
        mouseevent = event.mouseevent
        if mouseevent.dblclick or mouseevent.button != 1:
            return
        # Tell onclick to ignore this event
        self.last_pick = mouseevent
        # Hide cursors
        self.unset_cursors()
        # Add selected grain to selection list
        for grain in self.grains:
            if event.artist is grain.patch:
                if grain.select():
                    self.selected_grains.append(grain)
                else:
                    self.selected_grains.remove(grain)
                break
    
    def onpress(self, event):
        """ Handle key presses """
        if event.key == 'c':
            self.create_grain()
        elif event.key == 'd' or event.key == 'delete':
            self.delete_grains()
        elif event.key == 'm':
            self.merge_grains()
        elif event.key == 'z':
            self.undo_grain()
        elif event.key == 'escape':
            self.unselect_all()
        # TODO: Handle occasional drawing failures by forcing a GUI update
        # elif event.key == 'f':
        #     bg = self.canvas.copy_from_bbox(self.fig.bbox)
        #     self.canvas.restore_region(bg)
        #     self.canvas.blit(self.fig.bbox)
        #     # self.canvas.draw()
        #     self.canvas.flush_events()

    def activate(self):
        """ Enable interactive features """
        for event, handler in self.events.items():
            self.cids.append(self.canvas.mpl_connect(event, handler))

    def deactivate(self):
        """ Disable interactive features """
        for cid in self.cids:
            self.canvas.mpl_disconnect(cid)
        self.cids = []
    
    # Output ---
    def get_mask(self):
        """ Return labeled image for Unet training """
        all_grains = [g.get_polygon() for g in self.grains]
        return segmenteverygrain.create_labeled_image(all_grains, self.image)

    def get_data(self) -> pd.DataFrame:
        """ Return up-to-date DataFrame of grain stats """
        return pd.concat([g.data for g in self.grains], axis=1).T
