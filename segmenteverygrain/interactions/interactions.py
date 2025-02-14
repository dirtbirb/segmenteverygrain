import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio.features
import segmenteverygrain
import shapely
import skimage


# Don't reset zoom level when pressing 'c' to create a grain
if 'c' in mpl.rcParams['keymap.back']:
    mpl.rcParams['keymap.back'].remove('c')


class Grain(object):
    ''' Stores data and plot representation for a single grain. '''
        
    def __init__(self, xy:list=[], data:pd.Series=None):
        '''
        Parameters
        ----------
        xy : list of (x, y) tuples
            Coordinates to draw this grain as a polygon.
        data : pd.Series (optional)
            Row from a DataFrame containing information about this grain. 
        '''
        
        # Input
        self.data = data
        self.xy = xy
        # Grain properties to calculate
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
        ''' Return a shapely.Polygon representing the matplot patch. '''
        return shapely.Polygon(self.patch.get_path().vertices)

    def make_data(self, ax) -> pd.Series:
        '''
        Calculate grain information from image and matplot patch.
        Overwrites self.data.

        Parameters
        ----------
        ax : matplotlib.Axes
            Axes instance to get the background image from.
            Used to calculate intensity in self.data.

        Returns
        -------
        self.data : pd.Series
            Row for a DataFrame containing computed grain info.
        '''
        # Get image and labeled region
        image = ax.get_images()[0].get_array()
        label = rasterio.features.rasterize(
            [self.get_polygon()], out_shape=image.shape[:2])
        # Calculate region properties
        # TODO: Avoid going list -> DataFrame -> Series
        data = skimage.measure.regionprops_table(
            label, intensity_image=image, properties=self.region_props)
        self.data = pd.DataFrame(data).iloc[0]
        # print('New grain:', self.data)
        return self.data

    def make_patch(self, ax):
        '''
        Draw this grain on the provided matplotlib axes and save the result.

        Parameters
        ----------
        ax : matplotlib.Axes
            Axes instance on which to draw this grain.

        Returns
        -------
        self.patch
            Object representing this grain on the plot.
        '''
                
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

    def select(self) -> bool:
        '''
        Record whether grain is selected/unselected in a plot.
        
        Returns
        -------
        self.selected : bool
            True if this grain is now selected.
        '''
        self.selected = not self.selected
        props = self.selected_props if self.selected else self.normal_props
        self.patch.set(**props)
        # if self.selected:
        #     print(self.data)
        return self.selected


class GrainPlot(object):
    ''' Interactive plot to create, delete, and merge grains. '''

    def __init__(self, grains:list=[], image=None, predictor=None, figsize=(6, 4)):
        '''
        Parameters
        ----------
        grains : list
            List of grains with xy data to plot over the backround image.
        image : np.ndarray
            Image under analysis, displayed behind identified grains.
        predictor:
            SAM predictor used to create new grains.
        figsize: tuple
            Figure size.
        '''

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
        self.points = []
        self.point_labels = []
        self.selected_grains = []
        # Plot
        self.fig = plt.figure(figsize=figsize)
        self.canvas = self.fig.canvas
        self.ax = self.fig.add_subplot(aspect='equal', xticks=[], yticks=[])
        if isinstance(image, np.ndarray):
            self.ax.imshow(image)
            self.ax.autoscale(enable=False)
        self.fig.tight_layout(pad=0)
        # Draw elements on plot without updating after each one
        with plt.ioff():
            for grain in grains:
                grain.make_patch(self.ax)
        # Seems to help with occasional failure to draw updates
        plt.pause(0.1)

    # Helper functions ---
    def set_point(self, xy:tuple, is_inside:bool=True):
        ''' Set prompt point '''
        color = 'lime' if is_inside else 'red'
        new_point = patches.Circle(xy, radius=5, color=color)
        self.ax.add_patch(new_point)
        self.points.append(new_point)
        self.point_labels.append(is_inside)

    def clear_points(self):
        ''' Clear all prompt points '''
        for point in self.points:
            point.remove()
        self.points = []
        self.point_labels = []

    def unselect_grains(self):
        ''' Unselect all selected grains. '''
        for grain in self.selected_grains:
            grain.select()
        self.selected_grains = []

    def unselect_all(self):
        ''' Clear point prompts and unselect all grains. '''
        self.clear_points()
        self.unselect_grains()

    # Manage grains ---
    def create_grain(self):
        ''' Attempt to find and add grain at most recent clicked position. '''
        # Verify that we've actually selected something
        if not self.points:
            return
        # Attempt to find new grain using given point(s)
        points = [p.get_center() for p in self.points]
        coords = segmenteverygrain.predict_from_prompts(
            predictor=self.predictor,
            points=points,
            point_labels=self.point_labels
        )
        # Record new grain (plot, data, and undo list)
        grain = Grain(coords)
        grain.make_patch(self.ax)
        self.grains.append(grain)
        self.created_grains.append(grain)
        # Clear prompts
        self.clear_points()

    def delete_grains(self):
        ''' Delete all selected grains. '''
        for grain in self.selected_grains:
            # Remove grain from plot, data, and undo list
            grain.patch.remove()
            self.grains.remove(grain)
            if grain in self.created_grains:
                self.created_grains.remove(grain)
        self.selected_grains = []

    def merge_grains(self):
        ''' Merge all selected grains. '''
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
        ''' Remove latest created grain. '''
        # TODO: Also allow undoing grain deletions
        # Verify that there is a grain to undo
        if len(self.created_grains) < 1:
            return
        # Select and remove latest grain
        self.unselect_all()
        self.selected_grains = [self.created_grains[-1]]
        self.delete_grains()

    # Events ---
    def onclick(self, event):
        ''' Handle clicking anywhere on plot.
        
        Parameters
        ----------
        event
            Matplotlib mouseevent (different than normal event!)
        '''
        # Only individual clicks
        # Only if not handled by onpick (didn't select a grain)
        # Only when no grains selected
        if (event.dblclick is True
            or event is self.last_pick
            or len(self.selected_grains) > 0):
            return
        # Left click: set grain prompt
        if event.button == 1:
            self.set_point((event.xdata, event.ydata))
        # Right click: set background prompt
        elif event.button == 3:
            self.set_point((event.xdata, event.ydata), False)
        # Neither: don't update the graph
        else:
            return
        # Draw results to canvas (necessary if plot is shown twice, for some reason)
        self.canvas.draw_idle()

    def onpick(self, event):
        '''
        Handle clicking on an existing grain to select/unselect it.
        
        Parameters
        ----------
        event
            Matplotlib event
        '''
        # Only individual left-clicks
        mouseevent = event.mouseevent
        if mouseevent.dblclick or mouseevent.button != 1:
            return
        # Tell onclick to ignore this event
        self.last_pick = mouseevent
        # Remove point prompts
        self.clear_points()
        # Add/remove selected grain to/from selection list
        for grain in self.grains:
            if event.artist is grain.patch:
                if grain.select():
                    self.selected_grains.append(grain)
                else:
                    self.selected_grains.remove(grain)
                break
        # Draw results to canvas (necessary if plot is shown twice)
        self.canvas.draw_idle()
    
    def onpress(self, event):
        ''' 
        Handle key presses.
        
        Parameters
        ----------
        event
            Matplotlib event
        '''
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
        else:
            return
        # Draw results to canvas (necessary if plot is shown twice)
        self.canvas.draw_idle()

    def activate(self):
        ''' Enable interactive features (clicking, etc). '''
        for event, handler in self.events.items():
            self.cids.append(self.canvas.mpl_connect(event, handler))

    def deactivate(self):
        ''' Disable interactive features (clicking, etc). '''
        for cid in self.cids:
            self.canvas.mpl_disconnect(cid)
        self.cids = []
    
    # Output ---
    def get_mask(self):
        ''' Return labeled image for Unet training. '''
        all_grains = [g.get_polygon() for g in self.grains]
        return segmenteverygrain.create_labeled_image(all_grains, self.image)

    def get_data(self) -> pd.DataFrame:
        ''' Return up-to-date DataFrame of grain stats. '''
        return pd.concat([g.data for g in self.grains], axis=1).T
