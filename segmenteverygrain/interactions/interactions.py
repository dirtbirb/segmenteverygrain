import matplotlib as mpl
import matplotlib.style as mplstyle
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np
import pandas as pd
import rasterio.features
import segmenteverygrain
import shapely
import skimage


# HACK: Attach parent grain reference to the Polygon class
mpatches.Polygon.grain = None

# Speed up rendering a little?
mplstyle.use('fast')

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

    def make_data(self, ax:mpl.axes.Axes) -> pd.Series:
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
        data = pd.DataFrame(data)
        if len(data):
            self.data = data.iloc[0]
        else:
            print('MAKE_DATA_ERROR ', pd.DataFrame(data))
            self.data = pd.Series()
        # print('New grain:', self.data)
        return self.data

    def make_patch(self, ax:mpl.axes.Axes) -> mpatches.Polygon:
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
            animated=True,
            **self.normal_props
        )
        # HACK: Save reference to parent grain within the patch itself
        self.patch.grain = self
        # Save original color (for select/unselect)
        self.normal_props['facecolor'] = self.patch.get_facecolor()
        # Compute grain data if not provided
        if self.data is None:
            self.data = self.make_data(ax)
        return self.patch

    def select(self) -> bool:
        '''
        Toggle whether grain is selected/unselected in a plot.
        
        Returns
        -------
        self.selected : bool
            True if this grain is now selected.
        '''
        self.selected = not self.selected
        self.patch.update(
            self.selected_props if self.selected else self.normal_props)
        return self.selected


class GrainPlot(object):
    ''' Interactive plot to create, delete, and merge grains. '''

    def __init__(self,
            grains:list=[], 
            image=None, 
            predictor=None, 
            figsize=(6, 4), 
            blit=True,
            minspan=10):
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
        self.blit = blit
        self.minspan = minspan
        # Interactions
        self.cids = []
        self.created_grains = []
        self.events = {
            'button_press_event': self.onclick,
            'key_press_event': self.onkey,
            'key_release_event': self.onkeyup,
            'pick_event': self.onpick
        }
        self.last_pick = (0, 0)
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
        # Box selector
        self.box = np.zeros(4, dtype=int)
        self.box_selector = mwidgets.RectangleSelector(
            self.ax,
            onselect=lambda a, b: None, # Shouldn't be required, but it is
            minspanx=minspan,           # Don't accidentally trigger on click
            minspany=minspan,   
            useblit=True,               # Always try to use blitting
            props={
                'facecolor': 'lime',
                'edgecolor': 'black',
                'alpha': 0.2,
                'fill': True},
            spancoords='pixels',
            button=[1],                 # Left mouse button only
            interactive=True)
        self.box_selector.set_active(False)
        # Draw grains and initialize plot
        for grain in grains:
            grain.make_patch(self.ax)
        if blit:
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    # Fast display method ---
    def do_blit(self):
        ''' Blit background image and draw animated art'''
        # Reset background
        self.canvas.restore_region(self.background)
        # Draw highlighted grains
        for grain in self.selected_grains:
            self.ax.draw_artist(grain.patch)
        # Draw point prompts
        for point in self.points:
            self.ax.draw_artist(point)
        # Push to canvas
        self.canvas.blit(self.ax.bbox)
        self.canvas.flush_events()

    # Selection helpers ---
    def set_point(self, xy:tuple, is_inside:bool=True):
        ''' Set point prompt. '''
        color = 'lime' if is_inside else 'red'
        new_point = mpatches.Circle(
            xy, radius=5, color=color, animated=self.blit)
        self.ax.add_patch(new_point)
        self.points.append(new_point)
        self.point_labels.append(is_inside)

    def clear_points(self):
        ''' Clear all prompt points. '''
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
        self.box_selector.set_active(False)
        self.box_selector.set_visible(False)
        self.clear_points()
        self.unselect_grains()

    # Manage grains ---
    def create_grain(self):
        ''' Attempt to find and add grain at most recent clicked position. '''
        # Interpret point prompts
        if len(self.points):
            points = [p.get_center() for p in self.points]
            point_labels = self.point_labels
        else:
            points = None
            point_labels = None
        # Interpret box prompt
        if self.box_selector.active:
            xmin, xmax, ymin, ymax = self.box_selector.extents
            box = [xmin, ymin, xmax, ymax]
        else:
            # Return if we haven't selected anything
            if points is None:
                return
            box = None
        # Use prompts to find a grain
        coords = segmenteverygrain.predict_from_prompts(
            predictor=self.predictor,
            box=box,
            points=points,
            point_labels=point_labels
        )
        # Record new grain (plot, data, and undo list)
        grain = Grain(coords)
        grain.make_patch(self.ax)
        self.grains.append(grain)
        self.created_grains.append(grain)
        # Clear prompts
        self.unselect_all()
        if self.blit:
            # Update background
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def delete_grains(self):
        ''' Delete all selected grains. '''
        for grain in self.selected_grains:
            # Remove grain from plot, data, and undo list
            grain.patch.remove()
            self.grains.remove(grain)
            if grain in self.created_grains:
                self.created_grains.remove(grain)
        self.selected_grains = []
        if self.blit:
            # Update background
            self.canvas.draw()
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def merge_grains(self):
        ''' Merge all selected grains. '''
        # Verify there are at least two grains to merge
        if len(self.selected_grains) < 2:
            return
        # Find vertices of merged grains using Shapely
        poly = shapely.unary_union(
            [g.get_polygon() for g in self.selected_grains])
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
    def onclick(self, event:mpl.backend_bases.MouseEvent):
        ''' Handle clicking anywhere on plot. Triggers on click release.
        
        Parameters
        ----------
        event
            Matplotlib mouseevent (different than normal event!)
        '''
        # No double-clicks
        # No box selecting
        # No pick events (no point prompts on existing grains)
        # No point prompts when grains are selected
        if (event.dblclick or
                self.box_selector.active or
                self.last_pick == (round(event.xdata), round(event.ydata)) or
                len(self.selected_grains) > 0):
            return
        # Left click: grain prompt
        if event.button == 1:
            self.set_point((event.xdata, event.ydata), True)
        # Right click: background prompt
        elif event.button == 3:
            self.set_point((event.xdata, event.ydata), False)
        # Neither: don't update the canvas
        else:
            return
        # Update canvas
        if self.blit:
            self.do_blit()
        else:
            # Apparently necessary if plot shown twice
            self.canvas.draw_idle()

    def onkey(self, event:mpl.backend_bases.KeyEvent):
        ''' 
        Handle key presses.
        
        Parameters
        ----------
        event
            Matplotlib KeyEvent
        '''
        # Handle key
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
        elif event.key == 'shift':
            self.box_selector.set_active(True)
        else:
            return
        # Update canvas
        if self.blit:
            self.do_blit()
        else:
            # Apparently necessary if plot shown twice
            self.canvas.draw_idle()

    def onkeyup(self, event:mpl.backend_bases.KeyEvent):
        # Cancel box selector if too small
        if event.key == 'shift':
            xmin, xmax, ymin, ymax = self.box_selector.extents
            area = abs(xmax-xmin) * abs(ymax-ymin)
            if area < self.minspan ** 2:
                self.box_selector.set_active(False)

    def onpick(self, event:mpl.backend_bases.PickEvent):
        '''
        Handle clicking on an existing grain to select/unselect it.
        
        Parameters
        ----------
        event
            Matplotlib event
        '''
        mouseevent = event.mouseevent
        # No doubleclicks, no scroll clicks, no box selection
        if (mouseevent.dblclick or
                mouseevent.button == 2 or
                self.box_selector.active):
            return
        # Block any point prompts that would be on a grain
        self.last_pick = (round(mouseevent.xdata), round(mouseevent.ydata))
        # Only pick on left-click when no point prompts exist
        if mouseevent.button != 1 or len(self.points) > 0:
            return
        # Add/remove selected grain to/from selection list
        grain = event.artist.grain
        if grain.select():
            self.selected_grains.append(grain)
        else:
            self.selected_grains.remove(grain)
        # Update canvas
        if self.blit:
            self.do_blit()
        else:
            # Apparently necessary if plot shown twice
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
    def get_mask(self) -> list:
        ''' Return labeled image for Unet training. '''
        all_grains = [g.get_polygon() for g in self.grains]
        return segmenteverygrain.create_labeled_image(all_grains, self.image)

    def get_data(self) -> pd.DataFrame:
        ''' Return up-to-date DataFrame of grain stats. '''
        return pd.concat([g.data for g in self.grains], axis=1).T
