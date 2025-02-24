import keras.utils
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


# Speed up rendering a little?
mplstyle.use('fast')

# Attach parent grain reference to the Polygon class
mpatches.Polygon.grain = None

# Don't reset zoom level when pressing 'c' to create a grain
if 'c' in mpl.rcParams['keymap.back']:
    mpl.rcParams['keymap.back'].remove('c')


class Grain(object):
    ''' Stores data and plot representation for a single grain. '''
        
    def __init__(self, xy:np.ndarray, data:pd.Series=None):
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
        self.xy = np.array(xy)
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
            # facecolor is set when patch is created
        }
        self.selected_props = {
            'alpha': 1.0,
            'facecolor': 'lime'
        }
        self.patch = None
        self.selected = False

    def get_polygon(self) -> shapely.Polygon:
        ''' Return a shapely.Polygon representing the matplotlib patch. '''
        poly = shapely.Polygon(self.xy.T)
        return poly

    def measure(self, ax:mpl.axes.Axes=None, image=None) -> pd.Series:
        '''
        Calculate grain information from image and matplotlib patch.
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
        if ax:
            image = ax.get_images()[0].get_array()
        label = rasterio.features.rasterize(
            [self.get_polygon()], out_shape=image.shape[:2])
        # Calculate region properties
        # TODO: Avoid going list -> DataFrame -> Series
        data = skimage.measure.regionprops_table(
            label, intensity_image=image, properties=self.region_props)
        data = pd.DataFrame(data)
        if len(data):
            self.data = data.iloc[0].drop('label')
        else:
            print('MEASURE_ERROR ', pd.DataFrame(data))
            self.data = pd.Series()
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
        # Save assigned color (for select/unselect)
        self.normal_props['facecolor'] = self.patch.get_facecolor()
        # Compute grain data if not provided
        if self.data is None:
            self.data = self.measure(ax)
        return self.patch

    def draw_axes(self, ax:mpl.axes.Axes):
        if self.data is None:
            self.measure(ax.get_images()[0].get_array())
        data = self.data

        x0, y0 = data['centroid-1'], data['centroid-0']
        ax.plot(x0, y0, '.k')
        orientation = data['orientation']
        x1 = x0 + np.cos(orientation) * 0.5 * data['minor_axis_length']
        y1 = y0 - np.sin(orientation) * 0.5 * data['minor_axis_length']
        ax.plot((x0, x1), (y0, y1), '-k')
        x2 = x0 - np.sin(orientation) * 0.5 * data['major_axis_length']
        y2 = y0 - np.cos(orientation) * 0.5 * data['major_axis_length']
        ax.plot((x0, x2), (y0, y2), '-k')

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
            blit=True,
            minspan=10,
            image_alpha=1,
            **kwargs):
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
            'draw_event': self.ondraw,
            'key_press_event': self.onkey,
            'key_release_event': self.onkeyup,
            'pick_event': self.onpick
        }
        self.last_pick = (0, 0)
        self.points = []
        self.point_labels = []
        self.selected_grains = []
        
        # Plot
        self.fig = plt.figure(**kwargs)
        self.canvas = self.fig.canvas
        self.ax = self.fig.add_subplot(aspect='equal', xticks=[], yticks=[])
        if isinstance(image, np.ndarray):
            self.ax.imshow(image, alpha=image_alpha)
            self.ax.autoscale(enable=False)
        self.fig.tight_layout(pad=0)
        
        # Interactive toolbar: inject unselect_all before any zoom/pan changes
        toolbar = self.canvas.toolbar
        toolbar._update_view = self.unselect_before(toolbar._update_view)
        toolbar.release_pan = self.unselect_before(toolbar.release_pan)
        toolbar.release_zoom = self.unselect_before(toolbar.release_zoom)
        
        # Box selector
        self.box = np.zeros(4, dtype=int)
        self.box_selector = mwidgets.RectangleSelector(
            self.ax, 
            lambda *args: None,         # Shouldn't be necessary, but it is
            minspanx=minspan,           # Minimum selection size
            minspany=minspan,   
            useblit=True,               # Always try to use blitting
            props={
                'facecolor': 'lime',
                'edgecolor': 'black',
                'alpha': 0.2,
                'fill': True},
            spancoords='pixels',
            button=[1],                 # Left mouse button only
            interactive=True,
            state_modifier_keys={}
            )
        self.box_selector.set_active(False)
        # Replace RectangleSelector update methods to avoid redundant blitting
        if blit:
            self.box_selector.update = self.update
            self.box_selector.update_background = lambda *args: None
        
        # Info box
        self.info = self.ax.annotate('',
            xy=(0, 0),
            xytext=(0, 0),
            textcoords='offset points',
            ha='center',
            va='center',
            bbox={'boxstyle': 'round', 'fc':'w'}, 
            animated=blit)
        self.show_info = True

        # Draw grains and initialize plot
        for grain in grains:
            grain.make_patch(self.ax)
        if blit:
            self.canvas.draw()

    # Display helpers ---
    def unselect_before(self, f):
        ''' Wrap a function to call unselect_all before it. '''
        def newf(*args, **kwargs):
            if self.blit:
                self.unselect_all()
            return f(*args, **kwargs)
        return newf

    def update(self):
        ''' Blit background image and draw animated art'''
        # Reset background
        self.canvas.restore_region(self.background)
        # Draw animated artists
        artists = (tuple(g.patch for g in self.selected_grains)
            + tuple(self.points)
            + self.box_selector.artists
            + (self.info,))
        for a in artists:
            self.ax.draw_artist(a)
        # Push to canvas
        self.canvas.blit(self.ax.bbox)

    def toggle_info(self, show=None):
        # Toggle info box flag
        self.show_info = show or not self.show_info
        # Show info box if requested and grains selected
        self.info.set_visible(self.show_info and len(self.selected_grains))

    def update_info(self):
        # Update based on last selected grain, or hide if none selected
        if not len(self.selected_grains):
            self.info.set_visible(False)
            return
        grain = self.selected_grains[-1]
        # Update offset based on position
        ext = grain.patch.get_extents()
        img_x, img_y = self.canvas.get_width_height()
        x = -0.1 if (ext.x1 + ext.x0) / img_x > 1 else 1.1
        y = -0.1 if (ext.y1 + ext.y0) / img_y > 1 else 1.1
        # Scoot annotation a little more for small grains
        if abs(ext.y1 - ext.y0) < img_y / 20:
            if y <= 0:
                y -= 1.4 
            else:
                y += 1.4
        self.info.xy = (x, y)
        # Update position
        self.info.xycoords = grain.patch
        # Update text
        text = (f"Major: {grain.data['major_axis_length']:.0f}px\n"
                f"Minor: {grain.data['minor_axis_length']:.0f}px\n"
                f"Area: {grain.data['area']:.0f}px")
        self.info.set_text(text)
        # Show info box if requested
        if self.show_info:
            self.info.set_visible(True)

    # Selection helpers ---
    def activate_box(self, activate=True):
        box = self.box_selector
        alpha = 0.4 if activate else 0.2
        box.set_props(alpha=alpha)
        box.set_handle_props(alpha=alpha, visible=activate)
        box.set_active(activate)
    
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
        ''' Clear box and point prompts and unselect all grains. '''
        self.box_selector.clear()
        self.clear_points()
        self.unselect_grains()
        self.toggle_info(False)

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
        if self.box_selector._selection_completed:
            xmin, xmax, ymin, ymax = self.box_selector.extents
            box = [xmin, ymin, xmax, ymax]
        else:
            # Return if we haven't provided any prompts
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
        # Update background
        if self.blit:
            self.canvas.draw()

    def delete_grains(self):
        ''' Delete all selected grains. '''
        # Verify that at least one grain is selected
        if len(self.selected_grains) < 1:
            return
        # Remove selected grains from plot, data, and undo list
        for grain in self.selected_grains:
            grain.patch.remove()
            self.grains.remove(grain)
            if grain in self.created_grains:
                self.created_grains.remove(grain)
        self.unselect_all()
        # Update background
        if self.blit:
            self.canvas.draw()

    def merge_grains(self):
        ''' Merge all selected grains. '''
        # Verify there are at least two grains selected to merge
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
        # Delete old constituent grains
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
        # Must be in axis
        # No double-clicks
        # Not during box selection (shift is pressed)
        # Not during toolbar interactions (pan/zoom)
        # Not while grains are selected
        # Not a pick event (don't put point prompts on existing grains)
        if (False    
                or event.dblclick
                or self.box_selector.get_active()
                or self.canvas.toolbar.mode != ''
                or len(self.selected_grains) > 0
                or self.last_pick == (round(event.xdata), round(event.ydata))):
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
            self.update()
        else:
            # Apparently necessary if plot shown twice
            self.canvas.draw_idle()

    def ondraw(self, event:mpl.backend_bases.DrawEvent):
        ''' Update saved background whenever a full redraw is triggered. '''
        if self.blit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

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
        elif event.key == 'i':
            self.toggle_info()
        elif event.key == 'm':
            self.merge_grains()
        elif event.key == 'z':
            self.undo_grain()
        elif event.key == 'escape':
            self.unselect_all()
        elif event.key == 'shift':
            self.unselect_grains()
            self.activate_box()
        else:
            return
        # Update canvas
        if self.blit:
            self.update()
        else:
            # Apparently necessary if plot shown twice
            self.canvas.draw_idle()

    def onkeyup(self, event:mpl.backend_bases.KeyEvent):
        # Cancel box selector if too small
        if event.key == 'shift':
            self.activate_box(False)
            xmin, xmax, ymin, ymax = self.box_selector.extents
            area = abs(xmax-xmin) * abs(ymax-ymin)
            if area < self.minspan ** 2:
                self.box_selector.clear()

    def onpick(self, event:mpl.backend_bases.PickEvent):
        '''
        Handle clicking on an existing grain to select/unselect it.
        
        Parameters
        ----------
        event
            Matplotlib event
        '''
        mouseevent = event.mouseevent
        # No doubleclicks, no toolbar, no scroll clicks, no box selection
        if (mouseevent.dblclick
                or self.canvas.toolbar.mode != ''
                or mouseevent.button == 2
                or self.box_selector.get_active()):
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
        # Update info box
        self.update_info()
        # Update canvas
        if self.blit:
            self.update()
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

    def savefig(self, fn):
        self.fig.savefig(fn, bbox_inches='tight', pad_inches=0)


# Load/save ---
def load_image(fn):
    return np.array(keras.utils.load_img(fn))


def load_grains(fn):
    grains = []
    for grain in pd.read_csv(fn).iterrows():
        out_coords = []
        for coord in grain[1].iloc[1][10:-2].split(', '):
            x, y = coord.split(' ')
            out_coords.append((float(x), float(y)))
        grains.append(shapely.Polygon(out_coords))
    grains = [Grain(np.array(p.exterior.xy)) for p in grains]
    return grains


def save_grains(fn, grains):
    pd.DataFrame([g.get_polygon() for g in grains]).to_csv(fn)


def get_summary(grains):
    return pd.concat([g.data for g in grains], axis=1).T


def save_summary(fn, grains):
    get_summary(grains).to_csv(fn)


def get_histogram(grains):
    df = get_summary(grains)
    fig, ax = segmenteverygrain.plot_histogram_of_axis_lengths(
        df['major_axis_length'], df['minor_axis_length'])
    return fig, ax
    

def save_histogram(fn, grains):
    fig, ax = get_histogram(grains)
    fig.savefig(fn, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def get_mask(grains, image):
    grains = [g.get_polygon() for g in grains]
    rasterized_image, mask = segmenteverygrain.create_labeled_image(
        grains, image)
    mask = keras.utils.img_to_array(mask)
    return mask


def save_mask(fn, grains, image, scale=False):
    keras.utils.save_img(fn, get_mask(grains, image), scale=scale)


# Point count ---
def make_grid(image:np.ndarray, spacing:int):
    img_y, img_x = image.shape[:2]
    pad_x = img_x % spacing
    pad_y = img_y % spacing
    x_vals = np.arange(round(pad_x / 2), img_x, spacing)
    y_vals = np.arange(round(pad_y / 2), img_y, spacing)
    xs, ys = np.meshgrid(x_vals, y_vals)
    points = shapely.points(xs, ys).flatten()
    return points, xs, ys


def filter_grains_by_points(grains:list, points:list) -> tuple:
    point_found = []
    point_grains = []
    for point in points:
        for grain in grains.copy():
            if grain.get_polygon().contains(point):
                grains.remove(grain)
                point_grains.append(grain)
                point_found.append(True)
                break
        else:
            point_found.append(False)
    return point_grains, point_found