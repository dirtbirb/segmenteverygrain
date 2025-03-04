import keras.utils
import logging
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
from tqdm import tqdm


# Images larger than this will be downscaled
# 4k resolution is (2160, 4096)
IMAGE_MAX_SIZE = np.asarray((2160, 4096))

# Init logger
logger = logging.getLogger(__name__)

# Speed up rendering a little?
mplstyle.use('fast')

# HACK: Bypass large image restriction
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# HACK: Don't reset zoom level when pressing 'c' to create a grain
if 'c' in mpl.rcParams['keymap.back']:
    mpl.rcParams['keymap.back'].remove('c')

# HACK: Attach parent grain reference to the mpatches.Polygon class
# Makes it easy to use matplotlib "pick" event when clicking on a grain
mpatches.Polygon.grain = None


class Grain(object):
    ''' Stores data and plot representation for a single grain. '''
        
    def __init__(self, xy: np.ndarray, data: pd.Series=None):
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
        
        # Grain properties to calculate (skimage)
        # {name: dimensionality}
        self.region_props = {
            'area': 2,
            'centroid': 0,
            'major_axis_length': 1,
            'minor_axis_length': 1, 
            'orientation': 0,
            'perimeter': 1,
            'max_intensity': 0,
            'mean_intensity': 0,
            'min_intensity': 0
        }
        
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

    @property
    def polygon(self) -> shapely.Polygon:
        ''' Return a shapely.Polygon representing the matplotlib patch. '''
        return shapely.Polygon(self.xy.T)

    def measure(self, image: np.ndarray) -> pd.Series:
        '''
        Calculate grain information from image and matplotlib patch.
        Overwrites self.data.

        Parameters
        ----------
        image : np.ndarray
            Background image.
            Used to calculate region intensity in self.data.

        Returns
        -------
        self.data : pd.Series
            Row for a DataFrame containing computed grain info.
        '''
        # Get rasterized shape
        # TODO: Just use a patch of the image and then convert coords
        rasterized = rasterio.features.rasterize(
            [self.polygon], out_shape=image.shape[:2])
        # Calculate region properties
        data = pd.DataFrame(skimage.measure.regionprops_table(rasterized,
            intensity_image=image, properties=self.region_props.keys()))
        if len(data):
            self.data = data.iloc[0]
        else:
            logger.error(f'MEASURE ERROR {pd.DataFrame(data)}')
            self.data = pd.Series()
            return
        return self.data

    def draw_axes(self, ax: mpl.axes.Axes) -> dict[str: object]:
        # Compute grain data if it hasn't been done already
        if self.data is None:
            image = ax.get_images()[0].get_array()
            self.data = self.measure(image)
        data = self.data
        # Keep track of drawn objects
        artists = {}
        # Centroid
        x0, y0 = data['centroid-1'], data['centroid-0']
        artists['centroid'] = ax.plot(x0, y0, '.k')
        # Major axis
        orientation = data['orientation']
        x = x0 - np.sin(orientation) * 0.5 * data['major_axis_length']
        y = y0 - np.cos(orientation) * 0.5 * data['major_axis_length']
        artists['major'] = ax.plot((x0, x), (y0, y), '-k')
        # Minor axis
        x = x0 + np.cos(orientation) * 0.5 * data['minor_axis_length']
        y = y0 - np.sin(orientation) * 0.5 * data['minor_axis_length']
        artists['minor'] = ax.plot((x0, x), (y0, y), '-k')
        # Return dict of artist objects, potentially useful for blitting
        return artists

    def draw_patch(self, ax: mpl.axes.Axes) -> mpatches.Polygon:
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
                
        # Create patch (filled polygon)
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
        # Compute grain data for the info box
        if self.data is None:
            image = ax.get_images()[0].get_array()
            self.data = self.measure(image)
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

    def rescale(self, scale: float):
        '''
        Scale polygon coordinates and measurements by given scale factor.
        '''
        # Convert coordinates
        self.xy *= scale
        # Convert data
        if type(self.data) is type(None):
            return
        # For each type of measured value,
        for k, d in self.region_props.items():
            # If it has any length dimensions associated with it
            if d:
                # Scale those values according to their length dimensions
                for col in [c for c in self.data.keys() if k in c]:
                    self.data[col] *= scale ** d 


class GrainPlot(object):
    ''' Interactive plot to create, delete, and merge grains. '''

    def __init__(self,
            grains: list = [],
            image: np.ndarray = None, 
            predictor = None,
            blit: bool = True,
            minspan: int = 10,
            image_alpha: float = 1.,
            image_max_size: tuple = IMAGE_MAX_SIZE,
            **kwargs):
        '''
        Parameters
        ----------
        grains: list
            List of grains with xy data to plot over the backround image.
        image: np.ndarray
            Image under analysis, displayed behind identified grains.
        predictor:
            SAM predictor used to create new grains.
        blit: bool, default True
            Whether to use blitting (much faster, potentially buggy).
        minspan: int, default 10
            Minimum size for box selector tool.
        image_alpha: float, default 1.0
            Alpha value for background image, passed to imshow().
        image_max_size: (y, x)
            Images larger than this will be downscaled for display.
            Grain creation and measurement will still use the full image.
        kwargs: dict
            Keyword arguments to pass to plt.figure().
        '''
        logger.info('Creating GrainPlot...')

        # Input
        self.image = image
        self.predictor = predictor
        self.blit = blit
        self.minspan = minspan

        # Events
        self.cids = []
        self.events = {
            'button_press_event': self.onclick,
            'draw_event': self.ondraw,
            'key_press_event': self.onkey,
            'key_release_event': self.onkeyup,
            'pick_event': self.onpick
        }
        
        # Interaction history
        self.ctrl_down = False
        self.last_pick = (0, 0)
        self.points = []
        self.point_labels = []
        self.created_grains = []
        self.selected_grains = []
        
        # Plot
        self.fig = plt.figure(**kwargs)
        self.canvas = self.fig.canvas
        self.ax = self.fig.add_subplot(aspect='equal', xticks=[], yticks=[])
        
        # Background image
        self.scale = 1.
        self.display_image = image
        if isinstance(image, np.ndarray):
            # Downscale image if needed
            max_size = np.asarray(image_max_size)
            if image.shape[0] > max_size[0] or image.shape[1] > max_size[1]:
                logger.info('Downscaling large image for display...')
                self.scale = np.max(max_size / image.shape[:2])
                self.display_image = skimage.transform.rescale(
                    image, self.scale, anti_aliasing=True, channel_axis=2)
                logger.info(f'Downscaled image to {self.scale} of original.')
            # Show image
            self.ax.imshow(self.display_image, alpha=image_alpha)
            self.ax.autoscale(enable=False)
        self.fig.tight_layout(pad=0)
        
        # Interactive toolbar: inject clear_all before any zoom/pan changes
        # Avoids manual errors and bugs with blitting
        toolbar = self.canvas.toolbar
        toolbar._update_view = self._clear_before(toolbar._update_view)
        toolbar.release_pan = self._clear_before(toolbar.release_pan)
        toolbar.release_zoom = self._clear_before(toolbar.release_zoom)
        
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
        self.showing_info = True

        # Draw grains and initialize plot
        logger.info('Drawing grains.')
        self.grains = grains
        for grain in tqdm(self._grains):
            grain.draw_patch(self.ax)
        if blit:
            self.canvas.draw()
        logger.info('GrainPlot created!')

    # Display helpers ---
    def _clear_before(self, f: object) -> object:
        ''' Wrap a function to call clear_all before it. '''
        def newf(*args, **kwargs):
            if self.blit:
                self.clear_all()
            return f(*args, **kwargs)
        return newf

    def update(self):
        ''' Blit background image and draw animated art. '''
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

    def show_info(self, show: bool=None) -> bool:
        ''' Toggle or set info box display. '''
        # Toggle info box flag
        self.showing_info = show or not self.showing_info
        # Show info box if requested and grains selected
        self.info.set_visible(self.showing_info and len(self.selected_grains))
        return self.showing_info

    def clear_info(self):
        ''' Alias for self.show_info(False). '''
        self.show_info(False)

    def update_info(self):
        ''' Update info box based on last selected grain. '''
        # Hide info box if no grains selected
        if not len(self.selected_grains):
            self.info.set_visible(False)
            return
        # Determine offset based on position within plot
        grain = self.selected_grains[-1]
        ext = grain.patch.get_extents()
        img_x, img_y = self.canvas.get_width_height()
        x = -0.1 if (ext.x1 + ext.x0) / img_x > 1 else 1.1
        y = -0.1 if (ext.y1 + ext.y0) / img_y > 1 else 1.1
        # Extra offset to avoid covering up small grains
        if abs(ext.y1 - ext.y0) < img_y / 20:
            if y <= 0:
                y -= 1.4
            else:
                y += 1.4
        # Update position
        self.info.xy = (x, y)
        self.info.xycoords = grain.patch
        # Update text
        text = (f"Major: {grain.data['major_axis_length']:.0f}px\n"
                f"Minor: {grain.data['minor_axis_length']:.0f}px\n"
                f"Area: {grain.data['area']:.0f}px")
        self.info.set_text(text)
        # Show info box if requested
        if self.showing_info:
            self.info.set_visible(True)

    # Selection helpers ---
    def activate_box(self, activate=True):
        ''' Activate/deactivate selection box. '''
        self.clear_grains()
        alpha = 0.4 if activate else 0.2
        box = self.box_selector
        box.set_handle_props(alpha=alpha, visible=activate)
        box.set_props(alpha=alpha)
        box.set_active(activate)

    def clear_box(self):
        self.box_selector.clear()
    
    def set_point(self, xy:tuple, foreground:bool=True):
        ''' Set point prompt, either foreground or background. '''
        color = 'lime' if foreground else 'red'
        new_point = mpatches.Circle(
            xy, radius=5, color=color, animated=self.blit)
        self.ax.add_patch(new_point)
        self.points.append(new_point)
        self.point_labels.append(foreground)

    def clear_points(self):
        ''' Clear all prompt points. '''
        for point in self.points:
            point.remove()
        self.points = []
        self.point_labels = []

    def clear_grains(self):
        ''' Unselect all selected grains. '''
        for grain in self.selected_grains:
            grain.select()
        self.selected_grains = []
        self.update_info()

    def clear_all(self):
        ''' Clear prompts, unselect all grains, and hide the info box. '''
        self.clear_box()
        self.clear_grains()
        self.clear_info()
        self.clear_points()

    # Manage grains ---
    @property
    def grains(self) -> list:
        ''' Return copy of self.grains in full-image coordinates. '''
        grains = self._grains.copy()
        if self.scale != 1.:
            for grain in grains:
                grain.rescale(1 / self.scale)
        return grains

    @grains.setter
    def grains(self, grains: list):
        ''' Save copy of given grains list in display-image coordinates. '''
        grains = grains.copy()
        if self.scale != 1:
            for grain in grains:
                grain.rescale(self.scale)
        self._grains = grains

    def create_grain(self):
        ''' Attempt to detect a grain based on given prompts. '''
        # Interpret point prompts
        if len(self.points):
            points = [
                np.asarray(p.get_center()) / self.scale for p in self.points]
            point_labels = self.point_labels
        else:
            points = None
            point_labels = None
        # Interpret box prompt
        if self.box_selector._selection_completed:
            xmin, xmax, ymin, ymax = np.asarray(self.box_selector.extents) / self.scale
            box = np.asarray((xmin, ymin, xmax, ymax))
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
        grain.rescale(self.scale)
        grain.draw_patch(self.ax)
        self._grains.append(grain)
        self.created_grains.append(grain)
        # Clear prompts
        self.clear_all()
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
            self._grains.remove(grain)
            if grain in self.created_grains:
                self.created_grains.remove(grain)
        self.clear_all()
        # Update background
        if self.blit:
            self.canvas.draw()

    def hide_grains(self, hide: bool=True):
        ''' Hide or unhide selected grains. '''
        show = not hide
        # Hide other elements if needed
        if hide:
            self.clear_box()
            self.clear_points()
            # Only hide info box temporarily
            self.info.set_visible(False)
        # Show/hide selected grains
        for grain in self.selected_grains:
            grain.patch.set_visible(show)
        # Update background
        if self.blit:
            self.canvas.draw()
        # Set selected grains to default color when hidden,
        # or restore selected color when unhidden.
        # Avoids drawing wrong color into background on unhide.
        for grain in self.selected_grains:
            grain.select()
        # Show info box again if hidden
        if self.showing_info and show and len(self.selected_grains):
            self.info.set_visible(True)

    def merge_grains(self):
        ''' Attempt to merge all selected grains. '''
        # Verify there are at least two grains selected to merge
        if len(self.selected_grains) < 2:
            return
        # Find vertices of merged grains using Shapely
        poly = shapely.unary_union(
            [g.polygon for g in self.selected_grains])
        # Verify grains actually overlap, otherwise reject selections
        if isinstance(poly, shapely.MultiPolygon):
            self.clear_grains()
            return
        # Make new merged grain
        new_grain = Grain(poly.exterior.xy)
        new_grain.draw_patch(self.ax)
        self._grains.append(new_grain)
        self.created_grains.append(new_grain)
        # Delete old constituent grains (since they are still selected)
        self.delete_grains()

    def undo_grain(self):
        ''' Remove latest created grain. '''
        # TODO: Also allow undoing grain deletions
        # Verify that there is a grain to undo
        if len(self.created_grains) < 1:
            return
        # Select and remove latest grain
        self.clear_all()
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
        # Ignore the following:
        # double-clicks
        # toolbar interactions (pan/zoom)
        # clicks with ctrl held (hiding grains)
        # clicks with box selection active
        # clicks with grains selected
        # Pick events (don't put point prompts on existing grains)
        if (event.dblclick   
                or self.canvas.toolbar.mode != ''
                or self.ctrl_down
                or self.box_selector.get_active()
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
        # Ignore keypresses while grains hidden or during box selection
        if self.ctrl_down or self.box_selector.get_active():
            return
        key = event.key
        if key == 'c':
            self.create_grain()
        elif key == 'd' or key == 'delete':
            self.delete_grains()
        elif key == 'i':
            self.show_info()
        elif key == 'm':
            self.merge_grains()
        elif key == 'z':
            self.undo_grain()
        elif key == 'control':
            self.ctrl_down = True
            self.hide_grains()
        elif key == 'escape':
            self.clear_all()
        elif key == 'shift':
            self.activate_box()
        else:
            # Don't update canvas if no key pressed
            return
        # Update canvas
        if self.blit:
            self.update()
        else:
            # Apparently necessary if plot shown twice
            self.canvas.draw_idle()

    def onkeyup(self, event:mpl.backend_bases.KeyEvent):
        key = event.key
        if key == 'control':
            self.ctrl_down = False
            self.hide_grains(False)
        elif key == 'shift':
            # Deactivate box selector
            self.activate_box(False)
            # Cancel box if too small (based on minspan)
            xmin, xmax, ymin, ymax = self.box_selector.extents
            if min(abs(xmax-xmin), abs(ymax-ymin)) < self.minspan:
                self.box_selector.clear()
        else:
            return
        # Update canvas
        if self.blit:
            self.update()
        else:
            # Apparently necessary if plot shown twice
            self.canvas.draw_idle()

    def onpick(self, event:mpl.backend_bases.PickEvent):
        '''
        Handle clicking on an existing grain to select/unselect it.
        
        Parameters
        ----------
        event
            Matplotlib event
        '''
        mouseevent = event.mouseevent
        # Ignore the following:
        # Double clicks
        # Toolbar interactions (pan/zoom)
        # Scroll clicks
        # Clicks while grains hidden (ctrl held)
        # Clicks during box selection
        if (mouseevent.dblclick
                or self.canvas.toolbar.mode != ''
                or mouseevent.button == 2
                or self.ctrl_down
                or self.box_selector.get_active()):
            return
        # Save click location to block attempts to set a point prompt
        self.last_pick = (round(mouseevent.xdata), round(mouseevent.ydata))
        # Only pick on left-click and when no point prompts exist
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
    def savefig(self, fn: str):
        ''' Save figure to disk. '''
        self.fig.savefig(fn, bbox_inches='tight', pad_inches=0)


# Load/save ---
def load_image(fn: str) -> np.ndarray:
    ''' Load an image as a numpy array. '''
    return np.array(keras.utils.load_img(fn))


def load_grains(fn: str) -> list:
    ''' Load grain boundaries from a csv. '''
    grains = []
    for grain in pd.read_csv(fn).iterrows():
        out_coords = []
        for coord in grain[1].iloc[1][10:-2].split(', '):
            x, y = coord.split(' ')
            out_coords.append((float(x), float(y)))
        grains.append(shapely.Polygon(out_coords))
    grains = [Grain(np.array(p.exterior.xy)) for p in grains]
    return grains


def save_grains(fn: str, grains: list):
    ''' Save grain boundaries to a csv. '''
    pd.DataFrame([g.polygon for g in grains]).to_csv(fn)


def get_summary(grains: list, px_per_m: float=1.) -> pd.DataFrame:
    ''' Get grain measurements as a pandas DataFrame. '''
    # Get DataFrame
    df = pd.concat([g.data for g in grains], axis=1).T
    # Convert units
    # HACK: Applies first grain's region_props to all
    for k, d in grains[0].region_props.items():
        if d:
            for col in [c for c in df.columns if k in c]:
                df[col] /= px_per_m ** d 
    return df


def save_summary(fn: str, grains: list, px_per_m: float=1.):
    ''' Save grain measurements as a csv. '''
    get_summary(grains, px_per_m).to_csv(fn)


def get_histogram(grains: list, px_per_m: float=1.) -> tuple[object, object]:
    ''' Get a histogram of axis lengths as a matplotlib plot. '''
    df = get_summary(grains, px_per_m)
    fig, ax = segmenteverygrain.plot_histogram_of_axis_lengths(
        df['major_axis_length'], df['minor_axis_length'])
    return fig, ax
    

def save_histogram(fn: str, grains: list, px_per_m: float=1.):
    ''' Save a histogram of axis lengths as an image. '''
    fig, ax = get_histogram(grains, px_per_m)
    fig.savefig(fn, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def get_mask(grains: list, image: np.ndarray) -> np.ndarray:
    ''' Get a rasterized, binary mask of grain shapes as np.ndarray. '''
    grains = [g.polygon for g in grains]
    rasterized_image, mask = segmenteverygrain.create_labeled_image(
        grains, image)
    mask = keras.utils.img_to_array(mask)
    return mask


def save_mask(fn: str, grains: list, image: np.ndarray, scale: bool=False):
    ''' Save binary mask of grain shapes, optionally scaled to 0-255. '''
    keras.utils.save_img(fn, get_mask(grains, image), scale=scale)


# Point count ---
def make_grid(image: np.ndarray, spacing: int) -> tuple[list, list, list]:
    ''' Construct a grid of measurement points given an image and spacing. '''
    img_y, img_x = image.shape[:2]
    pad_x = img_x % spacing
    pad_y = img_y % spacing
    x_vals = np.arange(round(pad_x / 2), img_x, spacing)
    y_vals = np.arange(round(pad_y / 2), img_y, spacing)
    xs, ys = np.meshgrid(x_vals, y_vals)
    points = shapely.points(xs, ys).flatten()
    return points, xs, ys


def filter_grains_by_points(grains: list, points: list) -> tuple[list, list]:
    ''' Return a list of grains at specified points. '''
    point_found = []
    point_grains = []
    for point in points:
        for grain in grains.copy():
            if grain.polygon.contains(point):
                # Remove grain from list so that it's not found twice
                grains.remove(grain)
                # Save detected grain
                point_grains.append(grain)
                # Record that a grain was found at this point
                point_found.append(True)
                break
        else:
            # Record that no grain was found at this point
            point_found.append(False)
    return point_grains, point_found