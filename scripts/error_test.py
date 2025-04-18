# pip
import numpy as np
import pandas as pd
from tqdm import tqdm
# local
import segmenteverygrain.interactions as si


# Comparison methods
def absolute(a, b):
    return abs(a - b)


def relative(a, b):
    return 100 * abs((a - b) / (a + b)) / 2


def absolute_angle(a, b):
    diff = absolute(a, b)
    return min(diff, np.pi - diff) * 180. / np.pi


class Comparison():
    def __init__(self, f: object, tolerance: float):
        self.f = f
        self.tolerance = tolerance


metrics = {
    'centroid-0': Comparison(absolute, 2.),
    'centroid-1': Comparison(absolute, 2.),
    'area': Comparison(relative, 2.),
    'perimeter': Comparison(relative, 2.),
    'orientation': Comparison(absolute_angle, 5.),
    'major_axis_length': Comparison(relative, 1.),
    'minor_axis_length': Comparison(relative, 2.),
    # 'max_intensity-0': Comparison(absolute, 5.),
    # 'min_intensity-0': Comparison(absolute, 10.),
    'mean_intensity-0': Comparison(relative, 2.),
    # 'max_intensity-1': Comparison(absolute, 5.),
    # 'min_intensity-1': Comparison(absolute, 10.),
    'mean_intensity-1': Comparison(relative, 2.),
    # 'max_intensity-2': Comparison(absolute, 5.),
    # 'min_intensity-2': Comparison(absolute, 10.),
    'mean_intensity-2': Comparison(relative, 2.)
}


# Load test input
image = si.load_image('examples/torrey_pines.jpg')
grains = si.load_grains(
    'examples/interactive/torrey_pines_grains.geojson', image=image)


# Compare measurement methods to find max error for each metric
results = []
for i, grain in enumerate(grains):
    a = grain.measure(raster=True)
    a.name = 'raster'
    b = grain.measure(raster=False)
    b.name = 'polygon'
    error = {}
    for metric, comparison in metrics.items():
        error[metric] = comparison.f(a[metric], b[metric])
        if error[metric] > comparison.tolerance:
            print(f"{i}: Error in {metric}: {error[metric]}")
            x = pd.DataFrame([a, b])
            with pd.option_context('display.max_columns', None):
                print(x)
    results.append(pd.Series(error))

df = pd.DataFrame(results)
print('max\n', df.max())
print('mean\n', df.mean())


# import matplotlib.pyplot as plt
# plot = si.GrainPlot(grains, image)

# for grain in grains:
#     grain.draw_axes(plot.ax)

# plot.activate()
# plt.show(block=True)


# Measure time
for v in (False, True):
    for grain in tqdm(grains):
        grain.measure(raster=v)
