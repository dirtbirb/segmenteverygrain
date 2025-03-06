import numpy as np
import os
import pandas as pd
import segmenteverygrain.interactions as si
import shapely


def load_old_grains(fn: str) -> list:
    ''' Load grain info from old csv format. '''
    grains = []
    for grain in pd.read_csv(fn).iterrows():
        out_coords = []
        for coord in grain[1].iloc[1][10:-2].split(', '):
            x, y = coord.split(' ')
            out_coords.append((float(x), float(y)))
        grains.append(shapely.Polygon(out_coords))
    grains = [si.Grain(np.array(p.exterior.xy)) for p in grains]
    return grains


folder = 'output/'
for fn in os.listdir(folder):
    if not fn.endswith('_grains.csv'):
        continue
    print(fn)
    fn = folder + fn
    grains = load_old_grains(fn)
    fn = fn.split('.')[0] + '.geojson'
    si.save_grains(fn, grains)


print('Complete.')
