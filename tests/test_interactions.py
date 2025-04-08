import unittest
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import segmenteverygrain as seg
import segmenteverygrain.interactions as si
import shapely


class TestGrainObject(unittest.TestCase):

    def setUp(self):
        # Create a mock image
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create a mock Axes object with patches
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image)
        self.patches = [
            mpl.patches.Polygon(
                np.array([[10, 10], [25, 15], [25, 25], [15, 25]]), closed=True),
            mpl.patches.Polygon(
                np.array([[30, 30], [40, 30], [40, 40], [30, 40]]), closed=True)
        ]
        for patch in self.patches:
            self.ax.add_patch(patch)
        plt.axis('equal')

        # Create Shapely polygons directly
        self.polygons = [shapely.Polygon(p.get_xy()) for p in self.patches]

        # Create and draw Grain objects
        self.grains = si.polygons_to_grains(self.polygons)
        for grain in self.grains:
            grain.draw_patch(self.ax)

    def test_init(self):
        # Check the number of grains
        self.assertEqual(len(self.grains), len(self.patches))

        # Check the type of grains
        for grain in self.grains:
            self.assertIsInstance(grain, si.Grain)

    def test_drawing(self):
        for grain, patch in zip(self.grains, self.patches):
            # Verify coordinates
            for grain_xy, patch_xy in zip(grain.patch.get_xy(), patch.get_xy()):
                self.assertTrue((grain_xy == patch_xy).all())

    def test_measure(self):
        for grain, poly in zip(self.grains, self.polygons):
            # ring = poly.exterior
            # Get grain data
            grain.measure(self.image)
            data = grain.data
            # Centroid
            centroid = poly.centroid
            self.assertTrue(abs(data['centroid-1'] - centroid.x) < 1.)
            self.assertTrue(abs(data['centroid-0'] - centroid.y) < 1.)
            # Area
            self.assertEqual(data['area'], poly.area)
            # Perimeter
            self.assertEqual(data['perimeter'], poly.length)
            # Orientation
            poly.oriented_envelope


class TestGrainPlot(unittest.TestCase):

    def setUp(self):
        pass

    def test_create_plot(self):
        pass
