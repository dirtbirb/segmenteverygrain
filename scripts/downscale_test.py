import segmenteverygrain.interactions as si


image = si.load_image('examples/torrey_pines.jpg')
grains = si.load_grains(
    'examples/interactive/torrey_pines_grains.geojson',
    image=image)

testgrain = grains[0]
testdata = testgrain.measure()
testxy = testgrain.xy

max_dim = 320
plot = si.GrainPlot(
    grains, image, image_max_size=(max_dim, max_dim))


print(testxy[0][0])
print(plot.grains[0].xy[0][0])
print(grains[0].xy[0][0])


# print(testdata.head())
# print(plot.grains[0].data.head())
# print(grains[0].data.head())
