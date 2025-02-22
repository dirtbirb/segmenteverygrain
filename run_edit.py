import matplotlib.pyplot as plt
import segment_anything
import segmenteverygrain.interactions as si


FIGSIZE = (12, 8)


# Load test image
fn = 'torrey_pines_beach_image.jpeg'
image = si.load_image(fn)

# Load SAM
fn = 'sam_vit_h_4b8939.pth'
sam = segment_anything.sam_model_registry['default'](checkpoint=fn)
predictor = segment_anything.SamPredictor(sam)
predictor.set_image(image)

# Load grains
fn = './output/test_grains.csv'
grains = si.load_grains(fn)


# Display editing interface
plot = si.GrainPlot(
    grains, 
    image=image, 
    predictor=predictor,
    figsize=FIGSIZE,
    blit=True
)
plot.activate()
plt.show(block=True)
plot.deactivate()


# Save results
fn = './output/test'
# Grain shapes
# for g in grains:
#     g.measure(image=image)
si.save_grains(fn + '_grains.csv', plot.grains)
# Grain image
plot.savefig(fn + '_grains.jpg')
# Summary data
si.save_summary(fn + '_summary.csv', plot.grains)
# Summary histogram
si.save_histogram(fn + '_summary.jpg', plot.grains)
# Training mask
si.save_mask(fn + '_mask.png', plot.grains, image, scale=False)
si.save_mask(fn + '_mask.jpg', plot.grains, image, scale=True)