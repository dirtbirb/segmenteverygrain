import matplotlib.pyplot as plt
import segment_anything
import segmenteverygrain.interactions as si


FIGSIZE = (12, 8)   # in
PX_PER_M = 1        # px/m; be sure not to convert units twice!


# Load test image
fn = 'torrey_pines_beach_image.jpeg'
image = si.load_image(fn)

# Load SAM
fn = 'sam_vit_h_4b8939.pth'
sam = segment_anything.sam_model_registry['default'](checkpoint=fn)
predictor = segment_anything.SamPredictor(sam)
predictor.set_image(image)

# Load grains
fn = './output/test_auto_grains.csv'
grains = si.load_grains(fn)
# grains = []


# Display editing interface
plot = si.GrainPlot(
    grains, 
    image=image, 
    predictor=predictor,
    blit=True,
    figsize=FIGSIZE
)
plot.activate()
plt.show(block=True)
plot.deactivate()


# Save results
fn = './output/test_edit'
# Convert units
pass
# Grain shapes
# for g in grains:
#     g.measure(image=image)
si.save_grains(fn + '_grains.csv', plot.grains)
# Grain image
plot.savefig(fn + '_grains.jpg')
# Summary data
si.save_summary(fn + '_summary.csv', plot.grains, px_per_m=PX_PER_M)
# Summary histogram
si.save_histogram(fn + '_summary.jpg', plot.grains, px_per_m=PX_PER_M)
# Training mask
si.save_mask(fn + '_mask.png', plot.grains, image, scale=False)
si.save_mask(fn + '_mask.jpg', plot.grains, image, scale=True)