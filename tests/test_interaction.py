import matplotlib.pyplot as plt

def onclick(event):
    print(event.xdata, event.ydata)
    ax.scatter(event.xdata, event.ydata)
    plt.draw()

for i in range(2):
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4])
    # ax.autoscale(enable=False)
    canvas = plt.gca().figure.canvas
    canvas.mpl_connect('button_press_event', onclick)
    with plt.ion():
        plt.show(block=True)