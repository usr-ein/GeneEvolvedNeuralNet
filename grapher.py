import matplotlib.pyplot as plt
import numpy as np
class Grapher():
    """docstring for Grapher"""
    def __init__(self, title, xlabel, ylabel, color='red'):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.xlim = 150
        self.xstep = 5

        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        self.color = color
        self.label = None

    def updateLabel(self, text, pos=(0,0), fontsize=10):
        if self.label:
            self.label.remove()
        self.label = plt.text(pos[0], pos[1], text, fontsize=fontsize)

    def newPoint(self, xpoint, ypoint):
        plt.axis()
        plt.scatter(xpoint, ypoint, c='red', s=4.2)
        plt.xticks(np.arange(0, xpoint, self.xstep), rotation=60, fontsize = 7)
        if xpoint > self.xlim:
            plt.xlim([xpoint-self.xlim, xpoint])
        plt.pause(0.005)

    def show(self):
        plt.show()

    def plotFuncs(self, funcs, arange=(0,1)):
        plt.clf()
        plt.cla()
        cmap = plt.get_cmap('jet_r')
        x = np.arange(arange[0], arange[1], 0.01)
        for i in range(len(funcs)):
            plt.plot(x, np.vectorize(funcs[i])(x), c=cmap(i/len(funcs)), label="Function {}".format(i))
        plt.legend()
        plt.pause(0.005)
        plt.show()