#The code is used for visulization, inspired from cocoapi
#  Licensed under the Simplified BSD License [see bsd.txt]

import os
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle
import numpy as np
import Anomaly_Detection_in_aerial_images.DOTA.DOTA_devkit.dota_utils as util
from collections import defaultdict
import cv2

EPS = 1e-2

class DOTAInference:
    def __init__(self, image_path, anns_path):
        self.image_path = image_path
        self.anns_path = anns_path

    def loadAnns(self):
        """
        :param catNms: category names
        :param imgId: the img to load anns
        :return: objects
        """
        objects = util.parse_dota_poly(self.anns_path)
        return objects

    def showAnns(self, objects, save_path=""):
        """
        :param catNms: category names
        :param objects: objects to show
        :param imgId: img to show
        :param range: display range in the img
        :return:
        """
        img = self.loadImg()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        width, height = img.shape[1], img.shape[0]
        fig = plt.figure("", frameon=False)
        dpi = fig.get_dpi()
        fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.imshow(img)
        plt.axis('off')
        ax = plt.gca()
        ax.set_autoscale_on(False)

        polygons = []
        color = []
        circles = []
        r = 5
        for obj in objects:
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            poly = obj['poly']
            polygons.append(Polygon(poly))
            color.append(c)
            point = poly[0]
            circle = Circle((point[0], point[1]), r)
            circles.append(circle)
        p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=2)
        ax.add_collection(p)
        p = PatchCollection(circles, facecolors='red')
        ax.add_collection(p)
        for obj in objects:
            ax.text(obj['poly'][0][0], obj['poly'][0][1], obj['name'], fontsize=5, ha='center')

        if save_path != "" :
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()


    def loadImg(self):
        """
        :param imgids: integer ids specifying img
        :return: loaded img objects
        """
        print('filename:', self.image_path)
        img = cv2.imread(self.image_path)
        return img

