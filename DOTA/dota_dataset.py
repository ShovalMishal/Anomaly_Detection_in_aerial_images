import os

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Circle
import DOTA_devkit.dota_utils as util
from torch.utils.data import Dataset
from collections import defaultdict
import cv2


class DotaDataset(Dataset):
    def __init__(self, is_train: bool = True, basepath: str = "./DOTA-v2/", transform=None, target_transform=None):
        self.is_train = is_train
        self.transform = transform
        self.target_transform = target_transform
        self.folder_path = os.path.join(basepath, "train") if self.is_train else os.path.join(basepath, "val")
        self.labelpath = os.path.join(self.folder_path, 'labelTxt', 'DOTA-v2.0_train') if self.is_train \
            else os.path.join(self.folder_path, 'labelTxt', 'DOTA-v2.0_val')
        self.imagepath = os.path.join(self.folder_path, 'images')
        self.imgpaths = util.GetFileFromThisRootDir(self.labelpath)
        self.imglist = [util.custombasename(x) for x in self.imgpaths]
        self.catToImgs = defaultdict(list)
        self.ImgToAnns = defaultdict(list)
        self.create_index()

    def create_index(self):
        for filename in self.imgpaths:
            objects = util.parse_dota_poly(filename)
            imgid = util.custombasename(filename)
            self.ImgToAnns[imgid] = objects
            for obj in objects:
                cat = obj['name']
                self.catToImgs[cat].append(imgid)

    def __len__(self):
        return len(self.imgpaths)

    def __getitem__(self, idx):
        img_id = self.imglist[idx]
        image = self.load_image(img_id)
        anns = self.loadAnns(catNms=[], imgId=img_id)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            anns = self.target_transform(anns)
        return image, anns

    def get_img_ids_accord_categories(self, catNms=[]):
        """
        :param catNms: category names
        :return: all the image ids contain the categories
        """
        if len(catNms) == 0:
            return self.imglist
        else:
            imgids = []
            for i, cat in enumerate(catNms):
                if i == 0:
                    imgids = set(self.catToImgs[cat])
                else:
                    imgids &= set(self.catToImgs[cat])
        return list(imgids)

    def load_image(self, imgid: str):
        """
        :param imgids: integer ids specifying img
        :return: loaded img objects
        """
        filename = os.path.join(self.imagepath, imgid + '.png')
        img = cv2.imread(filename)
        return img

    def getImgIds(self, catNms=[]):
        """
        :param catNms: category names
        :return: all the image ids contain the categories
        """
        if len(catNms) == 0:
            return self.imglist
        else:
            imgids = []
            for i, cat in enumerate(catNms):
                if i == 0:
                    imgids = set(self.catToImgs[cat])
                else:
                    imgids &= set(self.catToImgs[cat])
        return list(imgids)

    def loadAnns(self, catNms=[], imgId = None, difficult=None):
        """
        :param catNms: category names
        :param imgId: the img to load anns
        :return: objects
        """
        objects = self.ImgToAnns[imgId]
        if len(catNms) == 0:
            return objects
        outobjects = [obj for obj in objects if (obj['name'] in catNms)]
        return outobjects

    def showAnns(self, objects, imgId, range):
        """
        :param catNms: category names
        :param objects: objects to show
        :param imgId: img to show
        :param range: display range in the img
        :return:
        """
        img = self.loadImgs(imgId)[0]
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
        plt.show()


