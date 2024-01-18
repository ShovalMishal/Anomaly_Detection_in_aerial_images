import json
import os

import numpy as np
import torch
import torchvision.utils
from matplotlib.collections import PatchCollection
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Circle
import DOTA_devkit.dota_utils as util
from torch.utils.data import Dataset
from collections import defaultdict
import cv2
from PIL import Image
from tqdm import tqdm


def base_transform(image):
    return image


class DotaDataset(Dataset):
    def __init__(self, is_train: bool = True, basepath: str = "./DOTA-v2/", transform=base_transform, target_transform=None,
                 annfiles_path=os.path.join('labelTxt', 'DOTA-v2.0_train_hbb')):
        self.is_train = is_train
        self.transform = transform
        self.target_transform = target_transform
        self.folder_path = os.path.join(basepath, "train") if self.is_train else os.path.join(basepath, "val")
        self.labelpath = annfiles_path
        self.imagepath = os.path.join(self.folder_path, 'images')
        self.meta_data_path = os.path.join(self.folder_path, 'labelTxt/meta')
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
        meta_data = self.load_metadata(img_id)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            anns = self.target_transform(anns)

        return image, anns, meta_data

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

    @staticmethod
    def parse_metadata_file(data: str):
        metadata_dict = {}
        lines = data.split('\n')
        for line in lines:
            k = line.split(':')[0]
            v = line.split(':')[-1]
            if k == 'gsd':
                try:
                    v = float(v)
                except:
                    v = None
            metadata_dict = {k: v}
        return metadata_dict


    def load_metadata(self, imgid: str):
        """
        :param imgids: integer ids specifying img
        :return: image metadata dict
        """
        filename = os.path.join(self.meta_data_path, imgid + '.txt')
        with open(filename, 'r') as f:
            data = f.read()
        meta_data = self.parse_metadata_file(data)
        meta_data['imgid'] = imgid
        return meta_data

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


def extract_patches_from_images(output_dir: str = '/home/shoval/Documents/Repositories/data/DOTAV2_fg_patches/val'):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metadata'), exist_ok=True)
    dataset = DotaDataset(is_train = False, basepath="/home/shoval/Documents/Repositories/data/DOTAV2",
                          transform=base_transform, target_transform=None,
                          annfiles_path=os.path.join("/home/shoval/Documents/Repositories/data/DOTAV2",
                                                     'val', 'labelTxt', 'DOTA-v2.0_val_hbb'))
    for class_name in list(dataset.catToImgs.keys()):
        os.makedirs(os.path.join(output_dir, 'images', class_name), exist_ok=True)
    # image, annotations, metadata = dataset[0]
    for image, annotations, metadata in tqdm(dataset):
        height, width = image.shape[0], image.shape[1]
        for ann_index, ann in enumerate(annotations):
            poly = ann['poly']
            min_x = min(max(0, int(min([x for (x, y) in poly]))), width)
            min_y = min(max(0, int(min([y for (x, y) in poly]))), height)
            max_x = min(max(0, int(max([x for (x, y) in poly]))), width)
            max_y = min(max(0, int(max([y for (x, y) in poly]))), height)
            # crop patch
            if min_y > height or min_y >= max_y:
                continue
            if min_x > width or min_x >= max_x:
                continue
            patch = image[min_y:max_y, min_x:max_x, ...]
            # put patch in its class folder
            class_name = ann['name']
            patch_name = f"{metadata['imgid']}_{ann_index}"
            patch_path = os.path.join(output_dir, 'images', class_name, patch_name + '.png')

            patch = Image.fromarray(patch)
            patch.save(patch_path)
            # save the patch metadata
            metadata['class'] = class_name
            patch_metadata_path = os.path.join(output_dir, 'metadata', patch_name + '.json')
            with open(patch_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)


if __name__ == '__main__':
    dataset = extract_patches_from_images()
    # print(dataset[0])
