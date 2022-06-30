import glob
import os
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']


class OursDataset(Dataset):
    def __init__(self, path, transform=None, text_transform = None,cache_images=False, use_instance_id=False):
        try:
            f = []  # image files
            labels = []
            self.classes = []
            l_sn = 0
            for good in os.listdir(path):
                t = os.listdir(os.path.join(path, good))
                labels.extend([l_sn] * len(t))
                l_sn += 1
                self.classes.append(good)
                f += glob.iglob(os.path.join(path, good)+os.sep+"*.*")
            self.imgs = [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats]
        except Exception as e:
            raise Exception('Error loading data from %s: %s' % (path, e))
        image_number = len(self.imgs)
        assert image_number > 0, 'No images found in %s' % (path)
        print("The number of fashion images is %d"%image_number)
        self.labels = labels

        # Define text ################FFFFFFFFFFFFF
        self.image_texe_files = [x.replace('image', 'image_text').replace(os.path.splitext(x)[-1], '.txt') for x in
                            self.imgs]
        self.n = len(self.imgs)
        self.index = 0
        self.transform = transform

    def __len__(self):
        return self.n

    def __next__(self):
        if self.index >= self.n:
            self.index = 0
            raise StopIteration
        image, label, text =  self.__getitem__(self.index)
        self.index += 1
        return  image, label, text

    def __iter__(self):
        return self

    def __getitem__(self, index):
        try:
            img = self.load_image(index)
        except Exception as e:
            print(e)
            raise Exception('Error loading image: %s' % (self.imgs[index]))

        if self.transform is not None:
                img = self.transform(img)
        with open(self.image_texe_files[index]) as vec_file:
            ret_txt = np.array(vec_file.read().splitlines(), dtype=np.float32).reshape((100, 1))

        return img, self.labels[index], ret_txt  #####################FFFFFFFFFFFFFFFFFFFFF

    def load_image(self, index):
        p = Path(self.imgs[index])
        img = Image.open(str(p)).convert('RGB')
        # img =torch.from_numpy(np.array(img))
        return img

if __name__ == '__main__':
    dt = OursDataset(r'./target/image/train')
    for a,b,c in dt:
        print(a,b,c)
