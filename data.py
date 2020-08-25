import glob
from loguru import logger
from torch.utils.data import Dataset, Sampler
import cv2
import numpy as np
from PIL import Image, ImageFile

class ImgDataset(Dataset):
    def __init__(self, img_path, transform):
        self.path = img_path

        self.input_lists = glob.glob(self.path + "/*")
        self.len = len(self.input_lists)
        self.transform = transform

        logger.info(f"Loading Imgs from {self.path}, len: {self.len}")

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        input_name = self.input_lists[idx]
        # input_ = cv2.imread(input_name)
        input_ = Image.open(str(input_name)).convert('RGB')
        input_ = self.transform(input_)
        return input_

def InfiniteSampler(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0

class InfiniteSamplerWrapper(Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampler(self.num_samples))

    def __len__(self):
        return 2 ** 31
