# -*- coding: utf-8 -*-
# @Time    : 2022/12/21 15:45
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : dataset_builder.py
# @Software: PyCharm
import os
import json
import random
from pathlib import Path
from typing import Tuple

import torch
import rasterio
import numpy as np
import albumentations as A
from skimage import img_as_float
from torch.utils.data import Dataset, DataLoader

from mmseg.utils import register_all_modules

register_all_modules(init_default_scope=True)

palette = {
    1: '#db0e9a',
    2: '#938e7b',
    3: '#f80c00',
    4: '#a97101',
    5: '#1553ae',
    6: '#194a26',
    7: '#46e483',
    8: '#f3a60d',
    9: '#660082',
    10: '#55ff00',
    11: '#fff30d',
    12: '#e4df7c',
    13: '#3de6eb',
    14: '#ffffff',
    15: '#8ab3a0',
    16: '#6b714f',
    17: '#c5dc42',
    18: '#9999ff',
    19: '#000000'}

classes = {
    1: 'building',
    2: 'previous surface',
    3: 'impervious surface',
    4: 'bare soil',
    5: 'water',
    6: 'coniferous',
    7: 'deciduous',
    8: 'brushwood',
    9: 'vineyard',
    10: 'herbaceous vegetation',
    11: 'agricultural land',
    12: 'plowed land',
    13: 'swimming_pool',
    14: 'snow',
    15: 'clear cut',
    16: 'mixed',
    17: 'ligneous',
    18: 'greenhouse',
    19: 'other'}


def load_data(path_data, path_metadata, val_percent=0.8, use_metadata=True):
    def _gather_data(path_folders, path_metadata: str, use_metadata: bool, test_set: bool) -> dict:

        #### return data paths
        def get_data_paths(path, filter):
            for path in Path(path).rglob(filter):
                yield path.resolve().as_posix()

        #### encode metadata

        def coordenc_opt(coords, enc_size=32) -> np.array:
            d = int(enc_size / 2)
            d_i = np.arange(0, d / 2)
            freq = 1 / (10e7 ** (2 * d_i / d))

            x, y = coords[0] / 10e7, coords[1] / 10e7
            enc = np.zeros(d * 2)
            enc[0:d:2] = np.sin(x * freq)
            enc[1:d:2] = np.cos(x * freq)
            enc[d::2] = np.sin(y * freq)
            enc[d + 1::2] = np.cos(y * freq)
            return list(enc)

        def norm_alti(alti: int) -> float:
            min_alti = 0
            max_alti = 3164.9099121094
            return [(alti - min_alti) / (max_alti - min_alti)]

        def format_cam(cam: str) -> np.array:
            return [[1, 0] if 'UCE' in cam else [0, 1]][0]

        def cyclical_enc_datetime(date: str, time: str) -> list:
            def norm(num: float) -> float:
                return (num - (-1)) / (1 - (-1))

            year, month, day = date.split('-')
            if year == '2018':
                enc_y = [1, 0, 0, 0]
            elif year == '2019':
                enc_y = [0, 1, 0, 0]
            elif year == '2020':
                enc_y = [0, 0, 1, 0]
            elif year == '2021':
                enc_y = [0, 0, 0, 1]
            sin_month = np.sin(2 * np.pi * (int(month) - 1 / 12))  ## months of year
            cos_month = np.cos(2 * np.pi * (int(month) - 1 / 12))
            sin_day = np.sin(2 * np.pi * (int(day) / 31))  ## max days
            cos_day = np.cos(2 * np.pi * (int(day) / 31))
            h, m = time.split('h')
            sec_day = int(h) * 3600 + int(m) * 60
            sin_time = np.sin(2 * np.pi * (sec_day / 86400))  ## total sec in day
            cos_time = np.cos(2 * np.pi * (sec_day / 86400))
            return enc_y + [norm(sin_month), norm(cos_month), norm(sin_day), norm(cos_day), norm(sin_time),
                            norm(cos_time)]

        data = {'IMG': [], 'MSK': [], 'MTD': []}
        for domain in path_folders:
            data['IMG'] += sorted(list(get_data_paths(domain, 'IMG*.tif')), key=lambda x: int(x.split('_')[-1][:-4]))
            if test_set == False:
                data['MSK'] += sorted(list(get_data_paths(domain, 'MSK*.tif')),
                                      key=lambda x: int(x.split('_')[-1][:-4]))

        if use_metadata == True:

            with open(path_metadata, 'r') as f:
                metadata_dict = json.load(f)
            for img in data['IMG']:
                curr_img = img.split('/')[-1][:-4]
                enc_coords = coordenc_opt(
                    [metadata_dict[curr_img]["patch_centroid_x"], metadata_dict[curr_img]["patch_centroid_y"]])
                enc_alti = norm_alti(metadata_dict[curr_img]["patch_centroid_z"])
                enc_camera = format_cam(metadata_dict[curr_img]['camera'])
                enc_temporal = cyclical_enc_datetime(metadata_dict[curr_img]['date'], metadata_dict[curr_img]['time'])
                mtd_enc = enc_coords + enc_alti + enc_camera + enc_temporal
                data['MTD'].append(mtd_enc)

        if not test_set:
            if len(data['IMG']) != len(data['MSK']):
                print(
                    '[WARNING !!] UNMATCHING NUMBER OF IMAGES AND MASKS ! Please check load_data function for debugging.')
            if data['IMG'][0][-10:-4] != data['MSK'][0][-10:-4] or data['IMG'][-1][-10:-4] != data['MSK'][-1][-10:-4]:
                print('[WARNING !!] UNSORTED IMAGES AND MASKS FOUND ! Please check load_data function for debugging.')

        return data

    path_trainval = Path(path_data, "train")
    trainval_domains = [Path(path_trainval, domain) for domain in os.listdir(path_trainval)]
    random.shuffle(trainval_domains)
    idx_split = int(len(trainval_domains) * val_percent)
    train_domains, val_domains = trainval_domains[:idx_split], trainval_domains[idx_split:]

    dict_train = _gather_data(train_domains, path_metadata, use_metadata=use_metadata, test_set=False)
    dict_val = _gather_data(val_domains, path_metadata, use_metadata=use_metadata, test_set=False)

    path_test = Path(path_data, "test")
    test_domains = [Path(path_test, domain) for domain in os.listdir(path_test)]

    dict_test = _gather_data(test_domains, path_metadata, use_metadata=use_metadata, test_set=True)

    return dict_train, dict_val, dict_test


class Fit_Dataset(Dataset):

    def __init__(self,
                 dict_files,
                 num_classes=13,
                 use_metadata=True,
                 use_augmentations=None,
                 use_compose=None
                 ):

        self.list_imgs = np.array(dict_files["IMG"])
        self.list_msks = np.array(dict_files["MSK"])
        self.use_metadata = use_metadata
        if use_metadata:
            self.list_metadata = np.array(dict_files["MTD"])
        self.use_augmentations = use_augmentations
        self.num_classes = num_classes
        self.use_compose = use_compose
        self.metainfo = dict(classes=classes, palette=palette)

    @staticmethod
    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return array

    def read_msk(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_msk:
            array = src_msk.read()[0]
            array[array > self.num_classes] = self.num_classes
            array = array - 1
            array = np.stack([array == i for i in range(self.num_classes)], axis=0)
            return array

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        image_file = self.list_imgs[index]
        img = self.read_img(self, raster_file=image_file)

        mask_file = self.list_msks[index]
        msk = self.read_msk(raster_file=mask_file)

        if self.use_augmentations is not None:
            sample = {"image": img.swapaxes(0, 2).swapaxes(0, 1), "mask": msk.swapaxes(0, 2).swapaxes(0, 1)}
            transformed_sample = self.use_augmentations(**sample)
            img, msk = transformed_sample["image"].swapaxes(0, 2).swapaxes(1, 2).copy(), \
                transformed_sample["mask"].swapaxes(0, 2).swapaxes(1, 2).copy()

        img = img_as_float(img)

        if self.use_metadata:
            mtd = self.list_metadata[index]
            return (torch.as_tensor(img, dtype=torch.float), torch.as_tensor(mtd, dtype=torch.float)), \
                torch.as_tensor(msk, dtype=torch.float)

        else:
            return torch.as_tensor(img, dtype=torch.float), torch.as_tensor(msk, dtype=torch.float)


class Predict_Dataset(Dataset):

    def __init__(self,
                 dict_files,
                 num_classes=13, use_metadata=True
                 ):
        self.list_imgs = np.array(dict_files["IMG"])
        self.num_classes = num_classes
        self.use_metadata = use_metadata
        if use_metadata:
            self.list_metadata = np.array(dict_files["MTD"])
        self.metainfo = dict(classes=classes, palette=palette)

    @staticmethod
    def read_img(self, raster_file: str) -> np.ndarray:
        with rasterio.open(raster_file) as src_img:
            array = src_img.read()
            return array

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        image_file = self.list_imgs[index]
        img = self.read_img(self, raster_file=image_file)
        img = img_as_float(img)

        if self.use_metadata:
            mtd = self.list_metadata[index]
            return (torch.as_tensor(img, dtype=torch.float), torch.as_tensor(mtd, dtype=torch.float)), \
                '/'.join(image_file.split('/')[-4:])
        else:
            return torch.as_tensor(img, dtype=torch.float), '/'.join(image_file.split('/')[-4:])


def step_loading(path_data, path_metadata_file: str, use_metadata: bool) -> Tuple[dict, dict, dict]:
    print('+' + '-' * 29 + '+', '   LOADING DATA   ', '+' + '-' * 29 + '+')
    train, val, test = load_data(path_data, path_metadata_file, use_metadata=use_metadata)
    return train, val, test


train_pipeline = A.Compose([
    A.RandomBrightnessContrast(p=0.2),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.CoarseDropout(p=0.5),
    A.PixelDropout(p=0.5),
    A.ChannelDropout(channel_drop_range=(1, 4), p=0.2),
])


class DataModule:

    def __init__(
            self,
            dict_train=None,
            dict_val=None,
            dict_test=None,
            num_workers=1,
            batch_size=2,
            drop_last=True,
            num_classes=13,
            num_channels=5,
            use_metadata=True,
            use_augmentations=True
    ):
        super().__init__()
        self.dict_train = dict_train
        self.dict_val = dict_val
        self.dict_test = dict_test
        self.batch_size = batch_size
        self.num_classes, self.num_channels = num_classes, num_channels
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.pred_dataset = None
        self.drop_last = drop_last
        self.use_metadata = use_metadata
        self.use_augmentations = use_augmentations

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage == "validate":
            self.train_dataset = Fit_Dataset(
                dict_files=self.dict_train,
                num_classes=self.num_classes,
                use_metadata=self.use_metadata,
                use_augmentations=self.use_augmentations
            )

            self.val_dataset = Fit_Dataset(
                dict_files=self.dict_val,
                num_classes=self.num_classes,
                use_metadata=self.use_metadata
            )

        elif stage == "predict":
            self.pred_dataset = Predict_Dataset(
                dict_files=self.dict_test,
                num_classes=self.num_classes,
                use_metadata=self.use_metadata
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.pred_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )
