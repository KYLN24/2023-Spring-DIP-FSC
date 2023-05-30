import json
import os
import random

import cv2
import numpy as np
import torch.utils.data as data
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    hi = random.randint(0, res_h)
    wj = random.randint(0, res_w)
    return hi, wj, crop_h, crop_w


class FSCData(data.Dataset):
    def __init__(
        self,
        root,
        crop_size=384,
        method="train",
    ):
        anno_file = os.path.join(root, "annotation_FSC147_384.json")
        data_split_file = os.path.join(root, "Train_Test_Val_FSC_147.json")
        image_class_file = os.path.join(root, "ImageClasses_FSC147.txt")
        self.im_dir = os.path.join(root, "images_384_VarV2")
        self.gt_dir = os.path.join(root, "gt_density_map_adaptive_384_VarV2")

        with open(anno_file) as f:
            self.annotations = json.load(f)
        with open(data_split_file) as f:
            self.data_split = json.load(f)
        with open(image_class_file) as f:
            self.image_class = {}
            for line in f.readlines():
                line = line.strip()
                im_id, cls = line.split("	")
                self.image_class[im_id] = cls

        if method not in ["train", "val", "test"]:
            raise Exception("not implement")

        self.method = method
        self.im_ids = self.data_split[method]

        self.c_size = crop_size

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        self.ex_resize = transforms.Resize((64, 64))

        self.cache = {}

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, item):
        if self.cache.get(item, None) is None:
            im_id = self.im_ids[item]
            img_path = os.path.join(self.im_dir, im_id)
            gd_path = os.path.join(self.gt_dir, im_id).replace(".jpg", ".npy")

            try:
                img = Image.open(img_path).convert("RGB")
                dmap = np.load(gd_path)
                dmap = dmap.astype(np.float32, copy=False)
            except:
                raise Exception("Image open error {}".format(im_id))

            anno = self.annotations[im_id]

            # get examplars
            bboxes = anno["box_examples_coordinates"]
            examplars = []
            for bbox in bboxes:
                x1 = bbox[0][0]
                y1 = bbox[0][1]
                x2 = bbox[2][0]
                y2 = bbox[2][1]
                ex = img.crop((x1, y1, x2, y2))
                examplars.append(self.normalize(self.to_tensor(self.ex_resize(ex))))
                if len(examplars) == 3:
                    break
            examplar_1 = examplars[0]
            examplar_2 = examplars[1]
            examplar_3 = examplars[2]

            cls = f"a photo of {self.image_class[im_id]}"

            if self.method != "train":
                img = self.normalize(self.to_tensor(img))
                count = dmap.sum()
                dmap = Image.fromarray(dmap)
                dmap = self.to_tensor(dmap)

            self.cache[item] = (
                img,
                dmap,
                examplar_1,
                examplar_2,
                examplar_3,
                cls,
            )

        if self.method == "train":
            return self.train_transform(self.cache[item])
        else:
            return self.cache[item]

    def train_transform(self, sample):
        img, dmap, examplar_1, examplar_2, examplar_3, cls = sample
        wd, ht = img.size

        # rescale augmentation
        re_size = random.random() * 0.5 + 0.75
        wdd = int(wd * re_size)
        htt = int(ht * re_size)
        if min(wdd, htt) >= self.c_size:
            raw_size = (wd, ht)
            wd = wdd
            ht = htt
            img = img.resize((wd, ht))
            dmap = cv2.resize(dmap, (wd, ht))
            ratio = (raw_size[0] * raw_size[1]) / (wd * ht)
            dmap = dmap * ratio

        # random crop augmentation
        hi, wi, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = img.crop((wi, hi, wi + w, hi + h))
        dmap = dmap[hi : hi + h, wi : wi + w]

        # random horizontal flip
        if random.random() > 0.5:
            img = F.hflip(img)
            dmap = np.fliplr(dmap)

        dmap = Image.fromarray(dmap)

        return (
            self.normalize(self.to_tensor(img)),
            self.to_tensor(dmap),
            examplar_1,
            examplar_2,
            examplar_3,
            cls,
        )
