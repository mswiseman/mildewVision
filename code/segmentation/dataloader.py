import os
import cv2
import h5py
import random
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms.functional as TF

from pathlib import Path

np.random.seed(2021)
torch.manual_seed(2021)

import cv2
import numpy as np
import torch
from pathlib import Path
import torchvision.transforms.functional as TF


class HyphalDirDataset(torch.utils.data.Dataset):
    """
    Expects:
      image_dir: directory of images (jpg/png/etc)
      mask_dir:  directory of masks (png recommended)
    Pairs by filename stem:
      images/foo.jpg  <-> masks/foo.png  (same stem "foo")
    """

    def __init__(
            self,
            image_dir,
            mask_dir,
            train=True,
            normalize=True,
            thicken=True,
            kernel=(5, 5),
            image_exts=(".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"),
            mask_exts=(".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"),
            require_pair=True,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.train = train
        self.thicken = thicken
        self.kernel = np.ones(kernel, np.uint8)
        self.normalize_ = normalize

        # Index masks by stem for fast pairing
        mask_map = {}
        for ext in mask_exts:
            for p in self.mask_dir.glob(f"*{ext}"):
                mask_map[p.stem] = p

        # Build list of (image_path, mask_path)
        pairs = []
        for ext in image_exts:
            for img_path in self.image_dir.glob(f"*{ext}"):
                m = mask_map.get(img_path.stem)
                if m is None:
                    if require_pair:
                        continue
                    else:
                        m = None
                pairs.append((img_path, m))

        if require_pair and len(pairs) == 0:
            raise RuntimeError(
                f"No (image,mask) pairs found. "
                f"Check dirs:\n  image_dir={self.image_dir}\n  mask_dir={self.mask_dir}\n"
                f"and that filenames share the same stem."
            )

        self.pairs = sorted(pairs, key=lambda x: x[0].name)

    def transform(self, image, mask):
        # image: HxWx3 uint8, mask: HxW uint8
        if self.train:
            image_ = TF.to_pil_image(image)
            mask_ = TF.to_pil_image(mask)

            # If you want augmentations later, apply to BOTH here

            image = TF.to_tensor(image_)
            mask = TF.to_tensor(mask_)  # -> (1,H,W) float in [0,1]
        else:
            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)

        return image, mask

    def normalize(self, image):
        means = [118. / 255., 165. / 255., 92. / 255.]
        stds = [40. / 255., 35. / 255., 51. / 255.]
        return TF.normalize(image, means, stds)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # ---- Load image (RGB)
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ---- Load mask (single channel)
        if mask_path is None:
            raise RuntimeError(f"No mask for image: {img_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        # Make mask binary if needed (common if masks are 0/255 already)
        # If your masks are already 0/1, you can remove this.
        mask = (mask > 0).astype(np.uint8)

        if self.thicken:
            mask = cv2.dilate(mask, self.kernel, iterations=1)

        # Match your old behavior: expand + scale to 0/255 before TF.to_tensor
        mask = mask[..., np.newaxis] * 255  # HxWx1 uint8

        image, mask = self.transform(image, mask)

        if self.normalize_:
            image = self.normalize(image)

        return image, mask


class HyphalDataset(torch.utils.data.Dataset):

    label_class_map = {0: 'Non-infected', 1: 'Hyphae', 2: 'Conidiophore'}

    def __init__(self, dataset_path, train=True, normalize=True, thicken=True, kernel=(5, 5)):
        self.root_dir = Path(dataset_path['root_path'])
        self.train_filepath = self.root_dir / dataset_path['train_filepath']
        self.val_filepath = self.root_dir / dataset_path['val_filepath']
        self.train = train
        self.thicken = thicken
        self.kernel = np.ones(kernel, np.uint8)
        self.normalize_ = normalize
        if train is True:
            self.data_filepath = self.train_filepath
        else:
            self.data_filepath = self.val_filepath

        with h5py.File(self.data_filepath, 'r') as f:
            image_ds = f['images']
            self.images = image_ds[:, ]
            label_ds = f['masks']
            self.masks = label_ds[:]

    def transform(self, image, mask):
        if self.train:
            image_ = TF.to_pil_image(image)
            mask_ = TF.to_pil_image(mask)

            # if random.random() > 0.5:
            #     image_ = TF.hflip(image_)
            #     mask_ = TF.hflip(mask_)

            # if random.random() > 0.5:
            #     image_ = TF.vflip(image_)
            #     mask_ = TF.vflip(mask_)

            image = TF.to_tensor(image_)
            mask = TF.to_tensor(mask_)
        else:
            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)

        return image, mask

    def normalize(self, image):

        means = [118./255., 165./255., 92./255.]
        stds = [40./255., 35./255., 51./255.]

        image = TF.normalize(image, means, stds)

        return image

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        cur_images = self.images[idx, ...]
        cur_masks = self.masks[idx, ...]

        if self.thicken:
            # Preprocess mask (thicken hyphal lines)
            cur_masks = cv2.dilate(cur_masks, self.kernel, iterations=1)
        cur_masks = cur_masks[..., np.newaxis] * 255

        cur_images, cur_masks = self.transform(cur_images, cur_masks)
        if self.normalize_:
            cur_images = self.normalize(cur_images)

        return cur_images, cur_masks


def worker_init_fn(worker_id):
    print(torch.utils.data.get_worker_info())


# use to test the dataset class


def test_class():
    label_class_map = {0: 'Non-infected', 1: 'Infected'}

    # Parameters for dataset
    dataset_path = {
        'root_path': '/media/cornell/Data/tq42/Hyphal_2020/data/segmentation/cls_seg_mask',
        'meta_filepath': 'metadata.csv',
        'train_filepath': 'train_set_DeepLift.hdf5',
        'val_filepath': 'val_set.hdf5'
    }

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        # torchvision.transforms.Resize(299),
        # torchvision.transforms.RandomHorizontalFlip(p=0.5),
        # torchvision.transforms.RandomAffine(degrees=(0, 180), translate=(0.1, 0.1), scale=(0.8, 1.2)),
        # torchvision.transforms.RandomRotation(degrees=(0, 180)),
        # torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor()
    ])

    hyphal_train_ds = HyphalDataset(
        dataset_path, train=True, normalize=False, thicken=False, kernel=(5, 5))

    hyphal_dl = torch.utils.data.DataLoader(hyphal_train_ds,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=0,
                                            worker_init_fn=worker_init_fn)

    for image, mask in hyphal_dl:
        image = np.transpose(image[0], (1, 2, 0))
        mask = mask.squeeze(1)[0].numpy()

        # kernel = np.ones((3, 3), dtype=np.uint8)
        # processed_mask = cv2.erode(mask.astype(np.uint8), kernel)
        # processed_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        infected_pixels = len(mask[mask != 0])
        if infected_pixels > 0:
            f, axs = plt.subplots(1, 2)
            # import pdb; pdb.set_trace()
            axs[0].imshow(image)
            axs[1].imshow(mask, vmin=0, vmax=1, cmap='Reds')
            # axs[2].imshow(processed_mask, vmin=0, vmax=1, cmap='Reds')
            break

    plt.tight_layout()
    # plt.show()
    plt.savefig('test.png')


if __name__ == "__main__":
    test_class()
