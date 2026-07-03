import os
import pickle
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from refinefir.degradation import degrade_tensor, get_degrade_params


class FacePairDataset(Dataset):
    """Training dataset for reference-based face restoration.

    The pair list stores ids such as ``00001_2``. Following the original
    training layout, clean images are resolved as
    ``image_root / id.replace("_", "/") + ".png"`` and warped reference images
    are resolved as ``warp_root / f"{source_id}_{target_id}.png"``.
    """

    def __init__(self, image_root, pair_list, warp_root, resolution=256, nref=1):
        self.image_root = image_root
        self.warp_root = warp_root
        self.resolution = resolution
        self.nref = nref
        self.ids = self._load_pairs(pair_list)
        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.name_dict = self._generate_dict(self.ids) if nref > 1 else None

    @staticmethod
    def _load_pairs(path):
        with open(path, "rb") as fp:
            return np.array(pickle.load(fp))

    @staticmethod
    def _generate_dict(ids):
        result = {}
        for source, target in ids:
            result.setdefault(target, []).append(source)
        return result

    def __len__(self):
        return len(self.ids)

    def _clean_path(self, image_id):
        return os.path.join(self.image_root, image_id.replace("_", "/") + ".png")

    def _warp_path(self, source_id, target_id):
        return os.path.join(self.warp_root, f"{source_id}_{target_id}.png")

    def _load_tensor(self, path):
        return self.transform(Image.open(path).convert("RGB"))

    @torch.no_grad()
    def __getitem__(self, index):
        source_id, target_id = deepcopy(self.ids[index])

        source_path = self._clean_path(source_id)
        target_path = self._clean_path(target_id)
        warp_path = self._warp_path(source_id, target_id)

        if source_id == target_id or not os.path.isfile(warp_path):
            warp_path = source_path

        img_gt = self._load_tensor(target_path)
        img_source = self._load_tensor(source_path)

        if self.nref > 1:
            ref_warped_imgs = [self._load_tensor(warp_path)]
            while len(ref_warped_imgs) < self.nref:
                new_source_id = np.random.choice(self.name_dict[target_id])
                candidate_path = self._warp_path(new_source_id, target_id)
                if new_source_id == target_id or not os.path.isfile(candidate_path):
                    candidate_path = source_path
                ref_warped_imgs.append(self._load_tensor(candidate_path))
            img_warp = torch.cat(ref_warped_imgs, dim=0)
        else:
            img_warp = self._load_tensor(warp_path)

        if np.random.uniform() > 0.5:
            img_gt = torch.flip(img_gt, (-1,))
            img_warp = torch.flip(img_warp, (-1,))

        if np.random.uniform() > 0.5:
            img_source = torch.flip(img_source, (-1,))

        degrade_params = get_degrade_params(self.resolution, self.resolution)
        img_lq = degrade_tensor(img_gt, *degrade_params)

        return img_lq, img_gt, img_source, img_warp, degrade_params

