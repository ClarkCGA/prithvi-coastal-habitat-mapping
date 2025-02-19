import os, math, random
from pathlib import Path
import tqdm
import numpy as np
import pandas as pd
import torch
from affine import Affine
from torch.utils.data import Dataset
from src.utils import load_data, flip



class AquacultureData(Dataset):
    r"""
    Create an iterable dataset of image chips
    Arguments:
        src_dir (str): Path to the folder contains data folders and files.
        usage (str): can be either train, validate, or test.
        dataset_name (str): Name of the training/validation dataset containing 
                            structured folders for image, label, and mask.
        csv_path (str): full path to the csv file containing the id of tiles used 
                        for splitting the dataset into train and validation subsets.
        apply_normalization (binary): decides if normalization should be applied.
        global_stats (dict): Optional dictionary containing the 'mean', and 'std' arrays 
                             for each band.
        trans (list of str): List of different flip transformation:['v_flip','h_flip','d_flip']
    Returns:
        A tuple of (image, label) for training and validation but only the image iterable 
        if in the inference phase.
    """

    # version I am modifying
    def __init__(self, src_dir, usage, dataset_name, csv_path=None, apply_normalization=False, 
                 global_stats=None, trans=None, **kwargs):

        self.usage = usage
        self.dataset_name = dataset_name
        self.apply_normalization = apply_normalization
        self.trans = trans
        self.kwargs = kwargs

        assert self.usage in ["train", "validation", "inference"], "Usage is not recognized."

        if self.usage in ["train", "validation"]:
            assert csv_path is not None, "For training/validation you must provide a csv for data splitting."
            catalog = pd.read_csv(csv_path, header=None)
            flag_ids = catalog[0].tolist()
            #bad_tiles = ['305_343', '417_328', '419_322', '419_323', '417_321']
            #flag_ids = [item for item in flag_ids_unrefined if item not in bad_tiles]

            img_fnames = [Path(dirpath) / f
                        for (dirpath, dirnames, filenames) in os.walk(Path(src_dir) / self.dataset_name) 
                        for f in filenames 
                        if f.endswith(".tif") and ("band_chip" in f) and ('_'.join(Path(f).stem.split('_')[:-2]) in flag_ids)]
            img_fnames.sort()

            lbl_fnames = [Path(dirpath) / f 
                        for (dirpath, dirnames, filenames) in os.walk(Path(src_dir) / self.dataset_name) 
                        for f in filenames 
                        if f.endswith(".tif") and ("label_chip" in f) and ('_'.join(Path(f).stem.split('_')[:-2]) in flag_ids)]
            lbl_fnames.sort()

            self.img_chips = []
            self.lbl_chips = []

            for img_fname, lbl_fname in tqdm.tqdm(zip(img_fnames, lbl_fnames), 
                                                  total=len(img_fnames)):
                    
                img_chip = load_data(Path(src_dir) / self.dataset_name / img_fname,
                                     usage=self.usage,
                                     is_label=False,
                                     apply_normalization=self.apply_normalization,
                                     global_stats=global_stats)
                img_chip = img_chip.transpose((1, 2, 0))

                lbl_chip = load_data(Path(src_dir) / self.dataset_name / lbl_fname, 
                                     usage=self.usage,
                                     is_label=True)

                self.img_chips.append(img_chip)
                self.lbl_chips.append(lbl_chip)
        else:
            
            img_fnames = [Path(dirpath) / f
                        for (dirpath, dirnames, filenames) in os.walk(Path(src_dir) / self.dataset_name) 
                        for f in filenames 
                        if f.endswith(".tif") and ("band_chip" in f)]
            img_fnames.sort()

            lbl_fnames = [Path(dirpath) / f 
                        for (dirpath, dirnames, filenames) in os.walk(Path(src_dir) / self.dataset_name) 
                        for f in filenames 
                        if f.endswith(".tif") and ("label_chip" in f)]
            
            lbl_fnames.sort()
            self.img_chips = []
            self.lbl_chips = []
            self.ids = []
            self.meta_ls = []

            for img_fname, lbl_fname in tqdm.tqdm(zip(img_fnames, lbl_fnames),
                                                  total=len(img_fnames)):
                img_chip, meta = load_data(Path(src_dir) / self.dataset_name / img_fname,
                                           usage=self.usage,
                                           is_label=False,
                                           apply_normalization=self.apply_normalization,
                                           global_stats=global_stats)
                img_chip = img_chip.transpose((1, 2, 0))
                lbl_chip = load_data(Path(src_dir) / self.dataset_name / lbl_fname, 
                                     usage=self.usage, is_label=True)
                
                self.img_chips.append(img_chip)
                self.lbl_chips.append(lbl_chip)

                self.meta_ls.append(meta)

                img_id = '_'.join(img_fname.stem.split('_')[1:3])
                self.ids.append(img_id)

        
        print(f"------ {self.usage} dataset with {len(self.img_chips)} patches created ------")


    def __getitem__(self, index):
        """
        Support indexing such that dataset[index] can be used to get 
        the (index)th sample.
        """
        kwargs = self.kwargs
        
        if self.usage in ["train", "validation"]:
            img_chip = self.img_chips[index]
            lbl_chip = self.lbl_chips[index]

            if self.trans and self.usage == "train":
                trans_flip_ls = [m for m in self.trans if "flip" in m]
                if random.randint(0, 1) and len(trans_flip_ls) >= 1:
                    trans_flip = random.sample(trans_flip_ls, 1)
                    img_chip, lbl_chip = flip(img_chip, lbl_chip, trans_flip[0])           
                
            label = torch.from_numpy(np.ascontiguousarray(lbl_chip)).long()
            # shape from (H,W,C) --> (C,H,W)
            img_chip = torch.from_numpy(img_chip.transpose((2, 0, 1))).float()

            return img_chip, label
        
        else:
            img_chip = self.img_chips[index]
            lbl_chip = self.lbl_chips[index]
            img_id = self.ids[index]
            #img_meta = self.meta_ls[index]
            img_meta = self.get_metadata_dict(self.meta_ls[index])
            
            img_chip = torch.from_numpy(img_chip.transpose((2, 0, 1))).float()
            label = torch.from_numpy(np.ascontiguousarray(lbl_chip)).long()

            return img_chip, label, img_id, img_meta

    def get_metadata_dict(self, meta):
        img_meta = {}
        for k, v in meta.items():
            if isinstance(v, torch.Tensor):
                img_meta[k] = v.item()  # get Python number from tensor
            elif isinstance(v, list):
                img_meta[k] = [x.item() if isinstance(x, torch.Tensor) else x for x in v]  # get Python number from tensor in list
            else:
                img_meta[k] = v

        # Handle crs and transform if present in the metadata
        if "crs" in img_meta and not isinstance(img_meta["crs"], str):
            img_meta["crs"] = img_meta["crs"].to_string()

        if "transform" in img_meta and isinstance(img_meta["transform"], Affine):
            img_meta["transform"] = [x for x in img_meta["transform"]]

        return img_meta

    def __len__(self):
        return len(self.img_chips)