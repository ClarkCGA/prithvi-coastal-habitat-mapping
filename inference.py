import os
import argparse
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
from affine import Affine
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from src.custom_dataset import AquacultureData
from src.models import models
from src.utils import *
import glob
import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
import tqdm
import pandas as pd



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the inference YAML config file (e.g. configs/inference_config.yaml)')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    data_dir = config["data"]["data_dir"]
    dataset_name = config["data"]["dataset_name"]
    apply_normalization = config["data"]["apply_normalization"]
    global_stats = config["data"]["global_stats"]
    inference_batch_size = config["inference"]["inference_batch_size"]
    shuffle = config["inference"]["shuffle"]
    
    checkpoint = config["inference"]["prithvi_finetune_weight"]
    fpn1_type = config["model"]["fpn1_type"]
    arch = config["model"]["arch"]
    
    device=config["device_name"]
    n_channel=config["model"]["n_channel"]
    n_class=config["model"]["n_class"]
    n_frame=config["data"]["n_frame"]
    embed_size=config["model"]["encoder_embed_dim"]
    depth = config["model"]["depth"]
    
    output_dir=config["inference"]["pred_outdir"]
    
    input_size=config["data"]["input_size"]
    patch_size=config["data"]["patch_size"]
    
    
    # Print all the configuration parameters
    print(f"Dataset Name: {dataset_name}")
    print(f"Number of Input Channel: {n_channel}")
    print(f"Input size: {input_size}")
    print(f"Number of Segmentation Class: {n_class}")
    print(f"Checkpoint Path: {checkpoint}")
    print(f"Prediction output dir:{output_dir}")
    
    score_path = Path(output_dir) / "hardened_prob"
    score_path.mkdir(parents=True, exist_ok=True)
    prob_path = Path(output_dir) / "prob"
    prob_path.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(output_dir, 'inference_config.yaml'), 'w') as file:
        yaml.safe_dump(config, file)
    
    
    #initialize dataset class
    aquaculture_dataset_inference = AquacultureData(data_dir, usage="inference", dataset_name=dataset_name, 
                                                csv_path=None, apply_normalization=apply_normalization, 
                                                global_stats=global_stats, trans=None)
    
    #initialize dataloader
    inference_dataloader=DataLoader(aquaculture_dataset_inference, batch_size=inference_batch_size,
                                    shuffle=shuffle, collate_fn=meta_handling_collate_fn, 
                                    num_workers=0)

    print(f"Using device: {device}")

    #initialize model    
    model_wrapper = models[arch]
    #wrapper of prithvi 
    model=model_wrapper(n_channel, n_class, n_frame, embed_size, depth, input_size,
                          patch_size, prithvi_weight=None, fpn1_type=fpn1_type) 
  
    chkpt = torch.load(checkpoint, map_location=device)
    model_weights = chkpt['model_state_dict']
    model.load_state_dict(model_weights)
    model=model.to(device)

    model.eval()
    
    with torch.no_grad():
        for j, (input, target, img_ids, img_metas) in enumerate(inference_dataloader):
            
            input=input.to(device)
            target=target.to(device)
            
            out=model(input)
            
            soft_preds = F.softmax(out, 1)
            hard_preds = soft_preds.max(dim=1)
            
            batch, num_class, height, width = soft_preds.size()
            
            for i in range(batch):
                img_id = img_ids[i]
                name_prob = f"prob_id_{img_id}"
                name_crisp = f"crisp_id_{img_id}.tif"
                
                img_meta = img_metas[i]
                
                # Convert the string back to a CRS object
                img_meta["crs"] = rasterio.crs.CRS.from_string(img_meta["crs"])
                if "transform" in img_meta:
                    #img_meta["transform"] = [t.item() for t in img_meta["transform"]]
                    img_meta["transform"] = Affine(*img_meta["transform"][:6])
                
                meta_hard = img_meta.copy()
                meta_hard.update({
                    "dtype": "int16",
                    "count": 1,
                    })
                meta_soft = img_meta.copy()
                meta_soft.update({
                    "dtype": "float32",
                    "count": 1,
                    })
                
                hard_pred = hard_preds[1].cpu().numpy()[i, :, :].astype(meta_hard["dtype"])
                with rasterio.open(Path(score_path) / name_crisp, "w", **meta_hard) as dst:
                    dst.write(hard_pred, 1)
                
                for n in range(1, n_class):
                    name_prob_updated = f"{name_prob}_cat_{n}.tif"
                    soft_pred = soft_preds[:, n, :, :].data[i].cpu().numpy( ) * 100
                    #soft_pred = np.expand_dims(soft_pred, axis=0).astype(meta_soft["dtype"])
                    
                    with rasterio.open(Path(prob_path) / name_prob_updated, "w", **meta_soft) as dst:
                        dst.write(soft_pred, 1)
                
                print(f"Pred tile: {img_id} is written to {output_dir}")
    
if __name__ == "__main__":
    main()


