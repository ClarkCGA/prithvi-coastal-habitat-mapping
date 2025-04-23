import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from src.custom_dataset import AquacultureData
from src.models import models
from src.utils import *
import numpy as np
import yaml
import torch.nn as nn
import glob
import os
#import wandb
import argparse
from PIL import Image
import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
import tqdm
import pandas as pd


def main():

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    device = config["device_name"]
    n_channel = config["model"]["n_channel"]
    n_class = config["model"]["n_class"]
    n_frame = config["data"]["n_frame"]
    n_time_steps = config["data"]["n_time_steps"]
    n_iteration = config["n_iteration"]
    depth = config["model"]["depth"]
    embed_size = config["model"]["encoder_embed_dim"]
    dec_embed_size = config["model"]["dec_embed_dim"]
    data_dir = config["data"]["data_dir"]
    dataset_name = config["data"]["dataset_name"]
    train_csv_path = config["data"]["train_csv_path"]
    val_csv_path = config["data"]["val_csv_path"]             
    train_batch_size = config["training"]["train_batch_size"]
    val_batch_size = config["validation"]["val_batch_size"]
    apply_normalization = config["data"]["apply_normalization"]
    global_stats = config["data"]["global_stats"]
    transformations = config["data"]["transformations"]
    learning_rate = config["training"]["learning_rate"]
    class_weights = config["class_weights"]
    ignore_index = config["ignore_index"]
    freeze_backbone = config["training"]["freeze_backbone"]

    output_dir = config["output_dir"]

    input_size = config["data"]["input_size"]
    patch_size = config["data"]["patch_size"]
    arch = config["model"]["arch"]


    # Print all the configuration parameters
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {train_batch_size}")
    print(f"Number of Epochs: {n_iteration}")
    print(f"Number of Input Channel: {n_channel}")
    print(f"Number of Segmentation Class: {n_class}")
    print(f"Used device name: {device}")
    print(f"Checkpoint Dir: {output_dir}")
    print(f"Data input dir:{data_dir}")
    
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as file:
        yaml.safe_dump(config, file)
    
    # wandb.init(
    #          # set the wandb project where this run will be logged
    #          project=config["wandb_project"]
    #          )
    
    # wandb.run.log_code(".")
    
    
    #initialize dataset class
    aquaculture_dataset_train = AquacultureData(data_dir, usage="train", dataset_name=dataset_name, 
                                                csv_path=train_csv_path, apply_normalization=apply_normalization, 
                                                global_stats=global_stats, trans=transformations)
    aquaculture_dataset_val = AquacultureData(data_dir, usage="validation", dataset_name=dataset_name, 
                                                csv_path=val_csv_path, apply_normalization=apply_normalization, 
                                                global_stats=global_stats, trans=transformations)
    
    #initialize dataloader
    train_dataloader=DataLoader(aquaculture_dataset_train, batch_size=train_batch_size,
                                shuffle=config["training"]["shuffle"], num_workers=1)
    val_dataloader=DataLoader(aquaculture_dataset_val, batch_size=val_batch_size,
                              shuffle=config["validation"]["shuffle"], num_workers=1)

    #initialize model    
    model_weights = config["prithvi_model_new_weight"] 
    
    model_wrapper = models[arch]
    #wrapper of prithvi #initialization of prithvi is done by initializing prithvi_loader.py
    model=model_wrapper(n_channel, n_class, n_frame, embed_size, depth, input_size, patch_size, model_weights, freeze_backbone)
    #model=model_wrapper(n_channel, n_class, n_frame, embed_size, depth, input_size, patch_size, prithvi_weight=None, freeze_backbone=freeze_backbone) 
    model=model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.05)
    optimizer_config = {'grad_clip': None}
    #scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    #best_loss=torch.tensor(float('inf'))
    training_log = []
    best_loss = 0
    best_miou_val = 0
    

    for i in range(n_iteration):

        loss_i = 0.0
        miou_train = []
        acc_dataset_train = []

        print("iteration started")

        ##################################### train phase   #######################################
        model.train()
        
        inner_pbar = tqdm.tqdm(
            range(len(train_dataloader)), colour="blue", desc="Training Epoch", leave=True)

        for j, (input, target) in enumerate(train_dataloader):
        
            input = input.to(device)
            target = target.to(device)
        
            optimizer.zero_grad()
            out = model(input)

            #print(f"model output shape: {out.shape}")
            #print(f"Input shape: {input.shape}")
            #print(f"target shape: {target.shape}")
            loss=segmentation_loss(target, out, device, class_weights, ignore_index)
            loss_i += loss.item() * input.size(0)
            batch_acc = compute_accuracy(target, out)
            acc_dataset_train.append(batch_acc)
            miou_batch=calculate_miou(out, target, device)
            miou_train.append(miou_batch)

            loss.backward()
            optimizer.step()
            #scheduler.step()
            
            inner_pbar.update(1)
            inner_pbar.set_description(f"Training Batch Loss: {loss.item()}, Training Batch mIoU: {miou_batch}", refresh=True)

        acc_total_train=np.mean(acc_dataset_train)
        miou_train=np.mean(miou_train)
        epoch_loss_train=(loss_i)/len(train_dataloader.dataset)
        
        inner_pbar.close()
        # wandb.log({"epoch": i + 1, "train_loss": epoch_loss_train,"acc_train":acc_total_train,
        #            "learning_rate": optimizer.param_groups[0]['lr'],"miou_train":miou_train})

        ########### Validation Phase ############################################################
        model.eval()

        val_loss = 0.0
        miou_valid=[]
        acc_dataset_val=[]
        inner_pbar = tqdm.tqdm(range(len(val_dataloader)), colour="green", desc="Validation Epoch", leave=True)
        
        with torch.no_grad():
            for j,(input,target) in enumerate(val_dataloader):

                input=input.to(device)
                target=target.to(device)
            
                out=model(input)
                #print(f"model output shape: {out.shape}")
                #print(f"Input shape: {input.shape}")
                #print(f"target shape: {target.shape}")

                # if j==0:
                #     plot_output_image(target[0,0,:,:],out[0,0,:,:],i,output_dir)

                loss=segmentation_loss(target, out, device, class_weights, ignore_index)
                batch_acc=compute_accuracy(target, out)
                acc_dataset_val.append(batch_acc)
            
                val_loss += loss.item() * input.size(0)
                
                miou_batch=calculate_miou(out, target, device)
                miou_valid.append(miou_batch)
                
                inner_pbar.update(1)
                inner_pbar.set_description(f"Validation Batch Loss: {loss.item()}, Validation Batch mIoU: {miou_batch}", refresh=True)

    
        acc_total_val = np.mean(acc_dataset_val)
        epoch_loss_val = val_loss / len(val_dataloader.dataset)
        miou_valid = np.mean(miou_valid)

        inner_pbar.close()
        
        # wandb.log({"epoch": i + 1, "val_loss": epoch_loss_val,"accuracy_val": acc_total_val,
        #            "miou_val": miou_valid})
        
        print(f"Epoch: {i}, train loss: {epoch_loss_train}, val loss: {epoch_loss_val}, " 
              f"accuracy_train: {acc_total_train}, accuracy_val:{acc_total_val}, " 
              f"miou_train:{miou_train},miou_val:{miou_valid}")
        
        training_log.append({'Epoch': i, 'Training Loss': epoch_loss_train, 'Validation Loss': epoch_loss_val,
                             'Train Accuracy': acc_total_train, 'Validation Accuracy': acc_total_val, 'train mIOU': miou_train, 
                             'Validation mIoU': miou_valid})
        training_log_df = pd.DataFrame(training_log)
        training_log_df.to_csv(os.path.join(output_dir, 'training_log.csv'), index=False)

        # best checkpoints saved based on best mIOU for validation/test data
        if i == 0:
            best_loss = epoch_loss_val
            best_miou_val = miou_valid

        if miou_valid > best_miou_val:
            save_checkpoint(model, optimizer, i, epoch_loss_train, epoch_loss_val, 
                            os.path.join(output_dir, f"best_checkpoint_{i}.pt"))
            best_miou_val = miou_valid
        
        # if i % 20 == 0:
        #     plot_output_image(model, device, i, means, stds, segment_input, predicted_mask_dir)

        if i == n_iteration - 1:
            save_checkpoint(model, optimizer, i, epoch_loss_train, epoch_loss_val, 
                            os.path.join(output_dir, 'last_checkpoint.pt'))


    # wandb.finish()
    
if __name__ == "__main__":
    main()