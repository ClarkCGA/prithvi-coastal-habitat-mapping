import os
import numpy as np
import rasterio
import torch
import torch.nn as nn
from PIL import Image


warmup_iters = 1500
warmup_ratio = 1e-6
power = 1.0
total_steps=10000

def lr_lambda(current_step):
    if current_step < warmup_iters:
        # Linear warmup phase
        return warmup_ratio + (1 - warmup_ratio) * (current_step / warmup_iters)
    else:
        # Polynomial decay
        return (1 - (current_step - warmup_iters) / (total_steps - warmup_iters)) ** power


def segmentation_loss(target, pred, device, class_weights, ignore_index):
    
    target = target.long()
            
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights).to(device) 
    loss = criterion(pred, target) 

    return loss


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, filename):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss":val_loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at {filename}")

def plot_output_image(target, output, epoch, output_dir):

    target = target.detach().cpu().numpy()
    output = output.detach().cpu().numpy()
    img = Image.fromarray(output, mode='F') #PIL Image
    target_img = Image.fromarray(target, mode='F') #PIL Image

    # Save the image
    output_image_path = os.path.join(output_dir,f"segmentation_output_epoch_{epoch}.tif")
    img.save(output_image_path)
    if epoch == 0:
        target_image_path = os.path.join(output_dir,f"segmentation_target_epoch_{epoch}.tif")
        target_img.save(target_image_path)

# def plot_output_image(model, device, epoch, means, stds, input_path, prediction_img_dir):
    
#     model.eval()  

#     if_img=1
#     img=load_raster(input_path, if_img, crop=(224, 224))

#     final_image=preprocess_image(img, means, stds)
#     final_image=final_image.to(device)

    
#     with torch.no_grad():
#         output = model(final_image)  # [1, n_segmentation_class, 224, 224]

#     # Remove batch dimension
#     output = output.squeeze(0)  # [n_segmentation_class, 224, 224]

#     predicted_mask = torch.argmax(output, dim=0)  # shape [224, 224]
#     predicted_mask = predicted_mask.cpu().numpy()
#     binary_image = (predicted_mask * 255).astype(np.uint8)
#     img = Image.fromarray(binary_image, mode='L') #PIL Image

#     # Save the image
#     output_image_path = os.path.join(prediction_img_dir,f"segmentation_output_epoch_{epoch}.png")
#     img.save(output_image_path)


def compute_accuracy(labels, output):

    
    # (batch_size, 2_class,time_frame,224, 224) ->  (batch_size, 224, 224)
    predicted = torch.argmax(output, dim=1)

    correct = (predicted == labels).sum().item() 
    total = labels.numel()  # Total number of elements in labels
    accuracy = correct / total 

    return accuracy


def calculate_miou(output, target, device):
    
    eps=1e-6
    #output.shape = B,n_classes,H,W
    num_classes=output.shape[1]
    preds = torch.argmax(output, dim=1)

    # Flatten the tensors
    preds = preds.view(-1) 
    target = target.view(-1)  

    # Initialize intersection and union for each class
    intersection = torch.zeros(num_classes).to(device)
    union = torch.zeros(num_classes).to(device)

    for cls in range(num_classes): #starts from 0

        # Create binary masks for the current class
        pred_mask = (preds == cls).float()
        target_mask = (target == cls).float()

        # Calculate intersection and union
        intersection[cls] = (pred_mask * target_mask).sum()
        union[cls] = pred_mask.sum() + target_mask.sum() - intersection[cls]

    # Calculate IoU for each class for all images in one batch
    iou = intersection / (union + eps)  # Add eps to avoid division by zero

    # Calculate mean IoU (all class avergae)
    mean_iou = iou.mean().item()

    return mean_iou

def meta_handling_collate_fn(batch):
    images = []
    labels = []
    img_ids = []
    img_metas = []

    # Unpack elements from each sample in the batch
    for sample in batch:
        images.append(sample[0])
        labels.append(sample[1])
        img_ids.append(sample[2])
        img_metas.append(sample[3])  # append the dict to the list

    # Stack images and labels into a single tensor
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    
    return images, labels, img_ids, img_metas


def load_data(data_path, usage, is_label=False, apply_normalization=False, 
              global_stats=None, dtype=np.float32, verbose=False):
    r"""
    Open data using gdal, read it as an array and normalize it.

    Arguments:
            data_path (string): Full path including filename of the data source we wish to load.
            usage (string): Either "train", "validation", "inference".
            is_label (binary): If True then the layer is a ground truth (category index) and if
                                set to False the layer is a reflectance band.
            apply_normalization (binary): If true min/max normalization will be applied on each band.
            global_stats (dict): Optional dictionary containing the 'min', 'max', 'mean', and 'std' arrays 
                                 for each band. If not provided, these values will be calculated from the data.
            dtype (np.dtype): Data type of the output image chips.
            verbose (binary): if set to true, print a screen statement on the loaded band.

    Returns:
            image: Returns the loaded image as a 32-bit float numpy ndarray.
    """

    # Inform user of the file names being loaded from the Dataset.
    if verbose:
        print('loading file:{}'.format(data_path))

    # open dataset using rasterio library.
    with rasterio.open(data_path, "r") as src:

        if is_label:
            if src.count != 1:
                raise ValueError("Expected Label to have exactly one channel.")
            img = src.read(1)
            return img

        else:
            meta = src.meta
            if apply_normalization:
                img = do_normalization(src.read(), global_stats=global_stats)
                img = img.astype(dtype)
            else:
                img = src.read()
                img = img.astype(dtype)

    if usage in ["train", "validation"]:
        return img
    else:
        return img, meta


def do_normalization(img, global_stats):
    """
    Standardize the input image pixels.

    Args:
        img (np.ndarray): Stacked image bands with a dimension of (C, H, W).
        global_stats (dict): Optional dictionary containing the 'min', 'max', 'mean', and 'std' arrays
                      for each band. If not provided, these values will be calculated from the data.

    Returns:
        np.ndarray: Normalized image stack of size (C, H, W).
    """

    if global_stats is None:
        raise ValueError("Global statistics must be provided for global normalization.")
    else:
        means = np.array(global_stats['mean'])
        stds = np.array(global_stats['std'])
    
    normal_img = (img - means[:, None, None]) / stds[:, None, None]

    return normal_img


def flip(img, label, flip_type):
    r"""
    Synthesize a new pair of image, label chips by flipping the input chips around a user defined axis.

    Arguments:
            img (ndarray) -- Concatenated variables or brightness value with a dimension of (H,W,C)
            label (ndarray) -- Ground truth with a dimension of (H,W)
            flip_type (list) -- A flip type based on the choice of axis.
                                Provided transformation are:
                                    1) 'v_flip', vertical flip
                                    2) 'h_flip', horizontal flip
                                    3) 'd_flip', diagonal flip
    Returns:
            img -- A numpy array of flipped variables or brightness value.
            label --A numpy array of flipped labeled reference (ground truth).
    """

    def diagonal_flip(image):
        flipped = np.flip(image, 1)
        flipped = np.flip(flipped, 0)
        return flipped

    if isinstance(flip_type, str):
        # Horizontal flip
        if flip_type == "h_flip":
            img = np.flip(img, 0)
            label = np.flip(label, 0)

        # Vertical flip
        elif flip_type == "v_flip":
            img = np.flip(img, 1)
            label = np.flip(label, 1)

        # Diagonal flip
        elif flip_type == "d_flip":
            img = diagonal_flip(img)
            label = diagonal_flip(label)

        else:
            raise ValueError("Flip type must be one of: 'h_flip', 'v_flip' or 'd_flip'.")
    else:
        raise ValueError("Flip type param must be a tuple or list.")

    return img.copy(), label.copy()