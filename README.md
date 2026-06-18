# Assessing the Robustness of Prithvi Geospatial Foundation Model for Coastal Habitat Mapping under Data Availability and Domain Shift Scenarios

This repository contains the code for the analysis presented in [Assessing the Robustness of Prithvi Geospatial Foundation Model for Coastal Habitat Mapping under Data Availability and Domain Shift Scenarios](https://doi.org/10.1109/JSTARS.2026.3698337). Fine-tuning dataset and fine-tuned model weights are also published on Hugging Face. 

## Requirements

You need to have **Conda** installed before setting up the environment. You also need to have a GPU available on your machine. 


## How to use the code
### Step 1. Clone the repo
```bash
git clone https://github.com/git@github.com:ClarkCGA/prithvi-coastal-habitat-mapping.git
cd prithvi-coastal-habitat-mapping
```

### Step 2. Create the environment
```bash
conda create env -f environment.yaml
```

### Step 3. Activate the environment
```bash
conda activate coastal-habitat 
```

### Step 4. Download labeled dataset and pre-trained models
- Download the pre-trained weights for Prithvi EO 2.0 [300M parameter](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/blob/main/Prithvi_EO_V2_300M.pt) and [600M parameter](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M/blob/main/Prithvi_EO_V2_600M.pt) from Hugging Face.
- 
- Adapt `src/custom_dataset.py` to the specification of your dataset if needed.
- Update `configs/config_300m.yaml` or `configs/config_600m.yaml` with file paths for model weights, data directory, and outputs folder.

### Step 4. Training Models
To fine-tune Prithvi EO 2.0 model variants using the pre-trained weights, run `main_prithvi_aquaculture.py` as following:

```bash
CUDA_VISIBLE_DEVICES=[replace_with_the_GPU_index] torchrun --rdzv_endpoint=0.0.0.0:29500 main_prithvi_aquaculture.py --config configs/[name_of_the_config]
```
### Step 5. Train Prithvi architecture from scratch
To train Prithvi EO 2.0 model variants from scratch, run `main_prithvi_aquaculture_scratch.py` as following:

```bash
CUDA_VISIBLE_DEVICES=[replace_with_the_GPU_index] torchrun --rdzv_endpoint=0.0.0.0:29500 main_prithvi_aquaculture_scratch.py --config configs/[name_of_the_config]
```
You can use the same configs as in step 4 with no changes as `prithvi_weight=None` is hard-coded.

### Note on running the baseline UNet model
- To run the unet you need to use a different repo: [multi-temporal-crop-classification-baseline](https://github.com/ClarkCGA/multi-temporal-crop-classification-baseline)
- Update `configs/config_unet.yaml` and add it to the config folder of the "multi-temporal-crop-classification-baseline" repo and follow the readme instructions of that repo.

### Step 4. How to do inference

- Make sure `configs/inference_config.yaml` is up to date.
- Run from CLI:
```bash
CUDA_VISIBLE_DEVICES=[replace_with_the_GPU_index] torchrun --rdzv_endpoint=0.0.0.0:29500 inference.py --config configs/[name_of_the_config]
```

## Fine-tuned model weights to use or replicate our work

## Aquaculture Dataset

## Citation
If you use this code, please cit the following paper:
S. Khallaghi et al., "Assessing the Robustness of Prithvi Geospatial Foundation Model for Coastal Habitat Mapping under Data Availability and Domain Shift Scenarios," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, doi: [10.1109/JSTARS.2026.3698337](http://doi.org/10.1109/JSTARS.2026.3698337). 