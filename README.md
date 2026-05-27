# prithvi-usecases: Coastal Habitat Mapping (Semantic Segmentation task)

## How to use the code
### Step 1. Clone the repo
```bash
git clone https://github.com/ClarkCGA/prithvi-usecases.git
cd [cloned_repo_path]
```
#### **Notes:**
- Replace `[cloned_repo_path]` with the actual path where the repository is cloned.
- Make sure you have **Conda** installed before running the environment setup.

### Step 2. Create the environment
```bash
conda create env -f environments.yaml
```

### Step 3. How to do fine-tune for a semantic segmentation task
- Get the pre-trained weights for Prithvi EO v2 from [HuggingFace](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M). Models we used are "Prithvi-EO-2.0-300M" and "Prithvi-EO-2.0-600M".
- Adapt the `custom_dataset.py` to the specification of your dataset if needed.
- Update the `config_300m.yaml` or `config_600m.yaml` based on the Prithvi EO v2 variant you choose to run.
- if you made changes to the dataset script (e.g. want to use a different dataset), then also update the `main_prithvi_aquaculture.py` and then run it from the CLI:
```bash
CUDA_VISIBLE_DEVICES=[replace_with_the_GPU_index] torchrun --rdzv_endpoint=0.0.0.0:29500 main_prithvi_aquaculture.py
```
### Note: 
To run the Prithvi EO v2 model variants from scratch follow the instructions in step 3 but use the `main_prithvi_aquaculture_scratch.py` module instead.

### Note on running the baseline UNet model
- To run the unet you need to use a different repo: [multi-temporal-crop-classification-baseline](https://github.com/ClarkCGA/multi-temporal-crop-classification-baseline)
- Update the `config_unet.yaml` and add it to the config folder of the "multi-temporal-crop-classification-baseline" repo and follow the readme instructions of that repo.

### Step 4. How to do inference

- Make sure the inference section of config.py is uptodated.
- Run from CLI:
```bash
CUDA_VISIBLE_DEVICES=[replace_with_the_GPU_index] torchrun --rdzv_endpoint=0.0.0.0:29500 inference.py
```

## Fine-tuned model weights to use or replicate our work

## Aquaculture Dataset

## How to Reference our work