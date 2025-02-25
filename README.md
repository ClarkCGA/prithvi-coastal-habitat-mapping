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
- Get the pre-trained weights from HuggingFace
- Adapt the custom_dataset.py to the specification of your dataset if needed.
- Update the config.py
- if you made changes to the dataset script, then also update the main_prithvi_aquaculture.py and then run it from the CLI:
```bash
CUDA_VISIBLE_DEVICES=[replace_with_the_GPU_index] torchrun --rdzv_endpoint=0.0.0.0:29500 main_prithvi_aquaculture.py
```

### Step 4. How to do inference

- Make sure the inference section of config.py is uptodated.
- Run from CLI:
```bash
CUDA_VISIBLE_DEVICES=[replace_with_the_GPU_index] torchrun --rdzv_endpoint=0.0.0.0:29500 inference.py
```

## Fine-tuned model weights to use or replicate our work

## Aquaculture Dataset

## How to Reference our work