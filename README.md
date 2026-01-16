# Medical Slice Transformer: Improved Diagnosis and Explainability on 3D Medical Images with DINOv2 


### Private Data
* Add your own dataset to [mst/data/datasets](mst/data/datasets)
* Add your own dataset to `get_dataset()` in [scripts/main_train.py](scripts/main_train.py)  

## Run Training
### Option A: Use Trained Models
Skip training and download the weights from [Zenodo](https://doi.org/10.5281/zenodo.14500631).
### Option B: Train Models
Run Script: [scripts/main_train.py](scripts/main_train.py)
* Eg. `python scripts/main_train.py --dataset LIDC --model ResNet`
* Use `--model` to select:
    * ResNet = 3D ResNet50, 
    * ResNetSliceTrans = MST-ResNet, 
    * DinoV2ClassifierSlice = MST-DINOv2  

## Step 4: Predict & Evaluate Performance
Run Script: [scripts/main_predict.py](scripts/main_predict.py)
* Eg. `python scripts/main_predict.py --run_folder LIDC/ResNet`
* Use `--get_attention` to compute saliency maps
* Use `--get_segmentation` to compute segmentation masks and DICE score 
* Use `--use_tta` to enable Test Time Augmentation


# Run Attention to get Importance

------------------------
# MST – Explainable AI & Faithfulness Evaluation

Step 1 - Create Environment from `environment.yaml`

```python
conda env create -f environment.yaml
```

Step 2 - Install `requirements.txt`
```bash
pip install -r requirement.txt
```

# Train Model
Run Training Model
```bash
python scripts/main_train.py --dataset Local --model DinoV2ClassifierSlice
```



This repository extends the MST (Multi-Slice Transformer) framework with explainable AI (XAI) methods and faithfulness evaluation for 3D medical images.
The pipeline is designed to be:
* model-faithful (ViT / patch-based)
* method-agnostic (supports multiple XAI methods)
* reproducible (saliency saved once, reused for evaluation)

## Project Structure
mst_xai/
├── xai_methods/
│   ├── base.py                  # BaseSaliencyMethod interface
│   ├── gradcam.py               # Grad-CAM (baseline, not MST-faithful)
│   └── attention.py             # Attention-based saliency (primary method)
│
├── evaluation/
│   └── deletion.py              # Deletion faithfulness metric
│
├── utils/
│   └── load_saliency.py          # Load .pt / .npy saliency files
│
scripts/
├── run_attention.py             # Generate attention saliency
├── run_gradcam.py               # Generate Grad-CAM saliency
└── run_deletion.py              # Evaluate Deletion using saved saliency

## Running Deletion Evaluation
python scripts/run_deletion.py \
  --run_folder Local/DinoV2ClassifierSlice_Final \
  --xai_method attention \
  --max_samples 10
Outputs:
results/
└── Local/DinoV2ClassifierSlice_Final/
    ├── deletion_attention.csv
    ├── deletion_attention_summary.csv
    └── deletion_curves/attention/ (optional)


python scripts/run_insertion.py \
  --run_folder Local/DinoV2ClassifierSlice_Final \
  --xai_method attention \
  --steps 20 \
  --baseline zero \
  --save_curves
