# XAI for Medical Slice Transformer
by Peachapong Poolpol
(This project is cloned from [Muller Franzes Github](https://github.com/mueller-franzes/MST))

## MST – Explainable AI & Faithfulness Evaluation

### Step 1 - Create Environment from `environment.yaml`

```python
conda env create -f environment.yaml
```

### Step 2 - Install `requirements.txt`
```bash
pip install -r requirement.txt
```

### Step 3 
```bash
pip isntall -e .
```


### Data set
* Add your ODELIA dataset to [mst/data/datasets/datasets/ODELIA](mst/data/datasets/datasets/ODELIA)

## Run Training
### Train Models
Run Script: [scripts/main_train.py](scripts/main_train.py)
* Eg. `python scripts/main_train.py --dataset ODELIA --model DinoV2ClassifierSlice`
* Use `--model` to select:
    * ResNet = 3D ResNet50, 
    * ResNetSliceTrans = MST-ResNet, 
    * DinoV2ClassifierSlice = MST-DINOv2  

## Run Predict & Evaluate Performance
Run Script: [scripts/main_predict.py](scripts/main_predict.py)
* Eg. `python scripts/main_predict.py --run_folder ODELIA/DinoV2ClassifierSlice_Final --dataset ODELIA --get_attention`
* Use `--get_attention` to compute saliency maps
* Use `--get_segmentation` to compute segmentation masks and DICE score 
* Use `--use_tta` to enable Test Time Augmentation
Outputs:
```bash
results/
└── ODELIA/DinoV2ClassifierSlice_Final/
    ├── confusion_matrix_multiclass.png
    ├── main_predict.py.txt
    ├── result.csv
    ├── roc_multiclass.png
```


## Run XAI method
## Run Attention to get Importance
Run Script: [scripts/run_attention.py](scripts/run_attention.py)
* Eg. `python scripts/run_attention.py --run_folder ODELIA/DinoV2ClassifierSlice_Final`
* Use `-- only_images` to get saliency maps
* Use `-- max_image_per_class` to set limit of saliency map that you want
* Eg. `python scripts/run_attention.py --run_folder ODELIA/DinoV2ClassifierSlice_Final --only_images --max_images_per_class 20`

Outputs:
```bash
results/
└── ODELIA/DinoV2ClassifierSlice_Final/attention
    ├── attention_spatial_summary.csv
    ├── class_0
    │   ├── image/
    │   ├── npy/
    │   ├── py/
    ├── class_1
    │   ├── image/
    │   ├── npy/
    │   ├── py/
    ├── class_2
    │   ├── image/
    │   ├── npy/
    │   ├── py/
```

## Evaluation Test
### Deletion
#### Run Deletion
Run Script: [scripts/run_deletion.py](scripts/run_deletion.py)
* Eg. `python scripts/run_deletion.py   --run_folder ODELIA/DinoV2ClassifierSlice_Final   --xai_method attention --save_curves`

Outputs:
```bash
results/
└── ODELIA/DinoV2ClassifierSlice_Final/evaluation/deletion
    ├── deletion_attention.csv
    ├── deletion_attention_summary.csv
    └── deletion_curves/attention/ (optional)
```
#### Create ROC Curve for Deletion
Run file [results/ODELIA/DinoV2ClassifierSlice_Final/evaluation/CreateCurve_deletion.ipynb](results/ODELIA/DinoV2ClassifierSlice_Final/evaluation/CreateCurve_deletion.ipynb)

### Insertion
#### Run Insertion
Run Script: [scripts/run_insertion.py](scripts/run_insertion.py)
* Eg. `python scripts/run_insertion.py   --run_folder ODELIA/DinoV2ClassifierSlice_Final   --xai_method attention --save_curves`

Outputs:
```bash
results/
└── ODELIA/DinoV2ClassifierSlice_Final/evaluation/insertion
    ├── insertion_attention.csv
    ├── insertion_attention_summary.csv
    └── insertion_curves/attention/ (optional)
```
#### Create ROC Curve for Insertion
Run file [results/ODELIA/DinoV2ClassifierSlice_Final/evaluation/CreateCurve_insertion.ipynb](results/ODELIA/DinoV2ClassifierSlice_Final/evaluation/CreateCurve_insertion.ipynb)


This repository extends the MST (Multi-Slice Transformer) framework with explainable AI (XAI) methods and faithfulness evaluation for 3D medical images.
The pipeline is designed to be:
* model-faithful (ViT / patch-based)
* method-agnostic (supports multiple XAI methods)
* reproducible (saliency saved once, reused for evaluation)

## Project Structure
```bash
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
```

## Running Deletion Evaluation
python scripts/run_deletion.py \
  --run_folder Local/DinoV2ClassifierSlice_Final \
  --xai_method attention \
  --max_samples 10



python scripts/run_insertion.py \
  --run_folder Local/DinoV2ClassifierSlice_Final \
  --xai_method attention \
  --steps 20 \
  --baseline zero \
  --save_curves
