# XAI for Medical Slice Transformer

by Peachapong Poolpol
(This project is adopted model code from [Muller Franzes Github](https://github.com/mueller-franzes/MST))


## MST – Explainable AI & Faithfulness Evaluation

### Step 1 - Create Environment from `environment.yaml`

```python
conda env create -f environment.yaml
```

### Step 2 - Install `requirements.txt`

```bash
pip install -r requirements.txt
```

### Step 3

```bash
pip install -e .
```

### Data set
* Add your ODELIA dataset to [mst/data/datasets/datasets/ODELIA](mst/data/datasets/datasets/ODELIA)

## Run Training

### Train Models

Run Script: [scripts/main_train.py](scripts/main_train.py)

- Eg. `python scripts/main_train.py --dataset ODELIA --model DinoV2ClassifierSlice`





## Run Predict with new clean file

Run Script: [scripts/main_predict_eval.py](scripts/main_predict_eval.py)

### Command-Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--dataset` | `str` | Yes | — | Name of the dataset to use for inference (e.g., `ODELIA`). |
| `--model_name` | `str` | No | `DinoV2ClassifierSlice` | Model architecture name. |
| `--run_dir` | `str` | No | `./runs` | Base directory containing experiment runs. |
| `--run_folder` | `str` | Yes | — | Name of the specific experiment folder inside `run_dir`. |
| `--checkpoint_name` | `str` | No | `None` | Path or filename of the checkpoint to load. Can be a file or directory. |
| `--output_dir` | `str` | No | `./` | Directory where prediction results will be saved. |
| `--use_tta` | `flag` | No | `False` | Enable test-time augmentation (TTA) during inference. |

### Usage - Basic Run

```bash
python scripts/main_predict_eval.py \
    --dataset ODELIA \
    --run_folder DINOv2ViTS/DinoV2ClassifierSlice_Final
```

```bash
python scripts/main_predict_eval.py \
    --dataset ODELIA \
    --run_folder DINOv3ViTB \
    --checkpoint_name challenge_mstv3-vit_sch_CB_sub2_best.chkpt
```

output:

```bash
results/
└── ODELIA/DinoV2ClassifierSlice_Final/
    ├── confusion_matrix_multiclass.png
    ├── main_predict.py.txt
    ├── result.csv
    └── roc_multiclass.png
```

---

---

# Run XAI method

GradCAM (original)

```bash
python scripts/run_xai.py \
  --xai_method gradcam \
  --cam_method gradcam \
  --run_folder DINOv3ViTB \
  --checkpoint_name challenge_mstv3-vit_sch_CB_sub2_best.chkpt \
  --dataset ODELIA
```

GradCAM No ReLU

```bash
python scripts/run_xai.py \
  --xai_method gradcam \
  --cam_method gradcam \
  --norelu \
  --run_folder DINOv3ViTB \
  --checkpoint_name challenge_mstv3-vit_sch_CB_sub2_best.chkpt \
  --dataset ODELIA
```

GradCAM - NON-Class specific

```bash
python scripts/run_xai.py \
  --xai_method gradcam \
  --cam_method nonclass_gradcam \
  --run_folder DINOv3ViTB \
  --checkpoint_name challenge_mstv3-vit_sch_CB_sub2_best.chkpt \
  --dataset ODELIA
```

HiResCAM
```bash
python scripts/run_xai.py \
  --xai_method gradcam \
  --cam_method hires_cam \
  --run_folder DINOv3ViTB \
  --checkpoint_name challenge_mstv3-vit_sch_CB_sub2_best.chkpt \
  --dataset ODELIA
```

HiResCAM no ReLU
```bash
python scripts/run_xai.py \
  --xai_method gradcam \
  --cam_method hires_cam \
  --norelu \
  --run_folder DINOv3ViTB \
  --checkpoint_name challenge_mstv3-vit_sch_CB_sub2_best.chkpt \
  --dataset ODELIA
```

HiResCAM - NON Class specific
```bash
python scripts/run_xai.py \
  --xai_method gradcam \
  --cam_method nonclass_hires_cam \
  --run_folder DINOv3ViTB \
  --checkpoint_name challenge_mstv3-vit_sch_CB_sub2_best.chkpt \
  --dataset ODELIA
```

Last-later Attention
```bash
python scripts/run_xai.py \
  --xai_method attention \
  --attention_method last_layer\
  --run_folder DINOv3ViTB \
  --checkpoint_name challenge_mstv3-vit_sch_CB_sub2_best.chkpt \
  --dataset ODELIA
```

Attention Rollout
```bash
python scripts/run_xai.py \
  --xai_method attention \
  --attention_method slice_weighted_rollout\
  --run_folder DINOv3ViTB \
  --checkpoint_name challenge_mstv3-vit_sch_CB_sub2_best.chkpt \
  --dataset ODELIA
```

Gradient Self Attention Maps (GradSAM)
```bash
python scripts/run_xai.py \
  --xai_method attention \
  --attention_method grad_sam\
  --run_folder DINOv3ViTB \
  --checkpoint_name challenge_mstv3-vit_sch_CB_sub2_best.chkpt \
  --dataset ODELIA
```

Gradient Rollout
```bash
python scripts/run_xai.py \
  --xai_method attention \
  --attention_method grad_rollout\
  --run_folder DINOv3ViTB \
  --checkpoint_name challenge_mstv3-vit_sch_CB_sub2_best.chkpt \
  --dataset ODELIA
```

Using DINOv2

```bash
python scripts/run_xai.py \
  --xai_method attention \
  --attention_method slice_weighted_rollout\
  --run_folder DINOv2ViTS \
  --checkpoint_name DinoV2ClassifierSlice_Final \
  --dataset ODELIA
```

Outputs:

```bash
results/
└── ODELIA/DinoV2ClassifierSlice_Final/
    ├── attention/
    │   ├── attention_spatial_summary.csv
    │   ├── class_0
    │   │   ├── image/
    │   │   ├── npy/
    │   │   └── py/
    │   ├── class_1
    │   │   ├── image/
    │   │   ├── npy/
    │   │   └── py/
    │   └── class_2
    │       ├── image/
    │       ├── npy/
    │       └── py/
    └── attention_rollout/
        ├── attention_rollout_spatial_summary.csv
        ├── class_0
        │   ├── image/
        │   ├── npy/
        │   └── py/
        ├── class_1
        │   ├── image/
        │   ├── npy/
        │   └── py/
        └── class_2
            ├── image/
            ├── npy/
            └── py/
```

## Evaluation Test

Run Script: [scripts/run_perturbation_evaluation.py](scripts/run_perturbation_evaluation.py)

- Eg.

```bash
python scripts/run_perturbation_evaluation.py \
  --run_folder DINOv3ViTB \
  --checkpoint_name challenge_mstv3-vit_sch_CB_sub2_best.chkpt \
  --dataset ODELIA \
  --xai_method gradcam \
  --mode all \
  --replacement attention_mask \
  --steps 20 \
  --save_curves
```


- Use "--mode" with choices for `deletion`, `insertion`, `negative` or `all`
  - `deletion` is for Deletion: Initail image is original image and replace from highest importance-scored patch to lowest
  - `insertion` is for Insertion: Initial image is blank image and replace from highest importance-scored patch to lowest
  - `negative` is for Negative Perturbation test: Initial image is original image and replace from _lowest_ importance-scored patch to highest
  - `all` is for all methods.

- Use "--baseline" with choices for `minimum-intensity`, `black-3`, `black-5`, `black-10`, `white-5`, `white-10`, `zero`, `mean`, `zero_conf`, `gaussian_blur`,`attention_mask`
  - All "--baseline" mean what do you want to replace that patch with except `attention_mask`
  - Use "--baseline `attention_mask`" will use Attention Masking instead of replacing
  - `minimum-intensity` mean using the lowest intensity of that image as replacing patch
  - `black-[number]` mean replace with specific value for z score that less than 0 (assume as black)
  - `white-[number]` mean replace with specific value for z score that more than 0 (assume as white)
  - `zero` mean replace with z score = 0
  - `mean` mean replace with mean value of the image
  - `zero_conf` mean replace with additional patch that assume it's zero confidences
  - `gaussian_blur` mean replace with gaussian blur in this case set kernel_size == 9 and sigma == 2

- Use "`--save_curves`" will save numpy files for each image to use to calculate avarage later

Outputs:

```bash
results/
├── ODELIA/DinoV2ClassifierSlice_Final/evaluation_results/deletion
│   ├── deletion_attention_baseline.csv
│   ├── deletion_attention_summary_baseline.csv
│   └── deletion_curves/attention/[baseline] (optional)
├── ODELIA/DinoV2ClassifierSlice_Final/evaluation_results/insertion
│   ├── insertion_attention_baseline.csv
│   ├── insertion_attention_summary_baseline.csv
│   └── insertion_curves/attention/[baseline] (optional)
└── ODELIA/DinoV2ClassifierSlice_Final/evaluation_results/negative
    ├── negative_attention_baseline.csv
    ├── negative_attention_summary_baseline.csv
    └── negative_curves/attention/[baseline] (optional)

```

#### Create ROC Curve for Deletion

Run file [results/ODELIA/DinoV2ClassifierSlice_Final/evaluation/CreateCurve_deletion.ipynb](results/ODELIA/DinoV2ClassifierSlice_Final/evaluation/CreateCurve_deletion.ipynb)

#### Create ROC Curve for Insertion

Run file [results/ODELIA/DinoV2ClassifierSlice_Final/evaluation/CreateCurve_insertion.ipynb](results/ODELIA/DinoV2ClassifierSlice_Final/evaluation/CreateCurve_insertion.ipynb)

---

This repository extends the MST (Multi-Slice Transformer) framework with explainable AI (XAI) methods and faithfulness evaluation for 3D medical images.
The pipeline is designed to be:

- model-faithful (ViT / patch-based)
- method-agnostic (supports multiple XAI methods)
- reproducible (saliency saved once, reused for evaluation)

## Project Structure

```bash
mst_xai/
├── xai_methods/
│   ├── base.py                     # BaseSaliencyMethod interface
│   ├── gradcam.py
│   └── attention.py                # Attention-based saliency (Raw Attnetion OR Attention Rollout)
│
├── evaluation_methods/
│   ├── perturbation_core_code/
│   │   └── perturbation_core.py    # Core code for perturbation for using Deletion, Insertion and NPT
│   ├── deletion.py                 # Deletion faithfulness metric
│   ├── insetion.py                 # Insertion metric
│   └── negative_perturbation.py    # Negative Perturbation Test (NPT)
│
├── utils/
│   └── load_saliency.py          # Load .pt / .npy saliency files
│
scripts/
├── main_train.py
├── main_predict.py
├── run_attention.py
└── run_perturbation_evaluation.py
```
