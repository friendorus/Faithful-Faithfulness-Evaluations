import argparse
from pathlib import Path
from datetime import datetime
import wandb 
import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor


from mst.data.datasets.dataset_3d_duke import DUKE_Dataset3D
from mst.data.datasets.dataset_3d_lidc import LIDC_Dataset3D
from mst.data.datasets.dataset_3d_mrnet import MRNet_Dataset3D
from mst.data.datasets.dataset_3d_local import Local_Dataset3D

from mst.data.datamodules import DataModule
from mst.models.resnet import ResNet, ResNetSliceTrans
from mst.models.dino import DinoV2ClassifierSlice

def get_dataset(name, split, **kwargs):
    if name == 'DUKE':
        return DUKE_Dataset3D(split=split, **kwargs)
    elif name == 'LIDC':
        return LIDC_Dataset3D(split=split, **kwargs)
    elif name == 'MRNet':
        return MRNet_Dataset3D(split=split, **kwargs)
    elif name == 'Local':
        return Local_Dataset3D(split=split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def get_model(name, num_classes=2, **kwargs):
    if name == 'ResNet':
        return ResNet(in_ch=1, out_ch=num_classes, spatial_dims=3, **kwargs)
    elif name == 'ResNetSliceTrans':
        return ResNetSliceTrans(in_ch=1, out_ch=num_classes, spatial_dims=2, **kwargs)
    elif name == 'DinoV2ClassifierSlice':
        return DinoV2ClassifierSlice(in_ch=3, out_ch=num_classes, spatial_dims=2, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['DUKE', 'LIDC', 'MRNet', 'Local'])
    parser.add_argument('--model', type=str, required=True, choices=['ResNet', 'ResNetSliceTrans', 'DinoV2ClassifierSlice'])
    parser.add_argument('--path_root_output', type=str, default='./runs', help="Root output path")
    args = parser.parse_args()

    #------------ Settings/Defaults ----------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path(args.path_root_output) / args.dataset / f'{args.model}_{current_time}_multi'
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')

    # ------------ Load Data ----------------
    ds_train = get_dataset(args.dataset, split='train', flip=True, noise=True, random_center=True, random_rotate=True)
    ds_val = get_dataset(args.dataset, split='val')
    
    samples = len(ds_train) + len(ds_val)
    batch_size = 2 
    accumulate_grad_batches = 1 
    steps_per_epoch = samples / batch_size / accumulate_grad_batches

    class_counts = ds_train.df[ds_train.LABEL].value_counts()
    class_weights = 0.5 / class_counts
    weights = ds_train.df[ds_train.LABEL].map(lambda x: class_weights[x]).values

    dm = DataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_val,
        batch_size=batch_size, 
        pin_memory=True,
        weights=weights,
        num_workers=0, # Adjust based on your system from 24
        num_train_samples=min(len(ds_train), 2000)
    )

    # ------------ Initialize Model ------------
    # Determine number of output classes based on dataset labels
    num_classes = len(ds_train.df[ds_train.LABEL].unique())
    
    model = get_model(args.model, num_classes=num_classes,
                    #   use_registers = True,
                    #   model_size='s',
                    #   use_bottleneck=True,
                    #   use_slice_pos_emb=True,
                    #   rotary_positional_encoding='RoPE'
                    #   rotary_positional_encoding='LiRE'
                    )
    
    # -------------- Training Initialization ---------------
    to_monitor = "val/AUC_ROC"
    min_max = "max"
    log_every_n_steps = 50
    logger = WandbLogger(project=f'Classifier_{args.dataset}', name=type(model).__name__, log_model=False)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0,
        patience=50,
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir),
        monitor=to_monitor,
        save_last=True,
        save_top_k=1,
        mode=min_max,
    )
    trainer = Trainer(
        accelerator=accelerator,
        accumulate_grad_batches=accumulate_grad_batches,
        precision='16-mixed',
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, lr_monitor, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=log_every_n_steps,
        limit_val_batches=min(len(ds_val), 200),
        max_epochs=1000,
        num_sanity_val_steps=2,
        logger=logger
    )

    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(path_run_dir, checkpointing.best_model_path)

    wandb.finish(quiet=True)