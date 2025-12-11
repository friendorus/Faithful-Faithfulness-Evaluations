from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import os 
import math 
import torch 
import torchio as tio
import numpy as np 
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt 
import seaborn as sns 
import torch.nn.functional as F
import pandas as pd 
from torchvision.utils import save_image
from monai.metrics import compute_average_surface_distance, compute_iou, DiceMetric, compute_dice

from mst.data.datasets.dataset_3d_duke import DUKE_Dataset3D
from mst.data.datasets.dataset_3d_lidc import LIDC_Dataset3D
from mst.data.datasets.dataset_3d_mrnet import MRNet_Dataset3D
from mst.data.datasets.dataset_3d_local import Local_Dataset3D

from mst.data.datamodules import DataModule
from mst.models.resnet import ResNet, ResNetSliceTrans
from mst.models.dino import DinoV2ClassifierSlice
from mst.utils.roc_curve import plot_roc_curve, cm2acc, cm2x
from mst.models.utils.functions import tensor2image, tensor_cam2image, minmax_norm, one_hot

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

def get_model(name, **kwargs):
    if name == 'ResNet':
        return ResNet
    elif name == 'ResNetSliceTrans':
        return ResNetSliceTrans
    elif name == 'DinoV2ClassifierSlice':
        return DinoV2ClassifierSlice
    else:
        raise ValueError(f"Unknown model: {name}")


def _pred_trans(model, source, src_key_padding_mask, save_attn=False, use_softmax=True):
    # Run model
    if isinstance(model, ResNetSliceTrans) and save_attn:
        pred = model(source, src_key_padding_mask=src_key_padding_mask, save_attn=save_attn)
    else:
        with torch.no_grad():
            pred = model(source, src_key_padding_mask=src_key_padding_mask, save_attn=save_attn)

    if use_softmax: # Necessary to standardize the scale before TTA average 
        pred = torch.softmax(pred, dim=-1)

    if not save_attn:
        return pred, None, None 

    # Spatial attention     
    weight = model.get_attention_maps()  # [B*D, Heads, HW]
    weight = weight.mean(dim=1) # Mean of heads 
    spatial_shape = weight.shape[-2:] if isinstance(model, ResNetSliceTrans) else torch.tensor(source.shape[3:])//14 
    weight = weight.view(1, 1, source.shape[2], *spatial_shape)

    # Slice attention 
    weight_slice = model.get_slice_attention() # [B*D, Heads, 1]
    weight_slice = weight_slice.mean(dim=1) # Mean of heads 
    weight_slice = weight_slice.view(1, 1, -1, 1, 1)*torch.ones_like(source, device=weight.device)
    return pred, weight, weight_slice




def _pred_resnet(model, source, src_key_padding_mask, save_attn=False, use_softmax=True):
    # Run model
    if save_attn: # Grads required 
        pred = model(source, src_key_padding_mask=src_key_padding_mask, save_attn=True)
    else:
        with torch.no_grad():
            pred = model(source, src_key_padding_mask=src_key_padding_mask, save_attn=False)
    
    if use_softmax: # Necessary to standardize the scale before TTA average 
        pred = torch.softmax(pred, dim=-1)
      
    if not save_attn:
        return pred, None, None 

    weight = model.get_attention_maps()

    # Slice attention (dummy)
    weight_slice = torch.ones_like(source, device=weight.device)

    return pred, weight, weight_slice


# def run_pred(model, batch, save_attn=False, use_softmax=True, use_tta=False):
#     source, src_key_padding_mask = batch['source'], batch.get('src_key_padding_mask', None)
#     pred_func = None 
#     if isinstance(model, ResNetSliceTrans): 
#         pred_func = _pred_trans
#     elif isinstance(model, ResNet):
#         pred_func = _pred_resnet
#     elif isinstance(model, DinoV2ClassifierSlice):
#         pred_func = _pred_trans

#     pred, weight, weight_slice = pred_func(model, source, save_attn, use_softmax)    

#     if use_tta:
#         for flip_dim in [(2,), (3,), (4,), (2,3), (2,4), (3,4), (2,3,4),]:
#             pred_i, weight_i, weight_slice_i = pred_func(model, torch.flip(source, flip_dim), save_attn, use_softmax)
#             pred = pred + pred_i
#             if save_attn:
#                 weight = weight + torch.flip(weight_i, flip_dim)
#                 weight_slice = weight_slice + torch.flip(weight_slice_i, flip_dim)

#         pred = pred / 8
#         if save_attn:
#             weight = weight / 8
#             weight_slice = weight_slice / 8

#     # Interpolate to required size 
#     if save_attn:
#         weight = F.interpolate(weight, size=source.shape[2:], mode='trilinear')

#     return pred, weight, weight_slice 

def run_pred(model, batch, save_attn=False, use_softmax=True, use_tta=False):
    source, src_key_padding_mask = batch['source'], batch.get('src_key_padding_mask', None)

    # select appropriate prediction wrapper
    pred_func = None
    if isinstance(model, ResNetSliceTrans):
        pred_func = _pred_trans
    elif isinstance(model, ResNet):
        pred_func = _pred_resnet
    elif isinstance(model, DinoV2ClassifierSlice):
        pred_func = _pred_trans

    # Correct call: pass src_key_padding_mask explicitly (use keyword args for clarity)
    pred, weight, weight_slice = pred_func(
        model,
        source,
        src_key_padding_mask,
        save_attn=save_attn,
        use_softmax=use_softmax
    )

    # Test-time augmentation: evaluate flipped copies and average
    if use_tta:
        for flip_dim in [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4),]:
            # flip source
            src_flipped = torch.flip(source, flip_dim)

            # flip padding mask if it's a tensor that matches spatial dims; otherwise reuse as-is
            if isinstance(src_key_padding_mask, torch.Tensor):
                try:
                    kmask_flipped = torch.flip(src_key_padding_mask, flip_dim)
                except Exception:
                    # if flipping mask fails, fall back to original mask
                    kmask_flipped = src_key_padding_mask
            else:
                kmask_flipped = src_key_padding_mask

            pred_i, weight_i, weight_slice_i = pred_func(
                model,
                src_flipped,
                kmask_flipped,
                save_attn=save_attn,
                use_softmax=use_softmax
            )

            pred = pred + pred_i
            if save_attn:
                weight = weight + torch.flip(weight_i, flip_dim)
                weight_slice = weight_slice + torch.flip(weight_slice_i, flip_dim)

        pred = pred / 8
        if save_attn:
            weight = weight / 8
            weight_slice = weight / 8

    # Interpolate attention maps to required size
    if save_attn:
        weight = F.interpolate(weight, size=source.shape[2:], mode='trilinear')

    return pred, weight, weight_slice




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', default='./runs', type=str)
    parser.add_argument('--run_folder', default='LIDC/ResNet', type=str)
    parser.add_argument('--output_dir', default='./', type=str)
    parser.add_argument('--get_attention', action='store_true', help='Flag to get attention')
    parser.add_argument('--get_segmentation', action='store_true', help='Flag to get attention')
    parser.add_argument('--use_tta', action='store_true', help='Use test time augmentation')
    parser.add_argument('--dataset', default=None, type=str,
                    help='Explicit dataset name (LIDC, DUKE, MRNet, Local)')

    args = parser.parse_args()
    get_attention = args.get_attention
    get_segmentation = args.get_segmentation
    use_tta = args.use_tta
    print(f"Using TTA {use_tta}")

    run_folder = Path(args.run_folder)
    # dataset = run_folder.parent.name
    if args.dataset is not None:
        dataset = args.dataset
    else:
        dataset = run_folder.parent.name
    model_name = run_folder.name.split('_', 1)[0]

    #------------ Settings/Defaults ----------------
    path_run = Path(args.run_dir)/run_folder
    results_folder = 'results_tta'if use_tta else 'results'
    path_out = Path(args.output_dir)/results_folder/run_folder
    path_out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fontdict = {'fontsize': 10, 'fontweight': 'bold'}
    torch.set_float32_matmul_precision('high')

    # ------------ Logging --------------------
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO) 
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(path_out / f'{Path(__file__).name}.txt', mode='w'))

    # ------------ Load Data ----------------
    ds_test = get_dataset(name=dataset, split='test')

    dm = DataModule(
        ds_test = ds_test,
        batch_size=1, 
        num_workers=24,
        pin_memory=True,
    ) 


    # ------------ Initialize Model ------------
    model = get_model(model_name).load_best_checkpoint(path_run)
    model.to(device)
    model.eval()


    results = []
    results_seg = []
    counter = 0 
    for n, batch in enumerate(tqdm(dm.test_dataloader())):
 
        source, target = batch['source'], batch['target']
        uid = batch['uid'][0] if isinstance(batch['uid'], list) else str(batch['uid'].item())

  
        if get_segmentation:
            # Skip cases without target label 
            # if target != 1:
            #     continue 
            
            # Skip cases without at least two raters
            if 'mask_1' not in batch:
                logger.info(f"Excluding UID: {uid}")
                continue

            # Run prediction 
            pred, weight, weight_slice = run_pred(model, batch, save_attn=True, use_softmax=use_tta, use_tta=use_tta)
            
            # Transfer weights to binary segmentation mask   
            weight = weight.detach().cpu()
            seg = (weight>np.quantile(weight, 0.999)).type(torch.int16)
            seg_hot = one_hot(seg[:, 0], 2)

            seg_gt = batch['mask']
            seg_hot_gt = one_hot(batch['mask'][:,0])
            spacing = batch['affine'][0].diag()[:3]
            vol = math.prod(spacing)

            dice = compute_dice(y_pred=seg_hot, y=seg_gt, include_background=True)
            iou = compute_iou(y_pred=seg_hot, y=seg_hot_gt, include_background=True)
            assd = compute_average_surface_distance(y_pred=seg_hot, y=seg_hot_gt, 
                    include_background=True, symmetric=True, spacing=spacing.tolist())


            results_seg.append({
                'UID': uid,
                'Path': batch['path'][0],
                'Voxel':seg_gt.sum().item(),
                'Volume': (seg_gt.sum()*vol).item(),
                'Dice':dice.mean().item(),
                'IOU':iou.mean().item(),
                'ASSD': assd.mean().item(),
                'Dice_foreground':dice[0, 1].item(),
                'IOU_foreground':iou[0, 1].item(),
                'ASSD_foreground': assd[0, 1].item(),
            })



        elif get_attention:
            # Output folder  
            path_out_dir = path_out/'attention'
            path_out_dir.mkdir(parents=True, exist_ok=True)
        
            # Skip cases without target label 
            if target != 1:
                continue

            # Only eval limited number
            counter += 1
            if counter > 5:
                break 

            # Run prediction 
            pred, weight, weight_slice = run_pred(model, batch, save_attn=True, use_softmax=use_tta, use_tta=use_tta)

            
            # Clip  
            weight_slice = weight_slice.detach().cpu()
            weight_slice /= weight_slice.sum()

            weight = weight.detach().cpu()
            weight = weight.clip(*np.quantile(weight, [0.995, 0.999]))


            # Save 
            save_image(tensor2image(source), path_out_dir/f'input_{uid}.png', normalize=True)
            save_image(tensor_cam2image(minmax_norm(source), minmax_norm(weight), alpha=0.5), 
                        path_out_dir/f"overlay_{uid}.png", normalize=False)
            save_image(tensor_cam2image(minmax_norm(source), minmax_norm(weight_slice), alpha=0.5), 
                        path_out_dir/f"overlay_{uid}_slice.png", normalize=False)
            if dataset in ['LIDC']:
                save_image(tensor_cam2image(minmax_norm(source), minmax_norm(batch['mask'].detach().cpu()), alpha=0.5),
                            path_out_dir/f"overlay_{uid}_gt.png", normalize=False) 
                
        else:
            # Run prediction 
             pred, _, _ = run_pred(model, batch, save_attn=False, use_softmax=use_tta, use_tta=use_tta)

        pred = pred.cpu()


        pred_binary = torch.argmax(pred, dim=1)
        # pred = torch.sigmoid(pred)
        pred = torch.softmax(pred, dim=-1)[:, 1]

        results.extend([{
            'UID': uid,
            'GT': target[b].item(),
            'NN': pred_binary[b].item(),
            'NN_pred': pred[b].item()
        } for b in range(target.shape[0])] )

    if get_segmentation:
        df_seg = pd.DataFrame(results_seg)
        df = pd.DataFrame(results)
        df = pd.merge(df_seg, df, on="UID")
        print("Save segmentation results to ", path_out)
        df.to_csv(path_out/'results_seg.csv', index=False)
        logger.info(f"Dice: {df['Dice'].mean():.2f}±{df['Dice'].std():.2f}") 
        logger.info(f"IOU: {df['IOU'].mean():.2f}±{df['IOU'].std():.2f}") 
        logger.info(f"ASSD: {df['ASSD'].mean():.2f}±{df['ASSD'].std():.2f}") 
        logger.info(f"Dice: {df['Dice_foreground'].mean():.2f}±{df['Dice_foreground'].std():.2f}") 
        logger.info(f"IOU: {df['IOU_foreground'].mean():.2f}±{df['IOU_foreground'].std():.2f}") 
        logger.info(f"ASSD: {df['ASSD_foreground'].mean():.2f}±{df['ASSD_foreground'].std():.2f}") 


    elif not get_attention:
        df = pd.DataFrame(results)
        df.to_csv(path_out/'results.csv', index=False)


        acc = accuracy_score(df['GT'], df['NN'])
        logger.info(f"Acc: {acc:.2f}") 

        #  -------------------------- Confusion Matrix -------------------------
        cm = confusion_matrix(df['GT'], df['NN'])
        tn, fp, fn, tp = cm.ravel()
        n = len(df)
        logger.info("Confusion Matrix: TN {} ({:.2f}%), FP {} ({:.2f}%), FN {} ({:.2f}%), TP {} ({:.2f}%)".format(tn, tn/n*100, fp, fp/n*100, fn, fn/n*100, tp, tp/n*100 ))

        
        # ------------------------------- ROC-AUC ---------------------------------
        fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(6,6)) 
        y_pred_lab = np.asarray(df['NN_pred'])
        y_true_lab = np.asarray(df['GT'])
        tprs, fprs, auc_val, thrs, opt_idx, cm = plot_roc_curve(y_true_lab, y_pred_lab, axis, fontdict=fontdict)
        fig.tight_layout()
        fig.savefig(path_out/f'roc.png', dpi=300)
        logger.info("AUC {:.2f}".format(auc_val))


        #  -------------------------- Confusion Matrix -------------------------
        acc = cm2acc(cm)
        _,_, sens, spec = cm2x(cm)
        df_cm = pd.DataFrame(data=cm, columns=['False', 'True'], index=['False', 'True'])
        fig, axis = plt.subplots(1, 1, figsize=(4,4))
        sns.heatmap(df_cm, ax=axis, cbar=False, fmt='d', annot=True) 
        axis.set_title(f'Confusion Matrix ACC={acc:.2f}', fontdict=fontdict) # CM =  [[TN, FP], [FN, TP]] 
        axis.set_xlabel('Prediction' , fontdict=fontdict)
        axis.set_ylabel('True' , fontdict=fontdict)
        fig.tight_layout()
        fig.savefig(path_out/f'confusion_matrix.png', dpi=300)

        logger.info(f"Malign  Objects: {np.sum(y_true_lab)}")
        logger.info("Confusion Matrix {}".format(cm))
        logger.info("Sensitivity {:.2f}".format(sens))
        logger.info("Specificity {:.2f}".format(spec))

