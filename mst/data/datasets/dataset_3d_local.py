from pathlib import Path 
import pandas as pd 
import torch.utils.data as data 
import torchio as tio
import torch

from .augmentations.augmentations_3d import ImageOrSubjectToTensor, RescaleIntensity, ZNormalization, CropOrPad


def parse_local_data(path_root):
    """
    Parse local dataset similar to challenge dataset.
    Reads annotation.csv from the local data directory.
    
    Args:
        path_root (Path): Root directory of the dataset.
        
    Returns:
        pd.DataFrame: DataFrame with UID index and label columns.
    """
    path_root = Path(path_root) if not isinstance(path_root, Path) else path_root
    df = pd.read_csv(path_root / "metadata_unilateral" / "annotation.csv", index_col=0, dtype={'PatientID': str})
    
    # Print label distribution
    lesion_counts = df['Lesion'].value_counts().sort_index()
    for label, count in lesion_counts.items():
        print(f"Label {label}: {count} ({count/len(df)*100:.1f}%)")
    
    return df


class Local_Dataset3D(data.Dataset):
    PATH_ROOT = Path('datasets')
    LABEL = 'Lesion'

    def __init__(
            self,
            path_root=None,
            split= None,
            transform = None,
            image_resize = None,
            resample=None,
            flip = False,
            random_rotate=False,
            image_crop = (224, 224, 32),
            random_center=False,
            noise=False, 
            to_tensor = True,
            sequence = ['Sub_1'],
            item_pointers = None,
        ):
        self.path_root = self.PATH_ROOT if path_root is None else Path(path_root)
        self.path_root_data = self.path_root/'data_unilateral'
        self.split = split
        self.sequence = sequence 

        if transform is None: 
            self.transform = tio.Compose([
                tio.Resize(image_resize) if image_resize is not None else tio.Lambda(lambda x: x),
                tio.Resample(resample) if resample is not None else tio.Lambda(lambda x: x),
                tio.Flip(1), # Just for viewing, otherwise upside down
                CropOrPad(image_crop, random_center=random_center, padding_mode='minimum') if image_crop is not None else tio.Lambda(lambda x: x),
                ZNormalization(per_channel=True, per_slice=False, masking_method=lambda x:(x>x.min()) & (x<x.max()), percentiles=(0.5, 99.5)),   # 0.5, 99.5   2.5, 97.5
                # tio.Lambda(lambda x: x.moveaxis(1, 2) if torch.rand((1,),)[0]<0.5 else x ) if random_rotate else tio.Lambda(lambda x: x), # WARNING: 1,2 if Subject, 2, 3 if tensor
                tio.RandomAffine(scales=0, degrees=(0, 0, 0, 0, 0,90), translation=0, isotropic=True, default_pad_value='minimum') if random_rotate else tio.Lambda(lambda x: x),
                tio.RandomFlip((0,1,2)) if flip else tio.Lambda(lambda x: x), # WARNING: Padding mask 
                tio.Lambda(lambda x:-x if torch.rand((1,),)[0]<0.5 else x, types_to_apply=[tio.INTENSITY]) if noise else tio.Lambda(lambda x: x),
                tio.RandomNoise(std=(0.0, 0.25)) if noise else tio.Lambda(lambda x: x),

                ImageOrSubjectToTensor() if to_tensor else tio.Lambda(lambda x: x)             
            ])
        else:
            self.transform = transform


        # Get split file
        path_csv = self.path_root/'metadata_unilateral/split_new.csv'
        path_or_stream = path_csv
        self.df = self.load_split(path_or_stream, split=split)

        # Ensure the split dataframe uses UID as index when available
        if 'UID' in self.df.columns:
            self.df = self.df.set_index('UID')

        # Use provided item_pointers if specified, otherwise use all UIDs from the dataframe index
        if item_pointers is not None:
            # keep only pointers that exist in the dataframe index
            self.item_pointers = [p for p in item_pointers if p in self.df.index]
        else:
            self.item_pointers = list(self.df.index)


    def __len__(self):
        return len(self.item_pointers)

    def load_img(self, path_img):
        return tio.ScalarImage(path_img)

    def load_map(self, path_img):
        return tio.LabelMap(path_img)

    def __getitem__(self, index):
        idx = self.item_pointers[index]
        item = self.df.loc[idx]
        target = item[self.LABEL]
        # `idx` is the UID (we set item_pointers to UID values), so use it directly
        uid = idx

        # Load image(s) based on sequence(s)
        path_item = [self.path_root_data/uid/f'{sequence}.nii.gz' for sequence in self.sequence]
        if len(path_item) == 1:
            img = self.load_img(path_item[0])
        else:
            # Load multiple sequences and stack them
            imgs = [self.load_img(p) for p in path_item]
            img = imgs[0]  # For now, use first sequence; can be extended to stack
        
        img = self.transform(img)

        return {'uid':uid, 'source': img, 'target':target}


    @classmethod
    def load_split(cls, filepath_or_buffer=None, split=None):
        df = pd.read_csv(filepath_or_buffer)
        if split is not None:
            df = df[df['Split'] == split]
        return df
