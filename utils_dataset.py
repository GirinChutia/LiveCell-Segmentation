from pycocotools.coco import COCO
import numpy as np
import tensorflow as tf
import cv2
import skimage.io as io
from typing import Any, Callable, List, Optional, Tuple
from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt

class Dataloder(tf.keras.utils.Sequence):
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):
        
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return tuple(batch)
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)   

class ImageData4:
    def __init__(
        self, 
        annotations: COCO, 
        img_ids: List[int], 
        cat_ids: List[int], 
        root_path: Path, 
        augmentation=None, 
        preprocessing=None):
        
        super().__init__()
        
        self.annotations = annotations
        self.img_data = annotations.loadImgs(img_ids)
        self.cat_ids = cat_ids
        self.files = [f'{root_path}/{img["file_name"]}' for img in self.img_data] #image file paths
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, i: int):
        
        ann_ids = self.annotations.getAnnIds(imgIds=self.img_data[i]['id'], 
                                            catIds=self.cat_ids, 
                                            iscrowd=None)
        
        anns = self.annotations.loadAnns(ann_ids)
        mask = np.max(np.stack([self.annotations.annToMask(ann) * ann["category_id"] for ann in anns]), axis=0)
        mask = np.expand_dims(mask, axis=2)
        
        try:
            img = io.read_image(self.files[i])
        except:
            img = cv2.imread(self.files[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']
            
        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']
        
        mask = mask.astype('float32')
        img = img.astype('float32')
            
        return img, mask
    
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n+1, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        image = (image-np.min(image))/(np.max(image)-np.min(image))
        plt.imshow(image)
    
    plt.subplot(1, n+1, i + 2)
    _im = (images['image'] - np.min(images['image']))/(np.max(images['image'])-np.min(images['image']))
    plt.imshow(_im)
    _m = (images['mask'] - np.min(images['mask']))/(np.max(images['mask'])-np.min(images['mask']))
    plt.imshow(_m, cmap='jet', alpha=0.4)
    plt.title('Mask Overlayed')
    plt.show()
    
def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

def get_training_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
        A.RandomCrop(height=512, width=512, p=0.5),
        A.IAAAdditiveGaussianNoise(p=0.4),
        A.IAAPerspective(p=0.1),
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.6,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=0.5),
                A.Blur(blur_limit=3, p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5),
            ],
            p=0.6,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=0.6),
                A.HueSaturationValue(p=0.6),
            ],
            p=0.6,
        ),
        A.Resize(512,512,always_apply=True),
        A.Normalize(always_apply=True)
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        A.Resize(512,512,always_apply=True),
        A.Normalize(always_apply=True)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)