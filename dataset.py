import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# --------------------------------------
# DATASET CLASS PART
# --------------------------------------
"""

NOTES:
transformations have 2 purposes:
1. data augmentation — used during training to improve generalization and reduce overfitting
2. standardization — used in all phases (train/val/test) to normalize pixel values and convert to tensors
For the first we should use:
transforms.ToTensor() - train/test - converts PILImage to tensor, scales to [0,1]
For the second we should use:
transforms.Normalize(mean, std) - train/test - helps network converge faster, stabilizes training
grayscale images - (mean, std) - ((0.5,), (0.5,))
color images - (mean, std) - ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

Customize the rest:
1. Based on image type
grayscale images	            simple transforms (no color jittering)
color images	                add ColorJitter, RandomGrayscale, etc.
medical/scientific	            avoid strong augmentations
natural photos	                more aggressive augmentation is okay

2. Based on image resolution
small (e.g. 48x48)	            avoid aggressive cropping or resizing
large (e.g. 224x224 or more)	you can use RandomCrop, Resize, etc.

3. Based on task type
classification	                horizontal flip, rotation, crop
object detection                use bounding-box aware augmentations
segmentation	                need to apply the same transform to mask
facial emotion recognition	    very limited changes — don’t alter facial structure

4. Based on dataset size
small	                        apply more augmentation to simulate more data
large	                        minimal augmentation, rely on real variation

"""

class FER2013Dataset:
    # constructor method used to initialize the dataset loader class
    # params: base_path (path where the dataset is located)
    # params: batch_size (number of samples per batch to load during training)
    def __init__(self, base_path='archive', batch_size=64):
        # ------------------------------------------------
        # DEFINING MAIN PARAMS
        # ------------------------------------------------
        # build full paths to the training and test (used as validation) directories
        self.train_dir = os.path.join(base_path, 'train')
        self.val_dir = os.path.join(base_path, 'test')
        self.batch_size = batch_size

        # ------------------------------------------------
        # INTRODUCE TRANSFORMATIONS FOR TEST AND TRAIN DATASET
        # ------------------------------------------------
        # training transforms include:
        # - random horizontal flip (for data augmentation, so it works on 'mirrored' images)
        # - random rotation (for robustness to small image rotations)
        # - tensor conversion (turns PIL image into torch tensor)
        # - normalization (scales pixel values to [-1, 1] range)
        self.train_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # normalization for grayscale images
        ])

        # define preprocessing for validation images
        # no augmentation is used here to keep evaluation consistent
        self.val_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # ------------------------------------------------
        # CREATE TRANSFORMED DATASETS
        # ------------------------------------------------
        # create the training and testing dataset using the folder structure and training transforms
        self.train_dataset = ImageFolder(root=self.train_dir, transform=self.train_transforms)
        self.val_dataset = ImageFolder(root=self.val_dir, transform=self.val_transforms)

        # ------------------------------------------------
        # WRAP TRANSFORMED DATASETS
        # ------------------------------------------------
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    # returns training and validation dataloaders
    # access to preloaded dataloaders for use in training and evaluation loops
    def get_loaders(self):
        return self.train_loader, self.val_loader

    # returns a list of class names
    def get_classes(self):
        return self.train_dataset.classes

    # returns a dictionary mapping class names to numeric labels
    def get_class_to_idx(self):
        return self.train_dataset.class_to_idx

    # returns number of samples in training dataset
    def __len__(self):
        return len(self.train_dataset)