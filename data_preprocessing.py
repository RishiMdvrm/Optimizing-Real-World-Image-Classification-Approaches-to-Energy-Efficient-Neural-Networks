# data_preprocessing.py
import os
import shutil
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def organize_validation_data(data_dir):
    """
    Organize Tiny ImageNet validation images into class-specific folders.
    Uses the 'val_annotations.txt' file to move images.
    """
    val_dir = os.path.join(data_dir, "val")
    val_images_dir = os.path.join(val_dir, "images")
    val_annotations_path = os.path.join(val_dir, "val_annotations.txt")
    
    with open(val_annotations_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            filename = parts[0]
            class_id = parts[1]
            class_dir = os.path.join(val_dir, class_id)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            shutil.move(os.path.join(val_images_dir, filename), os.path.join(class_dir, filename))
    shutil.rmtree(val_images_dir)
    print("Validation images organized into class directories!")

def get_data_loaders(data_dir, batch_size=64, num_workers=4, image_size=224):
    """
    Create DataLoaders for training and validation with images resized to 224x224.
    """
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader

if __name__ == "__main__":
    data_dir = "tiny-imagenet-200"
    # Uncomment the next line if you haven't organized the validation images yet.
    # organize_validation_data(data_dir)
    train_loader, val_loader = get_data_loaders(data_dir)
    print("Data loaders created.")
    print("Train loader size:", len(train_loader))
    print("Validation loader size:", len(val_loader))
