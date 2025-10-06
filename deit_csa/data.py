
import os
from typing import Tuple, List
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def build_loaders(dataset_path: str, img_size: int, batch_train: int, batch_eval: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(dataset_path, 'train')
    val_dir   = os.path.join(dataset_path, 'val')
    test_dir  = os.path.join(dataset_path, 'test')

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=transform_eval)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=transform_eval)

    class_names = train_dataset.classes

    train_loader = DataLoader(train_dataset, batch_size=batch_train, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=batch_eval, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_dataset, batch_size=batch_eval, shuffle=False,
                             num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    return train_loader, val_loader, test_loader, class_names
