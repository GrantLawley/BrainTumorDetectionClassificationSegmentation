from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataloader import *
import os, sys
sys.path.append("..") # needed to import dataloader.py
from dataloader import *
proj_dir = '/'.join(os.getcwd().split('/')[:-1])

# Initialize for 4-class classification
data_path = f"{proj_dir}/brisc2025"
class_dataset = MultiTaskBrainDataset(base_dir=data_path,
                                      mode="detection",
                                      train_or_test="train",
                                      augment=True,
                                      augment_bias_field=True,
                                      augment_noise=True,
                                      augment_deform=True)

# Create the DataLoader
train_loader = DataLoader(class_dataset, batch_size=32, shuffle=True)

# Usage in a training loop
for images, labels in train_loader:
    print(f"Batch images: {images.shape}, Batch labels: {labels.shape}")
    for idx, image in enumerate(images):
        label = labels[idx]
        plt.title(f'{label},{image.shape}')
        plt.imshow(image.squeeze().numpy(), cmap='gray')
        plt.axis('off')
        plt.show()

        # break
    break

