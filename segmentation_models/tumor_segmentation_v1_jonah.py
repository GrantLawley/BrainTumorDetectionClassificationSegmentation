import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
seed = 15
torch.manual_seed(seed)   # set random seed

import matplotlib.pyplot as plt
import os, platform, json
import numpy as np
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir) # needed to import dataloader.py
from dataloader import *

# --- PSPNet Components ---

class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, bins=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.features = nn.ModuleList()
        for bin in bins:
            self.features.append(nn.Sequential(nn.AdaptiveAvgPool2d(bin),
                                               nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                                               nn.BatchNorm2d(out_channels),
                                               nn.ReLU(inplace=True)))

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), size=(h, w), mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class PSPNet(nn.Module):
    def __init__(self, num_classes=1):
        super(PSPNet, self).__init__()
        
        resnet = models.resnet18(weights=None)
        self.backbone = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                      resnet.bn1,
                                      resnet.relu,
                                      resnet.maxpool,
                                      resnet.layer1,
                                      resnet.layer2,
                                      resnet.layer3,
                                      resnet.layer4)
        
        self.ppm = PPM(512, 128)
        
        self.cls = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                                 nn.BatchNorm2d(256),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(p=0.1),
                                 nn.Conv2d(256, num_classes, kernel_size=1))

    def forward(self, x):
        input_size = x.shape[2:]
        x = self.backbone(x)
        
        # --- MPS Compatibility Fix ---
        # current feature map is (Batch, 512, 7, 7)
        # pad to 12x12 for divisiblility by 1, 2, 3, and 6
        h, w = x.shape[2], x.shape[3]
        pad_h = (6 - h % 6) if h % 6 != 0 else 0
        pad_w = (6 - w % 6) if w % 6 != 0 else 0
        
        if pad_h > 0 or pad_w > 0: 
            x = F.pad(x, (0, pad_w, 0, pad_h))  # add padding
        # -----------------------------

        x = self.ppm(x)
        x = self.cls(x)
        return F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)

# --- Loss Functions ---
# Dice + BCE Loss
class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, eps=1e-6, pos_weight=10.0):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.eps         = eps
        self.pos_weight  = pos_weight

    def forward(self, pred_logits, target):
        probs = torch.sigmoid(pred_logits)

        # weighted BCE
        bce = -(target * torch.log(probs + 1e-6) * self.pos_weight +
                (1 - target) * torch.log(1 - probs + 1e-6))
        bce_loss = bce.mean()

        # dice
        intersection = (probs * target).sum(dim=(1,2,3))
        union        = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice_loss    = 1 - (2*intersection + self.eps) / (union + self.eps)
        dice_loss    = dice_loss.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

# Dice + Focal Loss
class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, gamma=2.0, alpha=0.25, eps=1e-6):
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred_logits, target):
        # focal loss 
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        focal_loss = focal_loss.mean()

        # dice loss
        probs = torch.sigmoid(pred_logits)
        probs = probs.view(probs.size(0), -1)
        target = target.view(target.size(0), -1)

        # weighting tumor pixels and background pixels
        w_l = 1.0 / (torch.sum(target, dim=1) + self.eps)**2
        w_bg = 1.0 / (torch.sum(1 - target, dim=1) + self.eps)**2
        
        intersection = (probs * target).sum(dim=1)
        union = (probs + target).sum(dim=1)
        numerator = w_l * intersection
        denominator = w_l * union
        
        # generalized Dice
        gdl = 1 - (2. * numerator + self.eps) / (denominator + self.eps)
        gdl_loss = gdl.mean()

        return (self.focal_weight * focal_loss) + (self.dice_weight * gdl_loss)

# --- Dice coefficient ---
def dice_coefficient(pred_logits, target, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(pred_logits)
    preds = (probs > threshold).float()
    intersection = (preds * target).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean().item()

# --- Training and Evaluation ---

def train_eval(nn_model, train_loader, test_loader, writer, run_name, epochs=2):
    model = nn_model().to(device)
    # criterion = nn.BCEWithLogitsLoss() 
    # criterion = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)
    criterion = DiceFocalLoss(dice_weight=0.6, focal_weight=0.4)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    loss_history = []
    patience = 5
    best_loss = float('inf')
    trigger_times = 0
    early_stop = False

    for epoch in range(epochs):
        print(f'        |- Epoch: {epoch+1} of {epochs}', end='\r')
        model.train()
        running_loss = 0.0
        for i, (data, target) in enumerate(train_loader):
            target = (target != 0).float()  # ensure mask is binary
            data, target = data.to(device), target.to(device).float()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # log training loss per batch
            # global_step = epoch * len(train_loader) + i
            # writer.add_scalar(f'Loss/Train_{run_name}', loss.item(), global_step)

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)

        # early stopping logic
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                target = (target != 0).float()  # ensure mask is binary
                data, target = data.to(device), target.to(device).float()
                output = model(data)
                val_loss += criterion(output, target).item()

                # log sample images from the first batch 
                if i == 0:
                    preds = torch.sigmoid(output) > 0.5
                    # Create a grid: Top row = Images, Middle = GT Masks, Bottom = Predictions
                    # img_grid = vutils.make_grid(data[:4], normalize=True)
                    # gt_grid = vutils.make_grid(target[:4])
                    # pred_grid = vutils.make_grid(preds[:4].float())
                    
                    # writer.add_image(f'Images/{run_name}', img_grid, epoch)
                    # writer.add_image(f'GroundTruth/{run_name}', gt_grid, epoch)
                    # writer.add_image(f'Predictions/{run_name}', pred_grid, epoch)
        
        avg_val_loss = val_loss / len(test_loader)
        # writer.add_scalar(f'Loss/Validation_{run_name}', avg_val_loss, epoch)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            trigger_times = 0
            best_model_path = os.path.join(proj_dir, "results", task, "tumor_segmentation_v1_jonah", f"{key}_best.pt")
            torch.save(model.state_dict(), best_model_path)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f'\n        |- Early Stop on Epoch {epoch+1}')
                early_stop = True

                break

    # load best model for evaluation
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"        |- Loaded best model from epoch {epoch+1-trigger_times} for final evaluation.")

    # calculate metrics
    model.eval()
    total_correct = 0
    total_pixels = 0
    total_dice = 0
    batch_count = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).float()
            output = model(data)
            
            # dice
            current_dice = dice_coefficient(output, target) 
            total_dice += current_dice
            
            # pixel-wise accuracy
            preds = (torch.sigmoid(output) > 0.5).float()
            total_correct += (preds == target).sum().item()
            total_pixels += torch.numel(target)
            
            batch_count += 1

    # final averages
    accuracy = total_correct / total_pixels
    avg_dice = total_dice / batch_count
            
    accuracy_str = f'        |- Pixel Accuracy: {accuracy:.4f} = {total_correct} / {total_pixels}'
    dice_str = f'        |- Dice: {avg_dice:.4f} = {total_dice} / {batch_count}'
    if not early_stop:
        accuracy_str = '\n' + accuracy_str
    print(accuracy_str)
    print(dice_str)

    return loss_history, accuracy, avg_dice

if __name__ == "__main__":
    # choose device
    if platform.system() == 'Darwin':
        device = torch.device("mps" if torch.mps.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'\n    > Device: {device}')   

    # set path and model
    data_path = os.path.join(proj_dir, 'brisc2025')
    nn_model = PSPNet
    model_name = nn_model.__name__
    print(f'    > Model: {model_name}')  

    task = "segmentation"
    os.makedirs(os.path.join(proj_dir, "results", task, "tumor_segmentation_v1_jonah"), exist_ok=True)
    png_out = os.path.join(proj_dir, "results", task, "tumor_segmentation_v1_jonah", "convergence.png")
    json_out = os.path.join(proj_dir, "results", task, "tumor_segmentation_v1_jonah", "metrics.json")

    # initialize TensorBoard writer
    # writer = SummaryWriter(log_dir=f"{proj_dir}/runs/{task}_{model_name}")
    writer = None

    # Define Data Loaders for Segmentation
    def get_loader(aug=False, noise=False, bias=False, deform=False):
        dataset = MultiTaskBrainDataset(base_dir=data_path, 
                                        mode=task, 
                                        train_or_test="train",
                                        augment=aug, 
                                        augment_noise=noise, 
                                        augment_bias_field=bias, 
                                        augment_deform=deform,
                                        seed=seed)
        return DataLoader(dataset, batch_size=16, shuffle=True) # Reduced batch size for memory

    train_loaders = {'no_aug': get_loader(),
                     'affine': get_loader(aug=True),
                     'noise': get_loader(aug=True, noise=True),
                     'bias_field': get_loader(aug=True, bias=True),
                     'morph': get_loader(aug=True, deform=True)}

    test_dataset = MultiTaskBrainDataset(base_dir=data_path, mode=task, train_or_test="test", augment=False, seed=seed)
    testLoader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    max_epochs = 100
    results_dict = {}

    # run variants
    for key, loader in train_loaders.items():
        print(f'\n    > {key.replace("_", " ").title()} Model')
        loss, acc, dice = train_eval(nn_model, loader, testLoader, writer, key, epochs=max_epochs)
        
        # create labels for plot legend
        label_map = {'no_aug': 'None', 'affine': 'Affine', 'noise': 'Affine + Noise',
                     'bias_field': 'Affine + Bias Field', 'morph': 'Affine + Morphology'}
        results_dict[key] = {'loss': loss,
                             'accuracy': acc,
                             'dice': dice, 
                             'label': f"{label_map[key]} ({round(dice,3)*100:.1f} % / {round(acc,3)*100:.1f}%)"}
    
    # writer.close()

    # save JSON
    with open(json_out, "w") as f:
        json.dump(results_dict, f, indent=4)

    # plot Convergence
    plt.figure(figsize=(10, 6))
    for key in results_dict:
        plt.plot(results_dict[key]['loss'], label=results_dict[key]['label'])
    plt.title(f'{model_name} Segmentation Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCE)')
    plt.legend(title='Augmentation (Dice / Pixel Acc)')
    plt.grid(True)
    plt.savefig(png_out)
    print(f'\n    > Results saved to {json_out} and {png_out}')