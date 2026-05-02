import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
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

# --- DANet Components ---

class PositionAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(PositionAttentionModule, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, H * W)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x

class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super(ChannelAttentionModule, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        proj_query = x.view(batch_size, C, -1)
        proj_key = x.view(batch_size, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1)
        proj_value = x.view(batch_size, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, C, H, W)
        return self.gamma * out + x

class DANet(nn.Module):
    def __init__(self, num_classes=1):
        super(DANet, self).__init__()
        
        # backbone 
        self.stem = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(2))

        inter_channels = 128
        self.pam = PositionAttentionModule(inter_channels)
        self.cam = ChannelAttentionModule()
        
        self.conv_pam = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(inter_channels),
                                      nn.ReLU(inplace=True))
        self.conv_cam = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                      nn.BatchNorm2d(inter_channels),
                                      nn.ReLU(inplace=True))

        self.head = nn.Sequential(nn.Conv2d(inter_channels, inter_channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.BatchNorm2d(inter_channels // 2),
                                  nn.ReLU(inplace=True),
                                  nn.Dropout2d(0.1),
                                  nn.Conv2d(inter_channels // 2, num_classes, kernel_size=1))

    def forward(self, x):
        input_size = x.shape[2:]
        
        # extract features 
        feat = self.stem(x)
        
        # dual Attention
        feat_pam = self.pam(feat)
        feat_pam = self.conv_pam(feat_pam)
        
        feat_cam = self.cam(feat)
        feat_cam = self.conv_cam(feat_cam)
        
        # fusion
        feat_fusion = feat_pam + feat_cam
        
        # final Prediction 
        out = self.head(feat_fusion)
        return F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)

# --- Dice + BCE Loss Function ---
class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, eps=1e-6, pos_weight=10.0):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.eps         = eps
        self.pos_weight  = pos_weight

    def forward(self, pred_logits, target):
        probs = torch.sigmoid(pred_logits)

        # Weighted BCE (GPU safe)
        bce = -(target * torch.log(probs + 1e-6) * self.pos_weight +
                (1 - target) * torch.log(1 - probs + 1e-6))
        bce_loss = bce.mean()

        # Dice
        intersection = (probs * target).sum(dim=(1,2,3))
        union        = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice_loss    = 1 - (2*intersection + self.eps) / (union + self.eps)
        dice_loss    = dice_loss.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

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
    #criterion = nn.BCEWithLogitsLoss() 
    criterion = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)
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
            best_model_path = os.path.join(proj_dir, "results", task, "tumor_segmentation_v2_jonah", f"{key}_best.pt")
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

    # calculate metric 
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
    nn_model = DANet
    model_name = nn_model.__name__
    print(f'    > Model: {model_name}')  

    task = "segmentation"
    os.makedirs(os.path.join(proj_dir, "results", task, "tumor_segmentation_v2_jonah"), exist_ok=True)
    png_out = os.path.join(proj_dir, "results", task, "tumor_segmentation_v2_jonah", "convergence.png")
    json_out = os.path.join(proj_dir, "results", task, "tumor_segmentation_v2_jonah", "metrics.json")

    # initialize TensorBoard writer
    # writer = SummaryWriter(log_dir=f"{proj_dir}/runs/{task}_{model_name}")
    writer = None

    # define data loaders for segmentation
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