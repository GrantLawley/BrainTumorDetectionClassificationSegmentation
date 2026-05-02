import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
seed = 15
torch.manual_seed(seed)   # set random seed

import matplotlib.pyplot as plt
import os, platform, json
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir) # needed to import dataloader.py
from dataloader import *

# define model
# dropout and batch normalization and/or adding layers doesn't seem to improve performance
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # input (1, 224, 224) -> conv (32, 222, 222) -> pool (32, 111, 111)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # input (32, 111, 111) -> conv (64, 109, 109) -> pool (64, 54, 54)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # fully connected layers
        self.fc1 = nn.Linear(64 * 54 * 54, 128)
        self.fc2 = nn.Linear(128, 4) # 4 output classes (0,1,2,3)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# define testing and evaluation 
def train_eval(nn_model,train_loader,test_loader,name,epochs=2):
    # initialize variables
    model = nn_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    loss_history = []
    patience = 5
    best_loss = float('inf')
    trigger_times = 0
    early_stop = False

    # training loop
    for epoch in range(epochs):
        print(f'        |- Epoch: {epoch+1} of {epochs}',end='\r')
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            target = torch.argmax(target, dim=1)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)

        # evaluate early stop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, torch.argmax(target, dim=1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(test_loader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            trigger_times = 0
            best_model_path = os.path.join(proj_dir, "results", task, "tumor_classification_v1_jonah", f"{name}_best.pt")
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

    # testing loop 
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = torch.argmax(target, dim=1)

            output = model(data)
            predict = torch.max(output.data, 1)[1]
            total += target.size(0)
            correct += (predict == target).sum().item()
    accuracy = correct / total
    accuracy_str = f'        |- Accuracy: {accuracy} = {correct} / {total}'
    if not early_stop:
        accuracy_str = '\n' + accuracy_str
    print(accuracy_str)

    return loss_history, accuracy


if __name__ == "__main__":
    # use gpu if possible
    if platform.system() == 'Darwin':
        device = torch.device("mps" if torch.mps.is_available() else "cpu")    # macOS
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # other
    print(f'\n    > Device: {device}') 

    # load data 
    data_path = os.path.join(proj_dir, 'brisc2025')
    nn_model = ConvNet
    model_name = nn_model.__name__
    print(f'    > Model: {model_name}')   
    task = "classification"
    os.makedirs(os.path.join(proj_dir, "results", task, "tumor_classification_v1_jonah"), exist_ok=True)
    png_out = os.path.join(proj_dir, "results", task, "tumor_classification_v1_jonah", "convergence.png")
    json_out = os.path.join(proj_dir, "results", task, "tumor_classification_v1_jonah", "metrics.json")

    # -- no augmentation 
    no_aug_train = MultiTaskBrainDataset(base_dir=data_path,
                                        mode=task,
                                        train_or_test="train",
                                        augment=False,
                                        seed=seed)
    no_aug_trainLoader = DataLoader(no_aug_train, batch_size=32, shuffle=True)

    # -- basic augmentation (affine transforms) 
    affine_train = MultiTaskBrainDataset(base_dir=data_path,
                                        mode=task,
                                        train_or_test="train",
                                        augment=True,
                                        seed=seed)
    affine_trainLoader = DataLoader(affine_train, batch_size=32, shuffle=True)

    # -- basic + noise augmentation 
    noise_train = MultiTaskBrainDataset(base_dir=data_path,
                                        mode=task,
                                        train_or_test="train",
                                        augment=True,
                                        augment_noise=True,
                                        seed=seed)
    noise_trainLoader = DataLoader(noise_train, batch_size=32, shuffle=True)

    # -- basic + noise + bias field augmentation 
    bias_field_train = MultiTaskBrainDataset(base_dir=data_path,
                                            mode=task,
                                            train_or_test="train",
                                            augment=True,
                                            augment_bias_field=True,
                                            seed=seed)
    bias_field_trainLoader = DataLoader(bias_field_train, batch_size=32, shuffle=True)

    # -- all augmentation (morphology added)
    morph_train = MultiTaskBrainDataset(base_dir=data_path,
                                        mode=task,
                                        train_or_test="train",
                                        augment=True,
                                        augment_deform=True,
                                        seed=seed)
    morph_trainLoader = DataLoader(morph_train, batch_size=32, shuffle=True)

    # -- test set
    test = MultiTaskBrainDataset(base_dir=data_path,
                                        mode=task,
                                        train_or_test="test",
                                        augment=False)
    testLoader = DataLoader(test, batch_size=32, shuffle=True)

    # run models
    max_epochs = 100
    print(f'\n    > No Augmentation Model')
    no_aug_name = f'{no_aug_trainLoader=}'.split('=')[0].replace('_trainLoader','')
    no_aug_loss, no_aug_accuracy = train_eval(nn_model, no_aug_trainLoader, testLoader, no_aug_name, epochs=max_epochs)
    no_aug_label = f'None ({round(no_aug_accuracy,3)*100:.1f}%)'
    print(f'\n    > Affine Augmentation Model')
    affine_name = f'{affine_trainLoader=}'.split('=')[0].replace('_trainLoader','')
    affine_loss, affine_accuracy = train_eval(nn_model, affine_trainLoader, testLoader, affine_name, epochs=max_epochs)
    affine_label = f'Affine ({round(affine_accuracy,3)*100:.1f}%)'
    print(f'\n    > Affine + Noise Augmentation Model')
    noise_name = f'{noise_trainLoader=}'.split('=')[0].replace('_trainLoader','')
    noise_loss, noise_accuracy = train_eval(nn_model, noise_trainLoader, testLoader, noise_name, epochs=max_epochs)
    noise_label = f'Affine + Noise ({round(noise_accuracy,3)*100:.1f}%)'
    print(f'\n    > Affine + Bias Field Augmentation Model')
    bias_field_name = f'{bias_field_trainLoader=}'.split('=')[0].replace('_trainLoader','')
    bias_field_loss, bias_field_accuracy = train_eval(nn_model, bias_field_trainLoader, testLoader, bias_field_name, epochs=max_epochs)
    bias_field_label = f'Affine + Bias Field ({round(bias_field_accuracy,3)*100:.1f}%)'
    print(f'\n    > Affine + Morphology Augmentation Model')
    morph_name = f'{morph_trainLoader=}'.split('=')[0].replace('_trainLoader','')
    morph_loss, morph_accuracy = train_eval(nn_model, morph_trainLoader, testLoader, morph_name, epochs=max_epochs)
    morph_label = f'Affine + Morphology ({round(morph_accuracy,3)*100:.1f}%)'
    print(' ')

    # save variables to dict and export as json
    cnn_dict = {}
    cnn_dict['no_aug'] = {'loss': no_aug_loss,
                        'accuracy': no_aug_accuracy,
                        'label': no_aug_label}
    cnn_dict['affine'] = {'loss': affine_loss,
                        'accuracy': affine_accuracy,
                        'label': affine_label}
    cnn_dict['noise'] = {'loss': noise_loss,
                        'accuracy': noise_accuracy,
                        'label': noise_label}
    cnn_dict['bias_field'] = {'loss': bias_field_loss,
                            'accuracy': bias_field_accuracy,
                            'label': bias_field_label}
    cnn_dict['morph'] = {'loss': morph_loss,
                         'accuracy': morph_accuracy,
                         'label': morph_label}
    with open(json_out, "w") as f:
        json.dump(cnn_dict, f, indent=4)

    # plot model convergence
    for key in list(cnn_dict.keys()):
        plt.plot(cnn_dict[key]['loss'], label=cnn_dict[key]['label'])
    plt.title(f'{model_name} Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(title='Augmentation (Accuracy)')
    plt.grid(True)
    plt.savefig(png_out)
