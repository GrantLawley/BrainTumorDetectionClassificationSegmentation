import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "classification_models"))
from dataloader import MultiTaskBrainDataset
from tumor_classification_v1_jonah import ConvNet
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import os

def calculate_metrics(y_true, y_pred, num_classes=4):
    metrics = []
    class_names = ["Glioma", "Meningioma", "Pituitary", "No tumor"] 

    total_correct = 0
    total_samples = len(y_true)
    for i in range(num_classes):
        tp = ((y_pred == i) & (y_true == i)).sum().item()
        fp = ((y_pred == i) & (y_true != i)).sum().item()
        fn = ((y_pred != i) & (y_true == i)).sum().item()
        tn = ((y_pred != i) & (y_true != i)).sum().item()
        
        n_class = (y_true == i).sum().item()
        total_correct += tp
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append({"TUMOR TYPE": class_names[i],
                        "N": n_class,
                        "CORRECT": tp,
                        "ACCURACY": round(accuracy, 4),
                        "PRECISION": round(precision, 4),
                        "RECALL": round(recall, 4),
                        "F1": round(f1, 4)})

    # stats
    overall_acc = total_correct / total_samples
    avg_prec = sum(m['PRECISION'] for m in metrics) / num_classes
    avg_rec = sum(m['RECALL'] for m in metrics) / num_classes
    avg_f1 = sum(m['F1'] for m in metrics) / num_classes

    metrics.append({"TUMOR TYPE": "Overall",
                    "N": total_samples,
                    "CORRECT": total_correct,
                    "ACCURACY": round(overall_acc, 4),
                    "PRECISION": round(avg_prec, 4),
                    "RECALL": round(avg_rec, 4),
                    "F1": round(avg_f1, 4)})
    
    return metrics

def generate_table(model_path, data_dir, device):    
    test_dataset = MultiTaskBrainDataset(base_dir=data_dir, 
                                         mode="classification", 
                                         train_or_test="test", 
                                         augment=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # load model
    model = ConvNet().to(device) 
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    model.eval()

    all_preds = []
    all_targets = []

    # evaluate
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            target_indices = torch.argmax(targets, dim=1)
            outputs = model(images)
            if len(outputs.shape) == 4:
                outputs = F.adaptive_avg_pool2d(outputs, (1, 1)).view(outputs.size(0), -1)
            
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu())
            all_targets.append(target_indices.cpu())

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)

    # print table
    results = calculate_metrics(y_true, y_pred)
    df = pd.DataFrame(results)
    
    print("\nClassification performance by tumor type (n = 1,000 samples)")
    print(df.to_string(index=False))
    
    # save 
    # df.to_csv("classification_report.csv", index=False)
    
if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_file = os.path.join(proj_dir, "results", "classification", "tumor_classification_v1_jonah", "affine_best.pt")
    data_dir = os.path.join(proj_dir, "brisc2025")
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    if os.path.exists(model_file):
        generate_table(model_file, data_dir, device)
    else:
        print(f"Error: Could not find model file at {model_file}")