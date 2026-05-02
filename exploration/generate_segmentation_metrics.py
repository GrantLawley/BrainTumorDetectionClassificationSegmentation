import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "segmentation_models"))
from dataloader import MultiTaskBrainDataset
from tumor_segmentation_v1_jonah import PSPNet, dice_coefficient

def generate_performance_table(model_path, data_path, device):
    model = PSPNet(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_dataset = MultiTaskBrainDataset(base_dir=data_path, 
                                         mode="segmentation", 
                                         train_or_test="test", 
                                         augment=False)
    
    # for storing results
    results = {"glioma": [],
               "meningioma": [],
               "pituitary": [],
               "no_tumor": []}
    print(f"Processing {len(test_dataset)} test samples...")
    
    # evaluate
    with torch.no_grad():
        for i in range(len(test_dataset)):
            image, mask = test_dataset[i]
            img_path, _ = test_dataset.samples[i]
            
            filename = os.path.basename(img_path)
            parent_dir = os.path.basename(os.path.dirname(img_path))

            if parent_dir == "no_tumor":
                tumor_type = "no_tumor"
            else:
                tumor_type = "unknown"
                for class_name in ["glioma", "meningioma", "pituitary"]:
                    class_check_path = os.path.join(data_path, "classification_task", "test", class_name, filename)
                    if os.path.exists(class_check_path):
                        tumor_type = class_name
                        break

            if tumor_type == "unknown":
                continue # Skip if mapping fails

            image = image.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device).float()
            output = model(image)
            dice = dice_coefficient(output, mask)
            results[tumor_type].append(dice)

    # make table
    summary_rows = []
    all_scores = []

    for t_type in ["glioma", "meningioma", "no_tumor", "pituitary"]:
        scores = np.array(results[t_type])
        all_scores.extend(results[t_type])
        
        summary_rows.append({"TUMOR TYPE": t_type.capitalize().replace("_", " "),
                             "N": len(scores),
                             "MEAN DICE": np.mean(scores) if len(scores) > 0 else 0,
                             "STD": np.std(scores) if len(scores) > 0 else 0,
                             "MIN": np.min(scores) if len(scores) > 0 else 0,
                             "MAX": np.max(scores) if len(scores) > 0 else 0})

    # add stats
    all_scores = np.array(all_scores)
    summary_rows.append({"TUMOR TYPE": "Overall",
                         "N": len(all_scores),
                         "MEAN DICE": np.mean(all_scores),
                         "STD": np.std(all_scores),
                         "MIN": np.min(all_scores),
                         "MAX": np.max(all_scores)})

    # print
    df = pd.DataFrame(summary_rows)
    pd.options.display.float_format = '{:.4f}'.format
    print("\nTest set performance by tumor type")
    print(f"n = {len(all_scores)} samples\n")
    print(df.to_string(index=False))

if __name__ == "__main__":
    proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_file = os.path.join(proj_dir, "results", "segmentation", "tumor_segmentation_v1_jonah", "bias_field_morph_best.pt")
    data_dir = os.path.join(proj_dir, "brisc2025")
    
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    if os.path.exists(model_file):
        generate_performance_table(model_file, data_dir, device)
    else:
        print(f"Error: Could not find model file at {model_file}")