import torch
import torch.nn as nn
import nibabel as nib
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
import json


class RotationRegressionModel(nn.Module):
    def __init__(self):
        super(RotationRegressionModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        self.fc_size = self._calculate_fc_size()

        self.fc1 = nn.Linear(self.fc_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def _calculate_fc_size(self):
        dummy_input = torch.zeros(1, 1, 64, 64, 64)
        output = self.conv_layers(dummy_input)
        return output.numel()

    def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(-1, self.fc_size)
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.relu(self.fc2(x))
            x = nn.functional.relu(self.fc3(x))
            return x


def preprocess_fa_map(img):
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    if np.all(img == 0) or np.max(img) - np.min(img) < 1e-6:
        return None

    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    img = torch.nn.functional.interpolate(img, size=(64, 64, 64), mode='trilinear', align_corners=False)
    img = img.squeeze().numpy()

    p1, p99 = np.percentile(img[img > 0], [1, 99])
    img = np.clip(img, p1, p99)
    return (img - p1) / (p99 - p1)


def infer_rotation(model, fa_map_path, device, output_json_path):

    try:
        img = nib.load(fa_map_path).get_fdata()
        img = preprocess_fa_map(img)
        if img is None:
            raise ValueError("Invalid FA map after preprocessing.")

        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            output_tensor = model(img_tensor).squeeze(0).cpu().numpy()  # Get the 128x1 tensor
        
        # Convert the tensor to a list and save as JSON
        output_list = output_tensor.tolist()
        with open(output_json_path, 'w') as json_file:
            json.dump(output_list, json_file)
        
        print(f"Tensor output saved to {output_json_path}")
    except Exception as e:
        print(f"Error during inference: {e}")


if __name__ == "__main__":
    MODEL_WEIGHTS_PATH = "final_model.pth"  # Replace with your model weights file path
    FA_MAP_PATH = "fa_map.nii.gz"  # Replace with your FA map file path

    OUTPUT_JSON_PATH = "features.json"  # Replace with your desired JSON output path

    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RotationRegressionModel().to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))

    # Perform inference and save the output tensor as JSON
    infer_rotation(model, FA_MAP_PATH, device, OUTPUT_JSON_PATH)