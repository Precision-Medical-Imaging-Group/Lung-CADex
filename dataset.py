import os
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPConfig, CLIPSegModel,CLIPTextConfig, CLIPVisionConfig
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torch.utils.data import Dataset
import glob
import os
from pathlib import Path
import torch
from tqdm import tqdm
import torch.nn.functional as F
transforms = Compose([Resize((224, 224)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class CLIPDatasetSixFeatures(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.all_img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        self.transform = transform
        print(len(self.all_img_list))

    def __len__(self):
        return len(self.all_img_list)
    
    def get_label(self, df, loc):
        #return str([df['subtlety'].iat[loc], df['internal structure '].iat[loc], df['calcification'].iat[loc], df['roundness'].iat[loc], df['margin'].iat[loc], df['lobulation '].iat[loc],df['spiculation'].iat[loc], df['internal texture '].iat[loc]]), torch.tensor(df['Label'].iat[loc])
        return torch.tensor([df['subtlety'].iat[loc], df['internal structure '].iat[loc], df['calcification'].iat[loc], df['roundness'].iat[loc], df['margin'].iat[loc], df['lobulation '].iat[loc],df['spiculation'].iat[loc], df['internal texture '].iat[loc]]), torch.tensor(df['Label'].iat[loc])

    def __getitem__(self, idx):
        file_name = str(self.df['Case No'].iloc[idx]) + '-' + f"{self.df['Slice No'].iloc[idx]:03d}" + '.jpg'
        img_path = Path(self.all_img_list[idx])
        file_name = img_path.stem
        num = None
        try:
            case_no, slice_no, num = file_name.split('-')
        except ValueError:
            case_no, slice_no = file_name.split('-')

        df = self.df[(self.df['Case No'] == int(case_no)) & (self.df['Slice No'] == int(slice_no))]
        if len(df) == 0:
            df = self.df[(self.df['Case No'] == int(1)) & (self.df['Slice No'] == int(43))]
            print(img_path)
        if num is not None:
            num = int(num) - 1 
            # label = self.get_label(df, num-1)
        else:
            num = 0
        feats, label = self.get_label(df, num)

        #print(f"Attempting to open {img_path}")  # Add this line
        image = Image.open(img_path).convert("RGB")
         
        #label = torch.tensor(self.df['Label'].iloc[idx])
        if self.transform:
            image = self.transform(image)
        return image, feats#, label