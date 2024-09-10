import torch 
import random
import numpy as np
from  torch.utils.data import Dataset , DataLoader 
import torchvision.transforms as transforms
from datasets import load_dataset 


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    num_params=  sum(p.numel() for p in model.parameters())
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_param)


def loading_data(dataset_name,num_data=int):
    data = load_dataset("ethz/food101",split='train')
    data  = data.select(range(num_data))
    data = data.train_test_split(0.2) 
    train_data = data['train']
    test_data = data['test']
    return train_data , test_data 



class ImageDataset(Dataset):
    def __init__(self,data):
        self.data = data
        self.image = data['image']
        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)), 
                        transforms.ToTensor(),          
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
                    ])

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        image = self.image[index]
        image = image.convert("RGB")
        image = self.transform(image)
        return image 