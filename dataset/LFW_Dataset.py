from torchvision.datasets import LFWPeople
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
import random
import cv2 

def preprocess(img) :
    img = cv2.resize(src=img, dsize=(250,250))
    img = img / 255.0
    return img


class LFWDataset(Dataset) :
    def __init__(self, lfw_dataset) :
        self.lfw_dataset = lfw_dataset
        self.labels = [label for _, label in lfw_dataset]
        

    def __len__(self):
        return len(self.lfw_dataset)
    
    def __getitem__(self, idx):
        img1, anchor_label = self.lfw_dataset[idx]
        anchor_img = img1

        positive_idx = random.choice([i for i, label in enumerate(self.labels) if label == anchor_label])
        negative_idx = random.choice([i for i, label in enumerate(self.labels) if label != anchor_label])

        p_img, p_label = self.lfw_dataset[positive_idx]
        n_img, n_label = self.lfw_dataset[negative_idx]

        return preprocess(anchor_img), preprocess(p_img), preprocess(n_img), p_label, n_label
    

lfw_dataset = LFWPeople(root='/opt/ml/data/lfw', split='train', download=False)
lfw_triplet_dataset = LFWDataset(lfw_dataset)
batch_size = 32
lfw_dataloader = DataLoader(lfw_triplet_dataset, batch_size=batch_size, shuffle=True)
