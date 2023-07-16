from typing import Any
from torch.utils.data import Dataset, DataLoader
from Crawling_Dataset import Crawling_Nomal_Dataset
import torch
import cv2 
import os
import numpy as np
from torchvision import transforms
import torch.utils.data as data
from facenet_pytorch import MTCNN, fixed_image_standardization
import math
import tqdm as tqdm
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from torchvision.transforms import Resize

def img_loader(path) :
    try : 
        with open(path, 'rb') as f :
            img = cv2.imread(path) 
            if len(img.shape) == 2 :
                img = np.stack([img] * 3, 2)
        
            return img
    except IOError :
        print('Cannot load image' + path)

class Embedding_vector :
    def __init__(self, model, transform=None) :
        self.transform = transform
        self.model = model
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __getitem__(self, image_path=None, feature=None) :
        # image_path로 기입된 경우와 image자체로 기입된 경우를 나눈다.
        if image_path is not None :
            img = img_loader(image_path)
        else : 
            img = feature
        
        if self.transform is not None : 
            img = self.transform(img)
        else :
            img = torch.from_numpy(img)

        # dataset을 거치지 않고 나온 이미지는 3차원이기 때문에 모델에 넣어줄 수 있게 4차원 변환
        if len(img.shape) == 3:         
            img = img.unsqueeze(0)
         
        img = img.to(self.device)

        # 만약 model이 GPU연산이 안되어있다면,
        if not next(self.model.parameters()).is_cuda:  
            self.model.to(self.device)
            
        embedding = self.model(img)
        embedding = embedding.to('cpu').numpy()   # embedding 연산은 CPU로 진행하기 위해 변환

        return embedding
    

class Embeddings_Manager :
    def __init__(self, file_path, embedding_vector: Embedding_vector) :
        self.functions = []
        self.file_path = file_path
        self.embedding_vector = embedding_vector

    def 
    

        
