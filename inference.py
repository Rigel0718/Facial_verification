from torch.utils.data import Dataset, DataLoader
from Crawling_Dataset import Crawling_Nomal_Dataset
import torch
import cv2 
import os
import numpy as np
from torchvision import transforms
import torch.utils.data as data
from facenet_pytorch import MTCNN, fixed_image_standardization, InceptionResnetV1
import math
import tqdm as tqdm
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from torchvision.transforms import Resize
# from cheff import bring

from Embedding import Embedding_vector, Embeddings_Manager
from Label_DataFrame import Label_DataFrame

model_path = '/opt/ml/insightface/recognition/arcface_torch/work_dirs/wf4m_r50_epoch20/model.pt'
train_path = '/opt/ml/data/celeb/cut_train'
test_path = '/opt/ml/data/celeb/cut_test' 


facenet = InceptionResnetV1(classify=False, pretrained='vgg_casia')

transform = transforms.Compose([
        np.float32,
        transforms.ToTensor(),  # range [0, 255] -> [0.0, 1.0]
        Resize((112, 112)),
        fixed_image_standardization
    ])

test_dataset = Crawling_Nomal_Dataset(test_path, transforms=transform)
train_dataset = Crawling_Nomal_Dataset(train_path, transforms=transform)
test_data_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=False)
train_data_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=False)

facenet_vector= Embedding_vector(model=facenet, transform=transform)
facenet_vector_imform = Embeddings_Manager(file_path=test_path, embedding_vector=facenet_vector, dataloader=test_data_loader)
facenet_identities = facenet_vector_imform.get_label_per_path_dict()
facenet_path2embedding = facenet_vector_imform.get_path_embedding_dict()
facenet_df = Label_DataFrame(identities=facenet_identities)
positive_df = facenet_df.get_positive_df()
negative_df = facenet_df.get_negative_df()
facenet_label_df = facenet_df.concate()





