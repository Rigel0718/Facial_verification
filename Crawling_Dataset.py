from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
import random
import cv2 
import os
import torch
import numpy as np

def get_test_path(TEST_FILE_PATH, label) :
    positive_image = []
    negative_image = []
    for dirname,_, filenames in os.walk(TEST_FILE_PATH) :
        for filename in filenames :
            img_path = os.path.join(dirname,filename)
            img_label = filename.split('_')[-2]
            if img_label == label :
                positive_image.append(img_path)
            else :
                negative_image.append(img_path)
    
    negative_path_num = np.random.choice(len(negative_image), len(positive_image)) 
    test_negative_image = []

    for idx in negative_path_num :
        test_negative_image.append(negative_image[idx])
    
    return positive_image, test_negative_image




def labels_NAME2NUM(set_img_label):
    label_dict = {}
    for i, key in enumerate(set_img_label) :
        label_dict[i] = key
    return label_dict


def get_images_labels(FILE_PAtH) :
    image_path_list = []
    set_image_label = set()
    for dirname,_, filenames in os.walk(FILE_PAtH) :
        for filename in filenames :
            img_path = os.path.join(dirname,filename)
            image_path_list.append(img_path)
            _,img_label,label_num = filename.split('_')
            set_image_label.add(img_label)
    
    return image_path_list, list(set_image_label)


class Crawling_Dataset(Dataset) :
    def __init__(self, Enrolled_FILE_PATH, TEST_FILE_PATH, transforms=None) :
        _enrolled_file_path, _labels = get_images_labels(Enrolled_FILE_PATH)
        
        self.enrolled_file_path = _enrolled_file_path
        self.name_labels = _labels
        self.num_labels_dict = labels_NAME2NUM(_labels)
        self.transforms = transforms
        self.test_file_path = TEST_FILE_PATH

    def __len__(self) :
        return len(self.enrolled_file_path)
    
    def get_label_dict(self) :
        return self.num_labels_dict
    
    def __getitem__(self, index) :
        image_path = self.enrolled_file_path[index]
        image_label = image_path.split('_')[-2]
        
        if self.transforms is not None :
            img = self.transforms(img)
        else :
            img = torch.from_numpy(img)
        
        positive_image_path, negative_image_path = get_test_path(self.test_file_path,image_label)

        result = {'enrolled_image' : img,
                  'positive_image_path' : positive_image_path,
                  'negative_image_path' : negative_image_path,
                  'image_label' : image_label}
        
        return result



