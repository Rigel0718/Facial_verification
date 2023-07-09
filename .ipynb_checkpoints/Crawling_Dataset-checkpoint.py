from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
import random
import cv2 
import os
import torch
import numpy as np
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import copy

def img_loader(path) :
    try : 
        with open(path, 'rb') as f :
            img = cv2.imread(path) 
            if len(img.shape) == 2 :
                img = np.stack([img] * 3, 2)
        
            return img
    except IOError :
        print('Cannot load image' + path)

def get_test_path(TEST_FILE_PATH, label) :
    positive_image = []
    negative_image = []
    for dirname,_, filenames in os.walk(TEST_FILE_PATH) :
        for filename in filenames :
            img_path = os.path.join(dirname,filename)
            img_path = str(img_path)
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
            img_label= filename.split('_')[-2]
            set_image_label.add(img_label)
    
    return image_path_list, list(set_image_label)

def image_path2block(IMAGE_PATHS, transforms) :
    image_list = []
    for _path in IMAGE_PATHS :
        img = img_loader(_path)
        if transforms is not None :
            img = transforms(img)
        else :
            img = torch.from_numpy(img)
        image_list.append(img)
    output = torch.stack(image_list,0)

    return output


class Crawling_Dataset(Dataset) :
    def __init__(self, Enrolled_FILE_PATH, TEST_FILE_PATH, transforms=None, loader=img_loader) :
        _enrolled_file_path, _labels = get_images_labels(Enrolled_FILE_PATH)
        
        self.enrolled_file_path = _enrolled_file_path
        self.name_labels = _labels
        self.num_labels = len(self.name_labels)
        self.num_labels_dict = labels_NAME2NUM(_labels)
        self.transforms = transforms
        self.test_file_path = TEST_FILE_PATH
        self.loader = loader

    def __len__(self) :
        return len(self.enrolled_file_path)
    
    def get_label_dict(self) :
        return self.num_labels_dict
    
    def __getitem__(self, index) :
        image_path = self.enrolled_file_path[index]
        filename = image_path.split('/')[-1]
        image_label = filename.split('_')[-2]
        
        img = self.loader(image_path)

        if self.transforms is not None :
            img = self.transforms(img)
        else :
            img = torch.from_numpy(img)
        
        img = img.unsqueeze(0)
        
        positive_image_path, negative_image_path = get_test_path(self.test_file_path,image_label)

        positive_image_block = image_path2block(positive_image_path, transforms=self.transforms)
        negative_image_block = image_path2block(negative_image_path, transforms=self.transforms)
        # print(positive_image_block.shape)
        # print(negative_image_block.shape)
        # print(img.shape)

        # result = {'enrolled_image' : img,
        #           'positive_image_path' : positive_image_path,
        #           'negative_image_path' : negative_image_path,
        #           'image_label' : image_label,
        #           'image_path' : image_path,
        #           'test' : filename}
        # print(type(positive_image_path), type(negative_image_path), type(image_label))
        # print(positive_image_path)
        # print(result)
        return img, image_label, positive_image_block, negative_image_block


if __name__ == '__main__' :
    train_path = '/opt/ml/data/celeb_30/cut_train'
    test_path = '/opt/ml/data/celeb_30/cut_test'

    transform = transforms.Compose([
        np.float32,
        transforms.ToTensor(),  # range [0, 255] -> [0.0, 1.0]
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    dataset = Crawling_Dataset(Enrolled_FILE_PATH=train_path, TEST_FILE_PATH=test_path, transforms=transform)
    train_lodaer = data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=False)
    # print(len(dataset))
    for a,b,c,d in train_lodaer :
        print(a.shape)
        print(b)
        print(c.shape)
        print(d.shape)
        break
