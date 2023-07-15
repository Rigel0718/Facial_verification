import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import cv2
import os
import torch

def img_loader(path) :
    try : 
        with open(path, 'rb') as f :
            img = cv2.imread(path) 
            if len(img.shape) == 2 :
                img = np.stack([img] * 3, 2)
        
            return img
    except IOError :
        print('Cannot load image' + path)

class CASIAWebFace(data.Dataset):
    def __init__(self, file_path, transform=None, loader=img_loader):
        self.transform = transform
        self.loader = loader

        image_list = []  # img_path
        label_list = []  # img_class_label_num

        for dirname, _ ,filenames in os.walk(file_path) :
            for filename in filenames :
                img_name, label_num = filename.split('_')
                img_path = os.path.join(dirname, filename)
                image_list.append(img_path)
                label_list.append(int(img_name))
        
        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))  # label 중복제거
        print("dataset size: ", len(self.image_list), '/','number_of_class : ', self.class_nums)
    
    def __len__(self) :
        return len(self.image_list)
    
    def get_classes(self) :
        return len(self.class_nums)

    def __getitem__(self, index) :
        img_path = self.image_list[index]
        label = self.label_list[index]

        img = self.loader(img_path)
        
        # transform

        if self.transform is not None :
            img = self.transform(img)
        else :
            img = torch.from_numpy(img)
        
        return img, label

    def __len__(self) :
        return len(self.image_list)
    
if __name__ == '__main__' :
    file_path = '/opt/ml/data/CASIA_WEBFAECE/CASIA-WebFace_crop'

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0, 1.0]
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)) # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    dataset = CASIAWebFace(file_path=file_path, transform=transform)
    train_lodaer = data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=False)
    print(len(dataset))
    for _data in train_lodaer :
        print(_data[0].shape)  # dataset 확인
        print(_data[1])
        break