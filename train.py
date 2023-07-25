import os
import torch.utils.data
from datetime import datetime
import numpy as np
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import torch.optim as optim
import time
from tqdm.auto import tqdm
import wandb

from .dataset.WebFace_Dataset import CASIAWebFace
from margin import ArcMarginProduct
from arcFacenet import SEResNet_IR

# setting
save_file_name = 'Test_first'
train_data_path = 'file_path = /opt/ml/data/CASIA_WEBFAECE/CASIA-WebFace_crop'
batch_size = 32
total_epoch = 10
root = '/opt/ml/result'


def save_model(model, file_name='test_model.pt'):
    output_path = os.path.join(root, file_name)
    torch.save(model, output_path)

def train() :

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log init
    save_dir = os.path.join(save_file_name+ '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    # logging = init_log(save_dir)
    # _print = logging.info

    transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    
    trainset = CASIAWebFace(train_data_path, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=8, drop_last=False)



    model = SEResNet_IR(50, feature_dim=128, mode='se_ir')
    margin = ArcMarginProduct(in_feature=128, out_feature=trainset.class_nums, s=32.0)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD([
        {'params': model.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4}
        ], lr=0.1, momentum=0.9, nesterov=True)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[6, 11, 16], gamma=0.1)

    model = model.to(device)
    margin = margin.to(device)

    best_lfw_acc = 0.0
    best_lfw_iters = 0
    total_iters = 0

    for epoch in tqdm(range(1, total_epoch + 1), leave=True) :
        exp_lr_scheduler.step()
        model.train()

        for data in trainloader :
            img, label = data[0].to(device), data[1].to(device) 
            optimizer_ft.zero_grad()
            
            feature = model(img)
            output = margin(feature, label)
            total_loss = criterion(output, label)
            total_loss.backward()
            optimizer_ft.step()

            total_iters += 1

            if total_iters % 100 == 0 :
                _, predict = torch.max(output.data, 1)
                total = label.size(0)
                correct = (np.array(predict.cpu()) == np.array(label.data.cpu())).sum()

            # save_model

            # vali_lfw
            model.eval()

                

            

