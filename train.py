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
import torch
from torch import distributed

from torch.nn.modules.distance import PairwiseDistance
from facenet_pytorch import InceptionResnetV1

from loss.semihardtriplet import TripletLoss
from backbone.cbam import SEModule
from test.Validation_statistic import validation
from dataset.Crawling_Dataset import Crawling_Nomal_Dataset
from utils.set_seed import setup_seed, seed_worker

# setting
save_file_name = 'Test_first'
train_data_path = '/opt/ml/data/celeb/train'
test_path = '/opt/ml/data/celeb/test'
batch_size = 10
total_epoch = 10
root = '/opt/ml/result'

try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


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

    # wandb.init(entity='hi-ai',
    #            project='semi_hard_triplet',
    #            name='test1')

    transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    
    train_dataset = Crawling_Nomal_Dataset(train_data_path, transforms=transform)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=8, drop_last=False)
    
    test_dataset = Crawling_Nomal_Dataset(test_path, transforms=transform)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=False)



    # model = SEResNet_IR(50, feature_dim=128, mode='se_ir')
    model = InceptionResnetV1(classify=False, pretrained='vggface2')

    # model = torch.nn.parallel.DistributedDataParallel(
    #     module=model, broadcast_buffers=False, device_ids=[local_rank], bucket_cap_mb=16,
    #     find_unused_parameters=True)
    
    # margin = ArcMarginProduct(in_feature=128, out_feature=train_dataset.class_nums, s=32.0)
    criterion = TripletLoss(device=device)    
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    # optimizer_ft = optim.SGD([
    #     {'params': model.parameters(), 'weight_decay': 5e-4},
    #     {'params': margin.parameters(), 'weight_decay': 5e-4}
    #     ], lr=0.1, momentum=0.9, nesterov=True)
    optimizer_ft = optim.SGD(
            params=model.parameters(),
            lr=0.1,
            momentum=0.9,
            dampening=0,
            nesterov=False,
            weight_decay=1e-5
        )
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[6, 11, 16], gamma=0.1)

    model = model.to(device)
    # margin = margin.to(device)

    total_iters = 0

    for epoch in tqdm(range(1, total_epoch + 1), leave=True) :
        exp_lr_scheduler.step()
        model.train()
        progress_bar = enumerate(tqdm(trainloader))

        for data in progress_bar :
            print(data[1][1][1].shape)
            img, label = data[1][0].to(device), data[1][1][1].to(device)
            

            optimizer_ft.zero_grad()
            
            feature = model(img)
            # output = margin(feature, label)
            output_loss = criterion(feature, label)
            # total_loss = criterion(output, label)
            
            output_loss.backward()
            # total_loss.backward()
            optimizer_ft.step()
            

            # total_iters += 1

            # if total_iters % 100 == 0 :
            #     _, predict = torch.max(output.data, 1)
            #     total = label.size(0)
            #     correct = (np.array(predict.cpu()) == np.array(label.data.cpu())).sum()

            # save_model

            # validation
            model.eval()
            accuracy, recall, f1, precision = validation(model, test_data_loader)
            print('accuracy : ',accuracy)
            print('recall : ', recall)
            print('f1 : ', f1)
            print('precision : ', precision)   
            wandb.log({'accuracy' : accuracy, 'recall' : recall, 'f1' : f1, 'precision' : precision})

if __name__ == '__main__' :
    train()

            

                

            

