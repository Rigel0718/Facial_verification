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
import argparse
import yaml

from torchvision.transforms import Resize
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization

from loss.semihardtriplet import TripletLoss
from backbone.cbam import SEModule
from test.Validation_statistic import validation
from dataset.Crawling_Dataset import Crawling_Nomal_Dataset
from utils.set_seed import setup_seed

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--config', help='Path to YAML configuration file', default='/opt/ml/Facial_verification/config.yaml')
parser.add_argument('--save_file_name', type=str, default='re_bat64_lr_1e-3')
parser.add_argument('--train_data_path', type=str, default='/opt/ml/data/celeb/train')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--total_epoch', type=int, default=15)
parser.add_argument('--save_num', type=int, default=5)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--validation', type=bool, default=False)
parser.add_argument('--seed_num', type=int, default=22)

args = parser.parse_args()

with open(args.config, 'r') as file :
     config=yaml.safe_load(file)
parser.set_defaults(**config)
args = parser.parse_args()
print('V : ', args)

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


def save_model(model, epoch, file_name='test_model.pt', final=False):
    if final :
        output_path = os.path.join(file_name, 'final.pt')
    else :
        output_path = os.path.join(file_name, f'{epoch+1}.pt')
    torch.save(model.state_dict(), output_path)

def train() :

    setup_seed(args.seed_num)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log init
    save_dir = os.path.join('./workspace/'+ args.save_file_name+ '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))

    os.makedirs(save_dir)

    if args.wandb_logging : 
        wandb.init(entity=args.wandb_entity,
                   project=args.wandb_project,
                   name=args.save_file_name)

    transform = transforms.Compose([
        np.float32,
        transforms.ToTensor(),  # range [0, 255] -> [0.0, 1.0]
        Resize((160, 160), antialias=True),
        fixed_image_standardization
    ])
    
    train_dataset = Crawling_Nomal_Dataset(args.train_data_path, transforms=transform)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=8, drop_last=False)
    
    test_dataset = Crawling_Nomal_Dataset(args.test_path, transforms=transform)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)



    # model = SEResNet_IR(50, feature_dim=128, mode='se_ir')
    model = InceptionResnetV1(classify=False, pretrained='vggface2')
    
    # margin = ArcMarginProduct(in_feature=128, out_feature=train_dataset.class_nums, s=32.0)
    criterion = TripletLoss(device=device)    
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    # optimizer_ft = optim.SGD([
    #     {'params': model.parameters(), 'weight_decay': 5e-4},
    #     {'params': margin.parameters(), 'weight_decay': 5e-4}
    #     ], lr=0.1, momentum=0.9, nesterov=True)

    optimizer_ft = optim.SGD(
            params=model.parameters(),
            lr=1e-3,
            momentum=0.9,
            dampening=0,
            nesterov=False,
            weight_decay=1e-4
        )
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[6, 11, 16], gamma=0.1)

    model = model.to(device)
    # margin = margin.to(device)

    best_acc = 0

    for epoch in tqdm(range(1, args.total_epoch + 1), leave=True) :
        
        model.train()
        progress_bar = enumerate(tqdm(trainloader))

        for data in progress_bar :
            
            img, label = data[1][0].to(device), data[1][1][1].to(device)
            

            optimizer_ft.zero_grad()
            
            feature = model(img)
            # output = margin(feature, label)
            output_loss = criterion(label, feature)
            output_loss.backward()
            optimizer_ft.step()
            exp_lr_scheduler.step()
        # validation
        if args.validation :
            with torch.no_grad() :
                model.eval()
                accuracy, recall, f1, precision = validation(model, test_data_loader)  
                if args.wandb_logging :
                        wandb.log({'accuracy' : accuracy, 'recall' : recall, 'f1' : f1, 'precision' : precision})
                if (epoch + 1) % args.save_num == 0 :
                    save_model(model, epoch ,save_dir)
                if best_acc < accuracy :
                    best_acc = accuracy
                    save_model(model, epoch, save_dir, final=True)

if __name__ == '__main__' :
    train()

            

                

            

