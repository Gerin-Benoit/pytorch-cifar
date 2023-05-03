'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.distributions import Normal

import os
import argparse
import random
import wandb
from copy import deepcopy

from models import *
# FrEIA imports
import FrEIA.framework as Ff
import FrEIA.modules as Fm


from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')


parser.add_argument('--wandb_project', type=str, default='CIFAR10', help='wandb project name')
#parser.add_argument('--name', default="idiot without a name", help='Wandb run name')

parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--model_name', type = str, help='name of main model')
parser.add_argument('--seed', default=1, type=int, help='set the random seed (should be the same as the model from path)')
parser.add_argument('--num_workers', type=int, default=12, help='Number of workers')
parser.add_argument('--norm_layer', default='batchnorm', help='norm layer to use : batchnorm or actnorm')
parser.add_argument('--n_layers', type = int, default=3, help='number of layer for density estimator')

args = parser.parse_args()


def neg_log_likelihood_2d(target, z, log_det):
    #log_likelihood_per_dim = target.log_prob(z) + log_det
    #return -log_likelihood_per_dim.mean()
    loss = 0.5*torch.sum(z**2, 1) - log_det
    return loss

seed = args.seed
random.seed(seed)
# np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# set_determinism(seed=seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_loss = 9e99  # best loss accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if args.wandb_project == 'CIFAR10':
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
elif args.wandb_project == 'CIFAR100':
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=args.num_workers)


if args.wandb_project == 'CIFAR10':
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
else:
    classes = ...  # TO DO
    

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
if args.wandb_project == 'CIFAR10':
    num_classes = 10 
elif args.wandb_project == 'CIFAR100':
    num_classes = 100

net = ResNet50(c=0, num_classes=num_classes, norm_layer=args.norm_layer, device=device)
net = net.to(device)


in_dim = 2048
"""
dim = 1024

out_dim = 4096
res_blocks = 1
bottleneck = True
size = 1
type = 'checkerboard'
flow = RealNVP(in_dim, 
              dim, 
              out_dim, 
              res_blocks, 
              bottleneck, 
              size, 
              type)
flow = flow.to(device)
"""
"""
n_layers = 3
flow = zuko.flows.MAF(in_dim, 0, transforms=n_layers, hidden_features=[dim] * n_layers)
"""
n_layers = args.n_layers
# we define a subnet for use inside an affine coupling block
# for more detailed information see the full tutorial
dim = 2048
def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, dim), nn.ReLU(),
                         nn.Linear(dim,  dims_out))

# a simple chain of operations is collected by ReversibleSequential
flow = Ff.SequenceINN(in_dim)
for k in range(n_layers):
    flow.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc)#, permute_soft=True)
    
flow = flow.to(device)
print('net')
print(net)

print('flow')
print(flow)

target = Normal(torch.tensor(0).float().cuda(), torch.tensor(1).float().cuda())

criterion = neg_log_likelihood_2d
optimizer = optim.Adam(flow.parameters(), lr=args.lr,weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


wandb.login()
project_name = 'flow_' + args.wandb_project
wandb.init(project=project_name, entity='max_and_ben')

wandb.config.update(args)

model_name = 'flow_' + args.model_name

wandb.run.name = model_name

# Training
def train(loader, epoch, net, flow, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    net.eval()
    flow.train()
    train_loss = 0
    N_DIM = 2048
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            outputs, features = net.get_features(inputs)

        #features = features.unsqueeze(-1).unsqueeze(-1)
        z, log_det = flow(features)
        loss = criterion(target, z, log_det)
        loss = loss.mean() / N_DIM
        #loss = -flow()
        loss.backward()
        optimizer.step()
        #net.clamp_norm_layers()

        train_loss += loss.item()


        progress_bar(batch_idx, len(loader), 'Loss: %.3f'
                     % (train_loss / (batch_idx + 1)))
    wandb.log(
        {'Total Loss/train': train_loss/len(loader)},
        step=epoch)

def test(loader, epoch, net, flow, criterion):
    global best_loss
    net.eval()
    flow.eval()
    test_loss = 0
    total = 0
    N_DIM = 2048
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, features = net.get_features(inputs)
            z, log_det = flow(features)
            loss = criterion(target, z, log_det)
            loss = loss.mean() / N_DIM
        

            test_loss += loss.item()

            progress_bar(batch_idx, len(loader), 'Loss: %.3f'
                         % (test_loss / (batch_idx + 1)))
    tot_loss =  test_loss / len(loader)
    wandb.log(
        {'Total Loss/val': tot_loss},
        step=epoch)

    # Save checkpoint.
    if tot_loss < best_loss:
        print('Saving..')
        """
        nvp = net.nvp()
        model_copy = deepcopy(net)
        net = net.to(device)
        for name, p in model_copy.named_modules():
            if args.c>0:
                try:
                    remove_spectral_norm(p)
                except:
                    pass
            elif args.c<0:
                try:
                    remove_spectral_norm_conv(p)
                except:
                    pass
            else:
                pass
             
        state = {
            'net': model_copy.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        """
        state = {
            'flow':flow.state_dict(),
            'loss':tot_loss,
            'epoch':epoch}
        
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}.pth'.format(model_name))
        best_loss = tot_loss
        
for epoch in range(start_epoch, start_epoch + 30):
    train(trainloader, epoch, net, flow, criterion, optimizer)
    test(testloader, epoch, net, flow, criterion)
    scheduler.step()

wandb.finish()

