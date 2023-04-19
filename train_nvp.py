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
from realnvp import REALNVP

from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')


parser.add_argument('--wandb_project', type=str, default='CIFAR10', help='wandb project name')
#parser.add_argument('--name', default="idiot without a name", help='Wandb run name')

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--path', type = str, help='path to weights of main model')
parser.add_argument('--seed', default=1, type=int, help='set the random seed (should be the same as the model from path)')
parser.add_argument('--num_workers', type=int, default=12, help='Number of workers')
parser.add_argument('--norm_layer', default='batchnorm', help='norm layer to use : batchnorm or actnorm')
parser.add_argument('--c', type=float, default=0, help='Lipschitz constant: 0 for no SN, positive for soft, negative '
                                                       'for hard')   # A retirer après car argument pas nécessaire
args = parser.parse_args()


def neg_log_likelihood_2d(target, z, log_det):
	log_likelihood_per_dim = target.log_prob(z) + log_det
	return -log_likelihood_per_dim.mean()

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
net = ResNet50(c=0, num_classes=num_classes, norm_layer= args.norm_layer, device=device)
net = net.to(device)

in_dim = 2048
dim = 2048
out_dim = 4096
res_blocks = 2
bottleneck = True
size = 16
type = 'checkerboard'
nvp = RealNVP(in_dim, 
              dim, 
              out_dim, 
              res_blocks, 
              bottleneck, 
              size, 
              type):
print('net')
print(net)

print('nvp')
print(nvp)
target = Normal(torch.tensor(0).float().cuda(), torch.tensor(1).float().cuda())
"""
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
"""


if args.resume:
    # Load checkpoint.
    print('==> Resuming main model from checkpoint..')
    checkpoint = torch.load(args.path)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    #best_acc = checkpoint['acc']
    #start_epoch = checkpoint['epoch']


criterion = neg_log_likelihood_2d
optimizer = optim.SGD(nvp.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


wandb.login()
project_name = 'nvp_' + args.wandb_project
wandb.init(project=project_name, entity='max_and_ben')
model_name = project_name
if args.c==0:
    model_name += '_unconstrained_'
elif args.c>0:
    model_name += '_soft_constrained_'
else:
    model_name += '_hard_constrained_'
model_name += args.norm_layer + '_'
model_name += str(args.seed)
wandb.run.name = model_name

# Training
def train(loader, epoch, net, nvp, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    net.eval()
    nvp.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            outputs, features = net.get_features(inputs)
        z, log_det = nvp(features)
        loss = criterion(target, z, log_det)
        loss.backward()
        optimizer.step()
        #net.clamp_norm_layers()

        train_loss += loss.item()


        progress_bar(batch_idx, len(loader), 'Loss: %.3f'
                     % (train_loss / (batch_idx + 1)))
    wandb.log(
        {'Total Loss/train': train_loss/len(loader)},
        step=epoch)

def test(loader, epoch, net, nvp, criterion):
    global best_loss
    net.eval()
    nvp.eval()
    test_loss = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, features = net.get_features(inputs)
            z, log_det = nvp(features)
            
            loss = criterion(target, z, log_det)

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
        state = 
        {
            'nvp':nvp.state_dict(),
            'loss':tot_loss,
            'epoch':epoch}
        
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}.pth'.format(model_name))
        best_loss = tot_loss
        
for epoch in range(start_epoch, start_epoch + 200):
    train(trainloader, epoch, net, nvp, criterion, optimizer)
    test(testloader, epoch, net, nvp, criterion)
    scheduler.step()

wandb.finish()

