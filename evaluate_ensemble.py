'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributions as distributions
import torchvision
import torchvision.transforms as transforms

import os
import argparse
from copy import deepcopy
from gmm_utils import *
from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR Evaluation')

parser.add_argument('--dataset_name', type=str, default='CIFAR10', help='dataset name')

parser.add_argument('--num_workers', type=int, default=12, help='Number of workers')

parser.add_argument('--c', type=float, default=0, help='Lipschitz constant: 0 for no SN, positive for soft, negative '
                                                       'for hard')
parser.add_argument('--norm_layer', default='batchnorm',
                    help='norm layer to use for constrained nets: batchnorm or actnorm')

parser.add_argument('--path_gmms', nargs='+', type=str, required=True, help='list of paths to the gmms')
parser.add_argument('--path_constrained_nets', nargs='+', type=str, required=True,
                    help='list of paths to the constrained nets')
parser.add_argument('--path_unconstrained_nets', nargs='+', type=str, required=True,
                    help='list of paths to the unconstrained nets')

parser.add_argument('--fix_statedict', action='store_true', default=False)
parser.add_argument('--mod', action='store_true', default=False, help='use increased sensitivity: average pooling shortcut and leaky relu')

parser.add_argument('--temp', type=float, default=1, help='temperature scaling')

parser.add_argument('--data_aug', type=str, default='blurring', help='data augmentation for domain shift')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if args.dataset_name == 'CIFAR10':
    num_classes = 10
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
        

elif args.dataset_name == 'CIFAR100':
    num_classes = 100
    classes = ...

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

else:
    print(f"Invalid dataset name. Expect CIFAR10 or CIFAR100, but got {args.dataset_name}")
    exit(-1)

# Model
print('==> Loading models..')


nets_unconstrained = []
for i, net_path in enumerate(args.path_unconstrained_nets):
    net = ResNet50(c=0, num_classes=num_classes, norm_layer='batchnorm', device=device)
    net = net.to(device)

    state_dict = torch.load(net_path)
    net.load_state_dict(state_dict["net"])
    print(f'Unconstrained model {i + 1} best epoch is {state_dict["epoch"]} for {state_dict["acc"]} accuracy')
    nets_unconstrained.append(net)

nets_constrained = []
x = torch.rand((1, 3, 32, 32)).to(device)
for i, net_path in enumerate(args.path_constrained_nets):
    net = ResNet50(c=0, num_classes=num_classes, norm_layer=args.norm_layer, device=device, mod=args.mod)
    net = net.to(device)

    with torch.no_grad():
        _ = net(x)

    state_dict = torch.load(net_path)
    net_dict = state_dict["net"]
    if args.fix_statedict:
        net_dict = fix_st(net_dict)

    net.load_state_dict(net_dict)
    print(f'Constrained model {i + 1} best epoch is {state_dict["epoch"]} for {state_dict["acc"]} accuracy')
    nets_constrained.append(net)

gmms_loc = []
gmms_cov = []
for _, gmm_path in enumerate(args.path_gmms):
    state_dict = torch.load(gmm_path)
    gmms_loc.append(state_dict["mean"])
    gmms_cov.append(state_dict["covariance_matrix"])
    # gmm = distributions.MultivariateNormal(loc=mean, covariance_matrix=covariance_matrix)

"""
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
"""

criterion = nn.NLLLoss()


def evaluate(testloader, nets, gmms_loc=None, gmms_cov=None, domain_shift = None):
    for net in nets:
        net.eval()
        net.to(device)

    test_mean_loss = 0
    correct_mean = 0
    total = 0
    test_wmean_loss = 0
    correct_wmean =0

    if gmms_loc is None or gmms_cov is None:

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = []
                for net in nets:
                    output= net(inputs)
                    output = F.softmax(output, dim=-1)
                    outputs.append(output)

                outputs = torch.stack(outputs)
                average_output = torch.mean(outputs, dim=0)
                mean_loss = criterion(average_output, targets)

                test_mean_loss += mean_loss.item()
                _, predicted = average_output.max(1)
                total += targets.size(0)
                correct_mean += predicted.eq(targets).sum().item()




                progress_bar(batch_idx, len(testloader), 'Average Loss: %.3f | Average Acc: %.3f%% (%d/%d)'
                             % (test_mean_loss / (batch_idx + 1), 100. * correct_mean / total, correct_mean, total))
    else:

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = []
                fms = []
                confidences = []
                for net in nets:
                    output, fm = net.get_features(inputs)
                    output = F.softmax(output, dim=-1)
                    outputs.append(output)
                    fms.append(fm)#.to('cpu'))

                outputs = torch.stack(outputs)

                #outputs[outputs>=0.5]=1
                #outputs[outputs<0.5]=0
                average_output = torch.mean(outputs, dim=0)
                mean_loss = criterion(average_output, targets)

                test_mean_loss += mean_loss.item()
                _, predicted = average_output.max(1)
                total += targets.size(0)
                correct_mean += predicted.eq(targets).sum().item()

                for i, (loc, cov) in enumerate(zip(gmms_loc, gmms_cov)):
                    cls = torch.argmax(outputs[i,:,1], dim=0)
                    print(cls, cls.shape)
                    out = gmm_get_logits_given_class(loc, cov, fms[i], cls)
                    gmm = distributions.MultivariateNormal(loc=loc, covariance_matrix=cov)
                    confidences.append(gmm_get_logits(gmm, fms[i]))
                confidences = torch.stack(confidences)

                #confidences = confidences - torch.min(confidences)
                #confidences = confidences/torch.max(confidences)


                temperature = args.temp

                confidences = F.softmax(confidences / temperature, dim=0)
                weighted_average_output = torch.sum(confidences*outputs, dim=0)#torch.mean(torch.sum(confidences*outputs, dim=0)/torch.sum(confidences, dim=0, keepdim=True), dim=0)

                '''
                print('-'*20)
                print('conf:', confidences[:,0,:])
                print('out:', outputs[:, 0, :])

                count_tensor = torch.zeros(100, dtype=torch.int64).to(device)
                max_tensor = torch.argmax(outputs,dim=-1)
                # Iterate through the rows of the input tensor

                count_tensor += torch.eq(max_tensor[0], max_tensor[1])
                count_tensor += torch.eq(max_tensor[1], max_tensor[2])
                count_tensor += torch.eq(max_tensor[2], max_tensor[1])
                print('diff',count_tensor)
                '''

                wmean_loss = criterion(weighted_average_output, targets)
                test_wmean_loss += wmean_loss.item()
                _, predicted = weighted_average_output.max(1)
                correct_wmean += predicted.eq(targets).sum().item()




                progress_bar(batch_idx, len(testloader), 'mLoss: %.3f | mAcc: %.3f%% (%d/%d) | wmLoss: %.3f | wmAcc: %.3f%% (%d/%d)'
                             % (test_mean_loss / (batch_idx + 1), 100. * correct_mean / total, correct_mean, total, test_wmean_loss / (batch_idx + 1), 100. * correct_wmean / total, correct_wmean, total))

    for net in nets:
        net = net.to('cpu')
        


evaluate(testloader, nets_constrained, gmms_loc, gmms_cov)
evaluate(testloader, nets_unconstrained)


if args.data_aug == 'blurring':
    for sigma in [1.0, 3.0, 5.0, 10.0]:
        transform_test = transforms.Compose([
            transforms.GaussianBlur(kernel_size=3, sigma=(sigma, sigma)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if args.dataset_name == 'CIFAR10':
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=100, shuffle=False, num_workers=args.num_workers)
                
        elif args.dataset_name == 'CIFAR100':
            num_classes = 100
            classes = ...

            testset = torchvision.datasets.CIFAR100(
                root='./data', train=False, download=True, transform=transform_test)
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

        else:
            print(f"Invalid dataset name. Expect CIFAR10 or CIFAR100, but got {args.dataset_name}")
            exit(-1)
        print('---------------')
        print('SIGMA:', sigma)
        evaluate(testloader, nets_unconstrained, domain_shift = args.data_aug)
        evaluate(testloader, nets_constrained, gmms_loc, gmms_cov, domain_shift = args.data_aug)
        print('---------------')
        
        
