import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import random
from gmm_utils import *
from models import *

parser = argparse.ArgumentParser(description='Get all command line arguments.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--weight_path', type=str, help='path to the weight of the unet model')

# data
parser.add_argument('--dataset_name', type=str, default='CIFAR10', help='Specify to use CIFAR10 or CIFAR100')
parser.add_argument('--save_path', type=str, help='path to a directory to save the features')
# parser.add_argument('--data_dir', type=str, required=True, help='Specify the path to the data files directory')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--num_workers', type=int, default=12, help='Number of workers for dataloaders.')
parser.add_argument('--batch_size_gmm', type=int, default=128, help='batch size for gmm')
parser.add_argument('--c', type=float, default=0, help='Lipschitz constant: 0 for no SN, positive for soft, negative '
                                                       'for hard')
parser.add_argument('--norm_layer', default='batchnorm', help='norm layer to use : batchnorm or actnorm')
parser.add_argument('--mod', action='store_true', default=False, help='use increased sensitivity: average pooling shortcut and leaky relu')
parser.add_argument('--fc_sn', action='store_true', default=False, help='apply SN on the last model MLP')

parser.add_argument('--fix_statedict', action='store_true', default=False)

args = parser.parse_args()

seed = args.seed
random.seed(seed)
# np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# set_determinism(seed=seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

if args.dataset_name == 'CIFAR10':
    num_classes = 10
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size_gmm, shuffle=True, num_workers=args.num_workers)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

elif args.dataset_name == 'CIFAR100':
    num_classes = 100
    classes = ...
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size_gmm, shuffle=True, num_workers=args.num_workers)
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

else:
    print(f"Invalid dataset name. Expect CIFAR10 or CIFAR100, but got {args.dataset_name}")
    exit(-1)

model_name = args.dataset_name
if args.c == 0:
    model_name += '_unconstrained_'
elif args.c > 0:
    model_name += '_soft_constrained_'
else:
    model_name += '_hard_constrained_'
if args.mod:
    model_name += 'sens_'
if args.fc_sn:
    model_name += 'fcsn_'

model_name += args.norm_layer + '_'
model_name += str(args.seed) + '_gmm.pth'

net = ResNet50(c=0, num_classes=num_classes, norm_layer=args.norm_layer, device=device, mod=args.mod)

net = net.to(device)
if args.norm_layer == 'actnorm':
    x = torch.rand((1, 3, 32, 32)).to(device)
    with torch.no_grad():
        _ = net(x)

state_dict = torch.load(args.weight_path)["net"]
if args.fix_statedict:
    state_dict = fix_st(state_dict)

net.load_state_dict(state_dict)
print(net)

print("==> GMM Model fitting...")
embeddings, labels = get_embeddings(
    net,
    trainloader,
    num_dim=512 * 4,  # 512 * bottleneck block expansion
    dtype=torch.double,  # dtype = torch.double per default
    device=device,
    storage_device=device,  # default, device
)
net = net.to('cpu')
torch.cuda.empty_cache()
gaussians_model, jitter_eps = gmm_fit(embeddings=embeddings, labels=labels, num_classes=num_classes)

print("==> Saving model...")
if not os.path.isdir(args.save_path):
    os.mkdir(args.save_path)
save_name = os.path.join(args.save_path, model_name)
state_dict = {'mean': gaussians_model.loc.float(), 'covariance_matrix': gaussians_model.covariance_matrix.float()}
print('mean', gaussians_model.loc.shape)
print('cov', gaussians_model.covariance_matrix.shape)
torch.save(state_dict, save_name)
