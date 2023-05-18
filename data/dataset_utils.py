import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn

import sys
sys.path.append('../')
from config import get_config

class AdditionDataset(Dataset):
    def __init__(self, data, x_transform, y_transform):
        self.data = data
        self.x_transform = x_transform
        self.y_transform = y_transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = self.x_transform(x)
        y = self.y_transform(y)
        return x, y

def _read_txt_data(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        x, y = line[:7], line[7:12]
        data.append((x,y))
    return data

def decode_addition(seqs):
    index2char = {i: char for i, char in enumerate(' _0123456789+')}
    result = []
    for seq in seqs:
        result.append("".join([index2char[char] for char in seq.tolist()]))
    return result

def load_addition():
    _config_gen = get_config("general")
    data = _read_txt_data("data/addition.txt")
    X, Y = zip(*data)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=_config_gen["test_size"])

    char2index = {char: i for i, char in enumerate(' _0123456789+')}
    transform = lambda x: torch.tensor(data=[char2index[char] for char in x], dtype=torch.long)

    train_data = AdditionDataset(list(zip(x_train, y_train)), transform, transform)
    test_data = AdditionDataset(list(zip(x_test, y_test)), transform, transform)

    trainloader = DataLoader(train_data, batch_size=_config_gen["batch_size"], shuffle=True)
    testloader = DataLoader(test_data, batch_size=_config_gen["batch_size"], shuffle=False)

    return trainloader, testloader


# CIFAR10データセットをロードする関数 -> (trainloader, testloader)
def load_cifar10():
    batch_size = get_config("general", "batch_size")
    num_workers = get_config("general", "num_workers")
    
    # データの前処理
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 訓練データセット
    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # テストデータセット
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return trainloader, testloader