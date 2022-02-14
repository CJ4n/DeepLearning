import matplotlib.pyplot as plt
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from classes_CNN import *
import tarfile
if __name__ == '__main__':

    # Dowload the dataset
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    download_url(dataset_url, '.')

    # Extract from archive
    # with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
    #     tar.extractall(path='./data')

    data_dir = './data/cifar10'
    dataset = ImageFolder(data_dir+'/train', transform=ToTensor())

    random_seed = 420
    torch.manual_seed(random_seed)
    val_size = int(len(dataset)*0.1)

    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size*2, num_workers=True, pin_memory=True)

    device = get_default_device()
    # device = "cpu"
    train_dl = DeviceDataLoader(train_loader, device)
    val_dl = DeviceDataLoader(val_loader, device)
    model =Cifar10CnnModel()
    model = to_device(model, device)
    num_epochs = 10
    opt_func = torch.optim.Adam
    lr = 0.001
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)


