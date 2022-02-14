from classes_resnet import *
from torchvision.datasets.utils import download_url

if __name__ == '__main__':
    # Dowload the dataset
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    download_url(dataset_url, '.')

    # Extract from archive
    # with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
    #     tar.extractall(path='./data')

    data_dir = './data/cifar10'
    classes = os.listdir(data_dir + "/train")

    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                             tt.RandomHorizontalFlip(),
                             # tt.RandomRotate
                             # tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)),
                             # tt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                             tt.ToTensor(),
                             tt.Normalize(*stats,inplace=True)])
    valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])

    train_ds = ImageFolder(data_dir+'/train', train_tfms)
    valid_ds = ImageFolder(data_dir+'/test', valid_tfms)

    batch_size = 350

    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)

    device = get_default_device()
    print(device)

    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)

    model = to_device(ResNet9(3, 10), device)
    # print(model)
    history = [evaluate(model, valid_dl)]
    print(history)

    epochs = 15
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam

    history += fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl,
                             grad_clip=grad_clip,
                             weight_decay=weight_decay,
                             opt_func=opt_func)
    plot_accuracies(history)
    plot_losses(history)
    plot_lrs(history)