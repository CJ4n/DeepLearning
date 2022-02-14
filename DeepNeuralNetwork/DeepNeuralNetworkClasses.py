import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.layer1 = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU()

    def forward(self, batch):
        out = self.layer1(batch)
        out = self.relu(out)
        return out

class MNISTModelv1(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.layer1 = Block(in_size, hidden_size)
        self.layer2 = Block(hidden_size, out_size)

    def forward(self, batch):
        batch = torch.flatten(batch, 1)
        out = self.layer1(batch)
        out = self.layer2(out)
        return out

    def training_step(self, batch):
        image, label = batch
        out = self(image)
        loss = F.cross_entropy(out, label)
        return loss

class MNISTModelv0(nn.Module):

    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.layer1 = nn.Linear(in_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, out_size)

    def forward(self, batch):
        batch = torch.flatten(batch, 1)
        out = self.layer1(batch)
        out = F.relu(out)
        out = self.layer2(out)
        return out

    def training_step(self, batch):
        image, label = batch
        out = self(image)
        loss = F.cross_entropy(out, label)
        return loss


def evaluate(test_dataloader, model):
    loss = 0
    accuracy = 0
    model.eval()
    for batch in test_dataloader:
        image, label = batch
        image = model(image)
        loss += F.cross_entropy(image, label)
        x, index = torch.max(image, dim=1)
        accuracy += torch.sum(index == label).item()/len(index)
    accuracy = accuracy/len(test_dataloader);
    print("accuracy: ", accuracy)




def fit(epochs, lr, train_dataloader, test_dataloader, opt_fun, model):
    for epoch in range(epochs):
        optimizer =opt_fun(model.parameters(), lr)
        for batch in train_dataloader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        evaluate(train_dataloader, model)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
