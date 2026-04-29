

from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from torchvision import transforms


my_trans = None
def init():
    global my_trans
    my_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.6).float())  # Binarization
    ])

# binarization of MNIST dataset



def getTrainData():
    train_dataset = MNIST(root='./MNIST', train=True, download=False, transform=my_trans)
    train_data = [(img.view(1, 28*28).numpy(), label) for img, label in train_dataset]
    return train_data


def getTestData():
    test_dataset = MNIST(root='./MNIST', train=False, download=False, transform=my_trans)
    test_data = [(img.view(1, 28*28).numpy(), label) for img, label in test_dataset][:1000]
    return test_data