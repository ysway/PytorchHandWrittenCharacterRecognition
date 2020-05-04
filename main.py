from __future__ import print_function
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models.conv import Net
from models.rnn_conv import ImageRNN
from models.customCNN import CustomCNN
from models.inceptionv4 import Inceptionv4
from models.inception_resnet_v2 import Inception_ResNetv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# For GPU Training Library
# Ref: https://github.com/xinntao/ESRGAN/issues/94#issuecomment-612903759
import ctypes

# functions to show an image
def imsave(img, Model_label):
    npimg = img.numpy()
    npimg = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
    im = Image.fromarray(npimg)
    filename = './results/' + Model_label + '_Test_IMG.jpeg'
    im.save(filename)

def train_cnn(log_interval, model, device, train_loader, optimizer, epoch, loss_list):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward(); optimizer.step()

        loss_list.append(loss.data.to(torch.device('cpu')).numpy()) #loss recorder

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def train_rnn(log_interval, model, device, train_loader, optimizer, epoch, loss_list):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # reset hidden states
        model.hidden = model.init_hidden()
        data = data.view(-1, 28, 28)
        outputs = model(data)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, target)
        loss.backward(); optimizer.step()

        loss_list.append(loss.data.to(torch.device('cpu')).numpy()) #loss recorder

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def train_MyCNN(log_interval, model, device, train_loader, optimizer, epoch, loss_list):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = model(data)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.data.to(torch.device('cpu')).numpy()) #loss recorder

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def train_MyRNN(log_interval, model, device, train_loader, optimizer, epoch, loss_list): # Waiting for K. W to finish her code
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # reset hidden states
        model.hidden = model.init_hidden()
        data = data.view(-1, 28, 28)
        outputs = model(data)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, target)
        loss.backward(); optimizer.step()

        loss_list.append(loss.data.to(torch.device('cpu')).numpy()) #loss recorder

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, IfRNN, accuracy_list):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if IfRNN == 1 or IfRNN == 4: # if training method == 1, AKA RNN Method
                data = torch.squeeze(data)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy_list.append(100. * correct / len(test_loader.dataset)) # accuracy recorder

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    epoches = 2
    gamma = 0.1
    log_interval = 10
    torch.manual_seed(1)
    save_model = True
    TRAIN_SIZE = 7000 # If using Inception Method, due to RAM limitation, size should be small
    TEST_SIZE = 2000
    loss_list = list()  
    accuracy_list = list()

    # Tranin Method [1: DefaultRNN, 2: CustomCNN, 3: DefaultCNN, 4: CustomRNN, 5: inceptionv4, 6: inception_resnetv2]
    Train_Method = 2
    ResizeIMG = False # Default: False, Will turn on when using Inception Methods

    # RNN Configuration
    N_STEPS = 28
    N_INPUTS = 28
    N_NEURONS = 150
    N_OUTPUTS = 62 # 62 classes in EMNIST, MNIST contains 10 classes

    # Check whether you can use Cuda
    use_cuda = torch.cuda.is_available()
    # Use Cuda if you can
    if use_cuda:
        device = torch.device("cuda")
        ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')
    else:
        device = torch.device("cpu")


    #######################    Build your network   ############################
    if Train_Method == 1:
        model = ImageRNN(64, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS, device).to(device)
        Model_label = 'DefaultRNN'
    elif Train_Method == 2:
        model = CustomCNN().to(device)
        Model_label = 'CustomCNN'
    elif Train_Method == 3:
        model = Net().to(device)
        Model_label = 'DefaultCNN'
    elif Train_Method == 4: # Waiting for K. W to finish her code
        model = Net().to(device)
        Model_label = 'DefaultCNN'
    elif Train_Method == 5:
        ResizeIMG = True
        model = Inceptionv4().to(device)
        Model_label = 'Inceptionv4'
    elif Train_Method == 6:
        ResizeIMG = True
        model = Inception_ResNetv2().to(device)
        Model_label = 'Inception_ResNetv2'

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    ######################3   Torchvision    ###########################3
    # Use data predefined loader
    # Pre-processing by using the transform.Compose
    # divide into batches
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {'num_workers': 2}
    if ResizeIMG:
        train_loader = torch.utils.data.DataLoader(
            datasets.EMNIST('./datasets', 'byclass', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Resize((299, 299), interpolation=Image.BICUBIC), # Resize Image to match min requirement of Inception
                            transforms.ToTensor()
                        ])),
            batch_size=TRAIN_SIZE, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.EMNIST('./datasets', 'byclass', train=False, transform=transforms.Compose([
                            transforms.Resize((299, 299), interpolation=Image.BICUBIC),
                            transforms.ToTensor()
                        ])),
            batch_size=TEST_SIZE, shuffle=True, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
            datasets.EMNIST('./datasets', 'byclass', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ])),
            batch_size=TRAIN_SIZE, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.EMNIST('./datasets', 'byclass', train=False, transform=transforms.Compose([
                            transforms.ToTensor()
                        ])),
            batch_size=TEST_SIZE, shuffle=True, **kwargs)

    '''
    # get some random training images
    dataiter = iter(train_loader)
    if TRAIN_SIZE <= 100:
        images, labels = dataiter.next()
        img = torchvision.utils.make_grid(images)
    else:
        for i in rage(100):
            images, labels = dataiter[i]
            img = torchvision.utils.make_grid(images)
    imsave(img, Model_label)
    '''
    

    #######################    Run your network   ############################
    for epoch in range(1, epoches + 1):
        if Train_Method == 1:
            train_rnn(log_interval, model, device, train_loader, optimizer, epoch, loss_list)
        elif Train_Method == 2:
            train_MyCNN(log_interval, model, device, train_loader, optimizer, epoch, loss_list)
        elif Train_Method == 3:
            train_cnn(log_interval, model, device, train_loader, optimizer, epoch, loss_list)
        elif Train_Method == 4: # Wating for K. W's code
            train_MyRNN(log_interval, model, device, train_loader, optimizer, epoch, loss_list)
        elif Train_Method == 5:
            train_MyCNN(log_interval, model, device, train_loader, optimizer, epoch, loss_list)
        elif Train_Method == 6:
            train_MyCNN(log_interval, model, device, train_loader, optimizer, epoch, loss_list)

        test(model, device, test_loader, Train_Method, accuracy_list)
        scheduler.step()

    if save_model:
        namestr = './results/' + Model_label + '_' + str(epoches) + 'Epoches_' + str(TRAIN_SIZE) + 'Size.pt'
        torch.save(model.state_dict(), namestr)

    plt.figure(Model_label, figsize=(12, 6))
    plt.subplot(121)
    plt.title('Training Loss')
    plt.plot(loss_list)
    plt.xlabel('Batch Training')
    plt.ylabel('Loss')
    plt.ylim(0, 3)
    
    plt.subplot(122)
    plt.title('Accuracy')
    plt.plot(accuracy_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.savefig(namestr[:-2]+"png")
    plt.show()

if __name__ == '__main__':
    main()
