#from __future__ import print_function
import torch
import torch.optim as optim
#import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models.conv import Net
from models.rnn_conv import ImageRNN
from models.customCNN import CustomCNN
from models.customRNN import CustomRNN
from models.inceptionv4 import Inceptionv4
from models.inception_resnet_v2 import Inception_ResNetv2
#import torch.nn.functional as F
#import matplotlib.pyplot as plt
import numpy as np
#from PIL import Image

from utils.dataloader import dataloader
from utils.train import train_cnn, train_rnn, train_MyCNN, train_MyRNN
from utils.test import test
from utils.imgproc import imsave, resultplot

# For GPU Training Library
# Ref: https://github.com/xinntao/ESRGAN/issues/94#issuecomment-612903759
import ctypes

def main():
    epoches = 1
    gamma = 0.1
    log_interval = 10
    torch.manual_seed(1)
    save_model = True
    TRAIN_SIZE = 6000 # If using Inception Method, due to RAM limitation, size should be small
    TEST_SIZE = 2000

    num_workers = 2

    loss_list = list()  
    accuracy_list = list()
    confusdata = np.array([], dtype=np.uint8)
    confustarget = np.array([], dtype=np.uint8)
    # Tranin Method [1: DefaultRNN, 2: CustomCNN, 3: DefaultCNN, 4: CustomRNN, 5: inceptionv4, 6: inception_resnetv2]
    Train_Method = 2
    ResizeIMG = False # Default: False, Will turn on when using Inception Methods
    IfRNN = False # Default: False, Will turn on when using RNN Methods
    
    TEST_ONLY = True # Test Mode, Load saved data and test, need to change Train_Method to matched NN as well
    ExistedModelPath = "./results/CustomCNN_15Epoches_1500Size.pt"

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')
    else:
        device = torch.device("cpu")

    if Train_Method == 1:
        IfRNN = True
        model = ImageRNN().to(device)
        Model_label = 'DefaultRNN'
    elif Train_Method == 2:
        model = CustomCNN().to(device)
        Model_label = 'CustomCNN'
    elif Train_Method == 3:
        model = Net().to(device)
        Model_label = 'DefaultCNN'
    elif Train_Method == 4:
        IfRNN = True
        model = CustomRNN().to(device)
        Model_label = 'CustomRNN'
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

    savepath = './results/' + Model_label + '_' + str(epoches) + 'Epoches_' + str(TRAIN_SIZE) + 'Size.pt'

    train_loader, test_loader = dataloader(TRAIN_SIZE, TEST_SIZE, num_workers, './datasets', 'byclass', ResizeIMG, use_cuda)

    {imsave(train_loader, Model_label)} if TRAIN_SIZE <= 100 else {}

    if TEST_ONLY:
        model.load_state_dict(torch.load(ExistedModelPath))
        confusdata, confustarget = test(model, device, test_loader, IfRNN, accuracy_list, confusdata, confustarget)
        accuracy_list = [0,0]
        loss_list = [0,0]
        savepath = ExistedModelPath[:-3] + "_TestOnly.TT"
        resultplot(Model_label, loss_list, accuracy_list, confusdata, confustarget, savepath)
        return


    for epoch in range(1, epoches + 1):
        if Train_Method == 1:
            train_rnn(log_interval, model, device, train_loader, optimizer, epoch, loss_list)
        elif Train_Method == 2:
            train_MyCNN(log_interval, model, device, train_loader, optimizer, epoch, loss_list)
        elif Train_Method == 3:
            train_cnn(log_interval, model, device, train_loader, optimizer, epoch, loss_list)
        elif Train_Method == 4:
            train_MyRNN(log_interval, model, device, train_loader, optimizer, epoch, loss_list)
        elif Train_Method == 5:
            train_MyCNN(log_interval, model, device, train_loader, optimizer, epoch, loss_list)
        elif Train_Method == 6:
            train_MyCNN(log_interval, model, device, train_loader, optimizer, epoch, loss_list)

        confusdata, confustarget = test(model, device, test_loader, IfRNN, accuracy_list, confusdata, confustarget)
        scheduler.step()

    {torch.save(model.state_dict(), savepath)} if save_model else {}

    resultplot(Model_label, loss_list, accuracy_list, confusdata, confustarget, savepath)

if __name__ == '__main__':
    main()