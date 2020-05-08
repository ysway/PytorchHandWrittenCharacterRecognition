import torch
import torch.nn.functional as F

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
        data = data.view(-1, 28, 28) # reshape data to (batch, time_step, input_size)
        outputs = model(data)
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(outputs, target)
        loss.backward(); optimizer.step()

        loss_list.append(loss.data.to(torch.device('cpu')).numpy()) #loss recorder

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))