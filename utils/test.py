import torch
import torch.nn.functional as F
import numpy as np

def test(model, device, test_loader, IfRNN, accuracy_list, confusdata, confustarget):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if IfRNN: # if training method == 1, Default RNN Method
                data = torch.squeeze(data)
            data, target = data.to(device), target.to(device)
            output = model(data)
            confusdata = np.append(confusdata, torch.max(output, 1)[1].data.numpy())
            confustarget = np.append(confustarget, target.numpy())
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy_list.append(100. * correct / len(test_loader.dataset)) # accuracy recorder

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return confusdata, confustarget