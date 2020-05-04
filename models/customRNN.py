import torch.nn as nn
import torch

class ImageRNN(nn.Module):
    def __init__(self):
        super(ImageRNN,self).__init__()
        self.rnn = nn.LSTM(
            input_size = 28,
            hidden_size = 64,
            num_layers = 150,#number of hidden layer
            batch_first = True,
        )
        self.out = nn.Linear(64,62)

    def forward(self,x):
        r_out,(h_n,h_c) = self.rnn(x,None)  #x(batch,time_step,input_size)
        #h_n,h_c are hidden state of the previous
        out = self.out(r_out[:,-1,:])#find the output of the latest time
        #self.out (batch,time step,input)
        return out