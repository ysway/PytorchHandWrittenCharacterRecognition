import torch.nn as nn
import torch

class CustomRNN(nn.Module):
    def __init__(self):
        super(CustomRNN,self).__init__()
        self.hidden_size = 392 # neurons 
        self.rnn = nn.LSTM(
            input_size = 28, # col of image
            hidden_size = self.hidden_size,
            num_layers = 6, # number of hidden layer
            batch_first = True,
        )
        self.out = nn.Linear(self.hidden_size, 62)

    def forward(self,x):
        r_out,(h_n,h_c) = self.rnn(x,None)  #x(batch,time_step,input_size)
        #h_n,h_c are hidden state of the previous
        out = self.out(r_out[:,-1,:])#find the output of the latest time
        #self.out (batch,time step,input)
        return out
