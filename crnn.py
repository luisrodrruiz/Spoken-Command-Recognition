import torch.nn as nn


# Simple CNN model

class CRNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential()
        self.cnn.append(nn.Conv2d(1,32,[4,4]))
        self.cnn.append(nn.ReLU())
        self.cnn.append(nn.MaxPool2d([2,2],[2,2]))
        self.cnn.append(nn.Conv2d(32,32,[5,5]))
        self.cnn.append(nn.ReLU())
        self.cnn.append(nn.Conv2d(32,32,[7,7]))
        self.cnn.append(nn.ReLU())
        self.cnn.append(nn.Conv2d(32,1,[1,1]))
        self.cnn.append(nn.ReLU())
        self.gru_num_layers = 3
        self.gru_hidden_size = 128

    def add_gru(self,input_size, n_classes = 10):

        self.rnn = nn.GRU(input_size,self.gru_hidden_size,num_layers = self.gru_num_layers)
        self.final_layer = nn.Linear(self.gru_hidden_size,n_classes)
            

    def forward(self,x):
        x = x.unsqueeze(1)
        cnn_out = self.cnn(x).squeeze(1)
        rnn_out = self.rnn(cnn_out)[0]
        return self.final_layer(rnn_out[:,-1])
        
        return out

