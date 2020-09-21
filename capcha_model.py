import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class CapchaRecognitionModel(nn.Module):
    def __init__(self,output_dim):
        super(CapchaRecognitionModel,self).__init__()

        self.conv1 = nn.Conv2d(3,128,kernel_size=(6,3),padding=(1,1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(128,64,kernel_size=(6,3),padding=(1,1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.linear1 = nn.Linear(768,64)
        self.drop = nn.Dropout(0.2)

        self.gru = nn.GRU(64,32,num_layers=2,bidirectional=True,dropout=0.25,batch_first=True)

        self.linear2 = nn.Linear(64,output_dim+1)

    def forward(self,images,targets=None):
        bs,_,_,_ = images.shape
        x = F.relu(self.conv1(images))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)


        x = x.permute(0,2,1,3)
        x = x.reshape(bs,x.shape[1],-1)

        x = self.drop(x)
        x = F.relu(self.linear1(x))

        x,_ = self.gru(x)

        x = self.linear2(x)
        x = x.permute(1,0,2)
        if targets is not None:
            x = F.log_softmax(x,2)

            input_len = torch.full(size=(bs,),fill_value=x.size(0),dtype=torch.int32)
            target_len = torch.full(size=(bs,),fill_value=targets.size(1),dtype=torch.int32)

            ctc_loss = nn.CTCLoss()

            loss = ctc_loss(x,targets,input_len,target_len)

            return x,loss




        return x


if __name__ == '__main__':
    model = CapchaRecognitionModel(75)

    x,y = model(torch.ones((1,3,200,50)),torch.ones((1,5)))

    print(x.shape,y)