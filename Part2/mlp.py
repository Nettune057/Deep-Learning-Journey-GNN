import torch
import torch.nn.functional as F
from torch.nn import Linear


class MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.linear1 = Linear(dim_in, dim_h)
        self.linear2 = Linear(dim_h, dim_out)

    def forward(self,x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        
        return F.log_softmax(x,dim=1)
    
    def fit(self, data, epochs):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        