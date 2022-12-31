import torch.nn as nn
import torch.nn.functional as F


class LocationMLP(nn.Module):
   def linear(self, in_size, out_size):
      linear = nn.Linear(in_size, out_size)
      nn.init.xavier_uniform_(linear.weight)
      return linear
   
   def bnorm(self, size):
      return nn.BatchNorm1d(size)
   
   def relu(self, size):
      return nn.ReLU(size)
   
   def layer(self, in_size, out_size, x, last_layer=False):
      x = self.linear(in_size, out_size)(x)
      x = self.bnorm(out_size)(x)
      if not last_layer:
         return x
      x = self.relu(out_size)(x)
      x = self.dropout(x)
      return x

   def __init__(self, dim_in, dim_out, dropout_probability):
      super(LocationMLP, self).__init__()
      self.dim_in = dim_in
      self.dim_out = dim_out
      self.dropout = nn.Dropout(dropout_probability)

   def forward(self, x):
      x = self.layer(self.dim_in, 1024, x)
      x = self.layer(1024, 512, x)
      x = self.layer(512, 256, x)
      x = self.layer(256, 128, x)
      x = self.layer(128, self.dim_out, x, last_layer=True)
      return F.softmax(x, dim=-1)
