import torch.nn as nn
import torch.nn.functional as F


class LocationMLP(nn.Module):
   def __init__(self, dim_in, dim_out, dropout_probability):
      super(LocationMLP, self).__init__()
      self.dim_in = dim_in[0]
      self.dim_out = dim_out
      print(self.dim_in)
      self.first_block = nn.Sequential(nn.Linear(self.dim_in, 1024), nn.BatchNorm1d(1024),
       nn.ReLU(1024), nn.Dropout(dropout_probability))
      self.second_block = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512),
       nn.ReLU(512), nn.Dropout(dropout_probability))
      self.third_block = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256),
       nn.ReLU(256), nn.Dropout(dropout_probability))
      self.fourth_block = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128),
       nn.ReLU(128), nn.Dropout(dropout_probability))
      self.final_block = nn.Sequential(nn.Linear(128, self.dim_out), nn.BatchNorm1d(self.dim_out))

   def forward(self, x):
      print(x.shape)
      x = self.first_block(x)
      x = self.second_block(x)
      x = self.third_block(x)
      x = self.fourth_block(x)
      x = self.final_block(x)
      return x

class DefenseMLP(nn.Module):
   def __init__(self, dim_in, dim_out):
      super(LocationMLP, self).__init__()
      self.dim_in = dim_in[0]
      self.dim_out = dim_out
      self.first_block = nn.Sequential(nn.Linear(self.dim_in, 256), nn.BatchNorm1d(256),
      nn.ReLU(256))
      self.second_block = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128),
      nn.ReLU(128))
      self.final_block = nn.Sequential(nn.Linear(128, self.dim_out), nn.BatchNorm1d(self.dim_out))
   
   def forward(self, x):
      x = self.first_block(x)
      x = self.second_block(x)
      x = self.final_block(x)
      return x
