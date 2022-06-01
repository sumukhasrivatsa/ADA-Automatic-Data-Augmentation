
from cmath import cos
import torch.nn.functional as F
from torch.distributions import Uniform
import torch.nn as nn
import torch

###for netA
class Affine(nn.Module):
  def __init__(self,image,label):
    super(Affine, self).__init__()
    self.nz=6
    self.image=image
    self.label=label
    
    self.mean = torch.tensor(0)
    self.std = torch.tensor(0)


    ########### ACTUAL NETA
    self.fc_loc = nn.Sequential(
            nn.Linear(self.nz, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 6  ),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
    self.fc1=nn.Linear(6,1)


  def get_rotation_matrix(self,noise):

     identitymatrix = torch.eye(2, 2)
     #print(identitymatrix)
     identitymatrix = identitymatrix.unsqueeze(0)
     #print(identitymatrix)
     identitymatrix = identitymatrix.repeat(noise.shape[0], 1, 1)
     #print(identitymatrix)
     theta = self.fc_loc(noise)
     theta = self.fc1(theta)


     affinematrix = identitymatrix
     inverse_matrix=identitymatrix
     affinematrix[:, 0, 0] = torch.cos(theta)
     affinematrix[:, 0, 1] = -torch.sin(theta)
     affinematrix[:, 1, 0] = torch.sin(theta)
     affinematrix[:, 1, 1] = torch.cos(cos)

     inverse_matrix[:, 0, 0] = torch.cos(-theta)
     inverse_matrix[:, 0, 1] = -torch.sin(-theta)
     inverse_matrix[:, 1, 0] = torch.sin(-theta)
     inverse_matrix[:, 1, 1] = torch.cos(-cos)
     
     return affinematrix,inverse_matrix


  def forward(self):
    if self.mean.device != self.image.device:
            self.mean = self.mean.to(self.image.device)
            self.std = self.std.to(self.image.device)
    bs = self.image.shape[0]
    self.uniform = Uniform(low=-torch.ones(bs, self.nz), high=torch.ones(bs, self.nz))
    #print(self.uniform)
    noise = self.uniform.rsample()
    #print(noise)
    # get transformation matrix
    
    self.affinematrix,self.inverse_matrix = self.get_rotation_matrix(noise)

  
    # compute transformation grid
    grids = F.affine_grid(self.affinematrix, self.image.size(), align_corners=True)
    
    # apply transformation
    x = F.grid_sample(self.image, grids, align_corners=True)
    
    
    return x,self.label


  def inverter(self,inverse_mat):
      gridd
    

    
        
                
