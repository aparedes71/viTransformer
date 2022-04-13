import torch
import torch.nn as nn
import torch.functional as F
from einops import repeat,rearrange
from einops.layers.torch import Rearrange, Reduce
import numpy as np


# class Image2TiledVec(nn.Module):
#     def __init__(self,patch_size = 16,stride = 16):
#         super().__init__()
#         self.patch_size = patch_size
#         self.stride = stride
#     def forward(self,x):
#         if len(x.shape) == 4:
#             x = x.unfold(2,self.patch_size,self.stride).unfold(3,self.patch_size,self.stride)
#             return x.reshape(x.shape[0],-1,self.patch_size,self.patch_size)
#         elif len(x.shape) == 3:
#             x = x.unfold(1,self.patch_size,self.stride).unfold(2,self.patch_size,self.stride)
#         else:
#             x = x.unfold(0,self.patch_size,self.stride).unfold(1,self.patch_size,self.stride)
#         return x.reshape(-1,self.patch_size,self.patch_size)


# class FlattenTiledVec(nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self,x):
        
  
# toy examples to make sure tiling is correct
# x = torch.arange(0,16)
# x = x.view(4,4)
# print(x)

# x1 = torch.stack([x,x,x],0)
# print(x1.shape)
# print(x1)

# x2 = torch.stack([x1,x1,x1],0)
# print(x2.shape)
# print(x2)


# tiler = Image2TiledVec(2,2)

# x3 = tiler(x)
# print(x3.shape)
# print(x3)

# x4 = tiler(x1)
# print(x4.shape)
# print(x4)


# x5 = tiler(x2)
# print(x5.shape)
# print(x5)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 3, patch_size = 16, emb_size = 768, img_size = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))

        
    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x
    

x = torch.arange(0,16).view(4,4)
x = torch.stack([x,x,x])
x = x.unsqueeze(0)
x = x.type(torch.LongTensor)
print(x.shape)

print(PatchEmbedding(3,2)(x).shape)