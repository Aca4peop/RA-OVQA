from typing import Optional, Type
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TempAttnPooling(nn.Module):
    def __init__(self,dim_in,dim_emb,dim_out):
        super().__init__()
        self.q=nn.Parameter(torch.randn(1,1,dim_emb))
        self.k=nn.Conv1d(dim_in, dim_emb,3,padding=0,stride=1)
        self.v=nn.Conv1d(dim_in, dim_emb,3,padding=0,stride=1)
        self.projection=nn.Linear(dim_emb,dim_out)
    def forward(self,x):
        q=self.q.expand(x.shape[0],-1,-1)
        k=self.k(x.transpose(-1,-2)).transpose(-1,-2)
        v=self.v(x.transpose(-1,-2)).transpose(-1,-2)
        x= F.softmax(q@k.transpose(-1,-2),dim=-1)@v 
        return self.projection(x)


class CrossAttn(nn.Module):
    def __init__(
            self,
            dim: int,
            out_dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k= nn.Linear(dim, dim, bias=qkv_bias)
        self.v= nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.head_dim, self.head_dim, bias=proj_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        # self.proj = nn.Linear(dim, out_dim, bias=proj_bias)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            dis: torch.Tensor,
            refs: torch.Tensor,

    ) -> torch.Tensor:
        B, N, C = dis.shape
        assert N==1
        B, N, C = refs.shape
        q=self.q(dis).reshape(B,1,self.num_heads, self.head_dim).permute( 0, 2, 1, 3)
        k=self.k(refs).reshape(B, N,self.num_heads, self.head_dim).permute( 0, 2, 1, 3)
        v_ref=self.v(refs).reshape(B, N,self.num_heads, self.head_dim).permute( 0, 2, 1, 3)
        v_dis = self.v(dis).reshape(B, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v=self.v_proj(F.gelu(v_dis-v_ref))
        q, k = self.q_norm(q), self.k_norm(k)
        #atten
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B,  C)
        x = self.norm(x)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        return x

class CrossAttnPro(nn.Module):
    def __init__(
            self,
            q_dim: int,
            kv_dim: int,
            dim: int,
            out_dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dim=dim

        self.q = nn.Linear(q_dim, dim, bias=qkv_bias)
        self.k= nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.v= nn.Linear(kv_dim, dim, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        # self.proj = nn.Linear(dim, out_dim, bias=proj_bias)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(
            self,
            dis: torch.Tensor,
            refs: torch.Tensor,

    ) -> torch.Tensor:
        B, N, C0 = dis.shape
        assert N==1
        B, N, C = refs.shape
        q=self.q(dis).reshape(B,1,self.num_heads, self.head_dim).permute( 0, 2, 1, 3)
        k=self.k(refs).reshape(B, N,self.num_heads, self.head_dim).permute( 0, 2, 1, 3)
        v=self.v(refs).reshape(B, N,self.num_heads, self.head_dim).permute( 0, 2, 1, 3)
        q, k = self.q_norm(q), self.k_norm(k)
        #atten
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, 1,self.dim )
        # x=F.softmax(x,dim=-1)*dis
        x = self.norm(x)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        return x

class PerceptionDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attn1 = CrossAttn(dim=128, out_dim=128,num_heads=2, qkv_bias=False, qk_norm=False, proj_bias=False)
        self.cross_attn2 = CrossAttn(dim=256, out_dim=128, num_heads=4, qkv_bias=False, qk_norm=False, proj_bias=False)
        self.cross_attn3 = CrossAttn(dim=512, out_dim=128, num_heads=8, qkv_bias=False, qk_norm=False, proj_bias=False)
        self.cross_attn4 = CrossAttn(dim=1024, out_dim=128, num_heads=8, qkv_bias=False, qk_norm=False, proj_bias=False)
        self.temp_proj1 = nn.Conv1d(128, 128,3,padding=0,stride=1)
        self.temp_proj2 = nn.Conv1d(256, 128, 3, padding=0, stride=1)
        self.temp_proj3 = nn.Conv1d(512, 128, 3, padding=0, stride=1)
        self.temp_proj4 = nn.Conv1d(1024, 128, 3, padding=0, stride=1)
        self.proj = nn.Linear(128,1)


    def forward(self,dis,refs):
        #dis B,T,C    refs B,T,N,C
        B,T,N,C=refs.shape
        refs=refs.reshape(B*T,N,C)
        dis=dis.reshape(B*T,1,C)
        x1=self.cross_attn1(dis[:,:,:128],refs[:,:,:128])
        x2=self.cross_attn2(dis[:,:,128:128+256],refs[:,:,128:128+256])
        x3=self.cross_attn3(dis[:,:,128+256:128+256+512],refs[:,:,128+256:128+256+512])
        x4=self.cross_attn4(dis[:,:,-1024:],refs[:,:,-1024:])
        x1=self.temp_proj1(x1.view(B,T,-1).transpose(1,2)).transpose(1,2)
        x1=F.leaky_relu(x1)
        x2=self.temp_proj2(x2.view(B,T,-1).transpose(1,2)).transpose(1,2)
        x2=F.leaky_relu(x2)
        x3=self.temp_proj3(x3.view(B,T,-1).transpose(1,2)).transpose(1,2)
        x3=F.leaky_relu(x3)
        x4=self.temp_proj4(x4.view(B,T,-1).transpose(1,2)).transpose(1,2)
        x4=F.leaky_relu(x4)
        x=x1+x2+x3+x4
        x=F.dropout(x,0.2)
        x=self.proj(x) #B,T,1
        return x

class DistortionEnhce(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = CrossAttnPro(q_dim=128,kv_dim=640,dim=128 ,num_heads=2,out_dim=256, qkv_bias=False,
                                  qk_norm=False, proj_bias=False)
        self.attn2 = CrossAttnPro(q_dim=256, kv_dim=640, dim=256, num_heads=4, out_dim=256, qkv_bias=False,
                                  qk_norm=False, proj_bias=False)
        self.attn3 = CrossAttnPro(q_dim=512, kv_dim=640, dim=512, num_heads=8, out_dim=256, qkv_bias=False,
                                  qk_norm=False, proj_bias=False)
        self.attn4 = CrossAttnPro(q_dim=1024, kv_dim=640, dim=1024, num_heads=8, out_dim=256, qkv_bias=False,
                                  qk_norm=False, proj_bias=False)
        tmp=np.load('models/dis_text_features.npy')
        self.text_features=torch.from_numpy(tmp).float()
        self.temp_proj1 = nn.Conv1d(128, 128, 3, padding=0, stride=1)
        self.temp_proj2 = nn.Conv1d(256, 128, 3, padding=0, stride=1)
        self.temp_proj3 = nn.Conv1d(512, 128, 3, padding=0, stride=1)
        self.temp_proj4 = nn.Conv1d(1024, 128, 3, padding=0, stride=1)
        self.proj = nn.Linear(128, 1)

    def forward(self,x):
        B,T,C=x.shape
        x=x.view(B*T,1,C)
        text_features=self.text_features.unsqueeze(0).expand(B*T,13,640).to(x.device)
        x1=self.attn1(x[:,:,:128],text_features)
        x2 = self.attn2(x[:, :, 128:128+256], text_features)
        x3 = self.attn3(x[:, :, 128+256:128+256+512], text_features)
        x4 = self.attn4(x[:, :, -1024:], text_features)
        x1 = self.temp_proj1(x1.view(B, T, -1).transpose(1, 2)).transpose(1, 2)
        x1 = F.leaky_relu(x1)
        x2 = self.temp_proj2(x2.view(B, T, -1).transpose(1, 2)).transpose(1, 2)
        x2 = F.leaky_relu(x2)
        x3 = self.temp_proj3(x3.view(B, T, -1).transpose(1, 2)).transpose(1, 2)
        x3 = F.leaky_relu(x3)
        x4 = self.temp_proj4(x4.view(B, T, -1).transpose(1, 2)).transpose(1, 2)
        x4 = F.leaky_relu(x4)
        x = x1 + x2 + x3 + x4
        x = F.dropout(x, 0.2)
        x = self.proj(x)  # B,T,1
        return x



class GatedMoM(nn.Module):
    def __init__(self):
        super().__init__()
        self.distortion=DistortionEnhce()
        self.perception=PerceptionDiff()
        self.q1=nn.Linear(1920,640)
        self.q2 = nn.Linear(1920, 640)
        self.k_percp=nn.Linear(1920,640)
        self.k_dist=nn.Linear(640,640)
        self.proj=nn.Linear(64,1)

    def forward(self,x,refs):

        B,T,C=x.shape
        q1=self.q1(x).unsqueeze(2)
        q1=q1*0.4 # sacle q
        q2=self.q2(x).unsqueeze(2)
        q2=q2*0.4
        #refs B T N C
        k_percp=self.k_percp(refs)
        a_percp=q1@k_percp.transpose(-2,-1)
        text_features=self.distortion.text_features.unsqueeze(0).unsqueeze(0).expand(B,T,13,640).to(q2.device)
        k_dist=self.k_dist(text_features)
        a_dist=q2@k_dist.transpose(-2,-1)
        a=torch.cat([a_percp.mean(dim=-1),a_dist.mean(dim=-1)],dim=-1)[:,1:-1,:]

        s1=self.perception(x,refs)

        s2=self.distortion(x)
        s=torch.cat([s1,s2],dim=-1)
        # if training:
        #     a=a+torch.randn_like(a)*0.2
        a=F.softmax(a,dim=-1)
        s=(s*a).sum(dim=-1)
        return s.mean(1).view(-1),s1.mean(1).view(-1),s2.mean(1).view(-1),a










