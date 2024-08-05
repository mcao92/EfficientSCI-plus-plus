from torch import nn 
import torch 
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
from .builder import MODELS

class TimesAttention3D(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
      
        self.qkv = nn.Linear(dim, (dim//2) * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim//2, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        C = C//2
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class FFN(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.p1 = nn.Sequential(
            nn.Conv3d(dim,dim*2,1),
            nn.LeakyReLU(inplace=True)
        )
        self.f1 = nn.Sequential(
            nn.Conv3d(dim,dim,3,1,1),
            nn.LeakyReLU(inplace=True)
        )
        self.f2 = nn.Sequential(
            nn.Conv3d(dim,dim,3,1,1),
            nn.LeakyReLU(inplace=True)
        )
        self.mlp = nn.Conv3d(dim*2,dim,1)
    def forward(self,x):
        x = self.p1(x)
        x1,x2 = torch.chunk(x,chunks=2,dim=1)
        f1_out = self.f1(x1)
        f2_out = self.f2(x2*F.gelu(f1_out))
        x = torch.cat([f1_out,f2_out],dim=1)
        out = self.mlp(F.leaky_relu(x))
        return out
    
class CFormerBlock(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.scb = nn.Sequential(
            nn.Conv3d(dim, dim, (1,3,3), padding=(0,1,1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim, (1,3,3), padding=(0,1,1)),
        )
        self.tsab = TimesAttention3D(dim,num_heads=4)
        self.ffn = FFN(dim)

    def forward(self,x):
        _,_,_,h,w = x.shape
        scb_out = self.scb(x)
        tsab_in = einops.rearrange(x,"b c d h w->(b h w) d c")
        tsab_out = self.tsab(tsab_in)
        tsab_out = einops.rearrange(tsab_out,"(b h w) d c->b c d h w",h =h,w=w)
        ffn_in = scb_out+tsab_out+x
        ffn_out = self.ffn(ffn_in)+ffn_in
        return ffn_out

class ResDNetBlock(nn.Module):
    def __init__(self,dim,group_num):
        super().__init__()
        self.cformer_list = nn.ModuleList()
        self.group_num = group_num
        group_dim = dim//group_num
        for i in range(group_num):
            self.cformer_list.append(CFormerBlock(group_dim))
            
        self.last_conv = nn.Conv3d(dim,dim,1)

    def forward(self, x):
        input_list = torch.chunk(x,chunks=self.group_num,dim=1)
        cf_in = input_list[0]
        out_list = []
        cf_out = self.cformer_list[0](cf_in)
        out_list.append(cf_out)
        for i in range(1,self.group_num):
            cf_in = input_list[i]
            cf_out = self.cformer_list[i](cf_in+cf_out)
            out_list.append(cf_out)
        out = torch.cat(out_list,dim=1)
        out = self.last_conv(out)
        out = x + out
        return out

class StageBlock(nn.Module):
    def __init__(self, dim, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        group_num=dim//64
        for i in range(depth):
            self.blocks.append(
                ResDNetBlock(dim,group_num=group_num)
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        out = x 
        return out

@MODELS.register_module 
class EfficientSCI_plus_plus(nn.Module):
    def __init__(self,dim=128,stage_num=1,depth_num=[2,1],color_ch=1):
        super().__init__()
        self.color_ch = color_ch
        self.conv_first = nn.Sequential(
            nn.Conv3d(1, dim, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(dim, dim*2, kernel_size=(1,3,3), stride=(1,2,2),padding=(0,1,1)),
            nn.LeakyReLU(inplace=True),
        )
        dim *= 2
        self.encoder = nn.ModuleList()
        for i in range(stage_num):
            self.encoder.append(
                nn.ModuleList([
                    StageBlock(
                        dim=dim, depth=depth_num[i]),
                    nn.Sequential(
                        nn.Conv3d(dim,dim,kernel_size=(1,3,3), stride=1,padding=(0,1,1)),
                        nn.LeakyReLU(inplace=True)
                    )
                                  
                ])
            )

        self.bottleneck = StageBlock(dim=dim,depth=depth_num[-1])

        self.decoder = nn.ModuleList()
        for i in range(stage_num):
            self.decoder.append(
                nn.ModuleList([
                    nn.Sequential(
                        nn.Conv3d(dim,dim,1),
                    ),
                    nn.Conv3d(dim*2, dim, 1, 1),
                    StageBlock(
                        dim=dim, depth=depth_num[stage_num - 1 - i]
                    ),
                ])
            )

        self.conv_last = nn.Sequential(
            nn.Conv3d(dim*2,dim*2,1),
            Rearrange("b c t h w-> b t c h w"),
            nn.PixelShuffle(2),
            Rearrange("b t c h w-> b c t h w"),
            nn.Conv3d(dim//2, color_ch, kernel_size=3, stride=1, padding=1),
        )

    def bayer_init(self,y,Phi,Phi_s):
        bayer = [[0,0], [0,1], [1,0], [1,1]]
        b,f,h,w = Phi.shape
        y_bayer = torch.zeros(b,h//2,w//2,4).to(y.device)
        Phi_bayer = torch.zeros(b,f,h//2,w//2,4).to(y.device)
        Phi_s_bayer = torch.zeros(b,h//2,w//2,4).to(y.device)
        for ib in range(len(bayer)):
            ba = bayer[ib]
            y_bayer[...,ib] = y[:,ba[0]::2,ba[1]::2]
            Phi_bayer[...,ib] = Phi[:,:,ba[0]::2,ba[1]::2]
            Phi_s_bayer[...,ib] = Phi_s[:,ba[0]::2,ba[1]::2]
        y_bayer = einops.rearrange(y_bayer,"b h w ba->(b ba) h w")
        Phi_bayer = einops.rearrange(Phi_bayer,"b f h w ba->(b ba) f h w")
        Phi_s_bayer = einops.rearrange(Phi_s_bayer,"b h w ba->(b ba) h w")

        meas_re = torch.div(y_bayer, Phi_s_bayer)
        meas_re = torch.unsqueeze(meas_re, 1)
        maskt = Phi_bayer.mul(meas_re)
        x = meas_re + maskt
        x = einops.rearrange(x,"(b ba) f h w->b f h w ba",b=b)

        x_bayer = torch.zeros(b,f,h,w).to(y.device)
        for ib in range(len(bayer)): 
            ba = bayer[ib]
            x_bayer[:,:,ba[0]::2, ba[1]::2] = x[...,ib]
        x = x_bayer.unsqueeze(1)
        return x
    def forward(self, y,Phi,Phi_s):
        out_list = []
        if self.color_ch==3:
            x = self.bayer_init(y,Phi,Phi_s)
        else:
            meas_re = torch.div(y, Phi_s)
            # meas_re = torch.unsqueeze(meas_re, 1)
            maskt = Phi.mul(meas_re)
            x = meas_re + maskt
            x = x.unsqueeze(1)
    
        f_x = self.conv_first(x)
        x = f_x

        fea_list = []
        for stage_block, Downsample in self.encoder:
            x = stage_block(x)
            fea_list.append(x)
            x = Downsample(x)
        x = self.bottleneck(x)

        for i, [Upsample, Fusion, stage_block] in enumerate(self.decoder):
            x = Upsample(x)
            x = Fusion(torch.cat([x, fea_list.pop()], dim=1))
            x = stage_block(x)
        out = self.conv_last(torch.cat([x,f_x],dim=1))

        if self.color_ch!=3:
            out = out.squeeze(1)
        out_list.append(out)
        return out_list
