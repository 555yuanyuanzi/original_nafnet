# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base
from basicsr.models.GDPM import GlobalDirectionalPriorModulation
from basicsr.models.afpm_refiner import AFPMFrequencyRefiner
from einops import rearrange

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., use_fft=False):
        super().__init__()
        self.use_fft = use_fft
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        
        self.patch_size = 8

        self.sg = SimpleGate()
        self.ffn_sg = SimpleGate()
        
        ffn_channel = FFN_Expand * c
        
        self.ffn_conv1 = nn.Conv2d(c, ffn_channel, 1, bias=True)  # 原始升维
        # 可学习频域缩放系数
        if self.use_fft:
            self.fft_weight = nn.Parameter(torch.ones((1, ffn_channel, 1, 1, 8, 8//2+1)))
        self.ffn_conv2 = nn.Conv2d(ffn_channel//2, c, 1, bias=True)  # 原始降维
        
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    
    def apply_fft(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        if pad_h != 0 or pad_w != 0:
            # Ensure height/width are divisible by patch size for FFT blocks.
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        # 分块
        x = rearrange(x, 'b c (h p1) (w p2) -> b c h w p1 p2', p1=P, p2=P)
        # FFT
        x_fft = torch.fft.rfft2(x.float())
        # 可学习频域加权
        x_fft = x_fft * self.fft_weight
        # 逆FFT
        x = torch.fft.irfft2(x_fft, s=(P, P))
        # 还原
        x = rearrange(x, 'b c h w p1 p2 -> b c (h p1) (w p2)', p1=P, p2=P)
        if pad_h != 0 or pad_w != 0:
            x = x[:, :, :H, :W]
        return x



    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        
        if self.use_fft:
            x = self.ffn_conv1(x) 
            x = self.apply_fft(x)   
            x = self.ffn_sg(x)          
            x = self.ffn_conv2(x)
        else:
            x = self.conv4(x)
            x = self.sg(x)
            x = self.conv5(x)
        
        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(
        self,
        img_channel=3,
        width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
        use_gdpm=False,
        gdpm_kwargs=None,
        use_fft=False,
        fft_stages=None,
        use_afpm_refiner=False,
        afpm_refiner_kwargs=None,
        afpm_refiner_stages=None,
    ):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.gdpm = None
        if use_gdpm:
            gdpm_kwargs = {} if gdpm_kwargs is None else gdpm_kwargs
            self.gdpm = GlobalDirectionalPriorModulation(
                feat_channels=width,
                in_channels=img_channel,
                **gdpm_kwargs,
            )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        fft_stages = set(fft_stages or [])
        afpm_refiner_stages = set(afpm_refiner_stages or [])
        afpm_refiner_kwargs = {} if afpm_refiner_kwargs is None else dict(afpm_refiner_kwargs)
        afpm_refiner_stage_kwargs = afpm_refiner_kwargs.pop('stage_kwargs', {})
        if use_afpm_refiner:
            invalid_stages = []
            for stage_name in afpm_refiner_stages:
                if not stage_name.startswith('dec'):
                    invalid_stages.append(stage_name)
                    continue
                try:
                    dec_idx = int(stage_name[3:])
                except ValueError:
                    invalid_stages.append(stage_name)
                    continue
                if dec_idx < 1 or dec_idx > len(dec_blk_nums):
                    invalid_stages.append(stage_name)
            if invalid_stages:
                raise ValueError(
                    "afpm_refiner_stages only supports valid decoder stage names, "
                    f"got {invalid_stages}."
                )

        self.afpm_refiner_modules = nn.ModuleDict()
        def build_block(channels, stage_name):
            return NAFBlock(
                channels,
                use_fft=use_fft and stage_name in fft_stages,
            )

        def register_afpm_refiner(channels, stage_name):
            if use_afpm_refiner and stage_name in afpm_refiner_stages:
                kwargs = dict(afpm_refiner_kwargs)
                kwargs.update(afpm_refiner_stage_kwargs.get(stage_name, {}))
                self.afpm_refiner_modules[stage_name] = AFPMFrequencyRefiner(
                    channels=channels,
                    **kwargs,
                )
        chan = width
        for stage_idx, num in enumerate(enc_blk_nums, start=1):
            stage_name = f'enc{stage_idx}'
            self.encoders.append(
                nn.Sequential(
                    *[build_block(chan, stage_name) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[build_block(chan, 'middle') for _ in range(middle_blk_num)]
            )

        for stage_idx, num in enumerate(dec_blk_nums, start=1):
            stage_name = f'dec{stage_idx}'
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[build_block(chan, stage_name) for _ in range(num)]
                )
            )
            register_afpm_refiner(chan, stage_name)

        self.padder_size = 2 ** len(self.encoders)

    def _apply_afpm_refiner(self, stage_name, x):
        if stage_name not in self.afpm_refiner_modules:
            return x
        return self.afpm_refiner_modules[stage_name](x)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        if self.gdpm is not None:
            x = self.gdpm(inp, x)

        encs = []

        for stage_idx, (encoder, down) in enumerate(zip(self.encoders, self.downs), start=1):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for stage_idx, (decoder, up, enc_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1]), start=1):
            stage_name = f'dec{stage_idx}'
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            x = self._apply_afpm_refiner(stage_name, x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    net = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
