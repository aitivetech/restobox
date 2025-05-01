import torch
import torch.nn.functional as func

class LayerNormFunction(torch.autograd.Function):

    # noinspection PyMethodOverriding
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        # noinspection PyPep8Naming
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        # noinspection PyPep8Naming
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(torch.nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', torch.nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(torch.nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(torch.torch.nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = torch.nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True
        )

        # Simplified Channel Attention
        self.sca = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1, padding=0,
                stride=1,
                groups=1,
                bias=True
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = ffn_expand * c
        self.conv4 = torch.nn.Conv2d(
            in_channels=c,
            out_channels=ffn_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True
        )
        self.conv5 = torch.nn.Conv2d(
            in_channels=ffn_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = torch.nn.Dropout(drop_out_rate) if drop_out_rate > 0. else torch.nn.Identity()
        self.dropout2 = torch.nn.Dropout(drop_out_rate) if drop_out_rate > 0. else torch.nn.Identity()

        self.beta = torch.nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

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

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class NAFBlockSR(torch.nn.Module):
    '''
    NAFBlock for Super-Resolution
    '''
    def __init__(self, c, fusion=False, drop_out_rate=0.):
        super().__init__()
        self.blk = NAFBlock(c, drop_out_rate=drop_out_rate)

    def forward(self, feats):
        feats = self.blk(feats)
        return feats

class NAFNet(torch.nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=None, dec_blk_nums=None):
        if enc_blk_nums is None:
            enc_blk_nums = []

        if dec_blk_nums is None:
            dec_blk_nums = []

        super().__init__()

        self.intro = torch.nn.Conv2d(
            in_channels=img_channel,
            out_channels=width,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True
        )
        self.ending = torch.nn.Conv2d(
            in_channels=width,
            out_channels=img_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=1,
            bias=True
        )

        self.encoders = torch.nn.ModuleList()
        self.decoders = torch.nn.ModuleList()
        self.middle_blks = torch.nn.ModuleList()
        self.ups = torch.nn.ModuleList()
        self.downs = torch.nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                torch.nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                torch.nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            torch.nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(chan, chan * 2, 1, bias=False),
                    torch.nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                torch.nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp: torch.Tensor):
        # noinspection PyPep8Naming
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = func.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNetSR(torch.nn.Module):
    '''
    NAFNet for Super-Resolution
    '''
    def __init__(self, up_scale=4, width=48, num_blks=16, img_channel=3, drop_out_rate=0., fusion_from=-1, fusion_to=-1):
        super().__init__()
        self.intro = torch.nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.body = torch.nn.Sequential(
            *[
                NAFBlockSR(
                    width,
                    fusion=(fusion_from <= i <= fusion_to),
                    drop_out_rate=drop_out_rate
                )for i in range(num_blks)]
        )

        self.up = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=width, out_channels=img_channel * up_scale**2, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            torch.nn.PixelShuffle(up_scale)
        )

        self.up_scale = up_scale

    def forward(self, inp):
        inp_hr = torch.nn.functional.interpolate(inp, scale_factor=self.up_scale, mode='bilinear')

        feats = self.intro(inp)
        feats = self.body(feats)
        out = self.up(feats)
        out = out + inp_hr
        return out