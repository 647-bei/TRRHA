import basicblock as B
import torch.nn as nn
from torch.nn import functional as F
import torch
from RefConv import RefConv
from ptflops import get_model_complexity_info


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = RefConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.PReLU(num_parameters=1, init=0.25) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



class EFFM(nn.Module):
    def __init__(self, channels, factor=32):
        super(EFFM, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.highpool = nn.AdaptiveAvgPool2d((None, 1))
        self.widthpool = nn.AdaptiveAvgPool2d((1, None))

        self.msc = RC(in_channels=channels // self.groups,
                      ch3x3red=channels // self.groups,
                      ch5x5red=channels // self.groups)

        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        # 可学习的参数用于调节注意力机制和前馈神经网络的重要性
        self.beta = nn.Parameter(torch.zeros((1, 64, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, 64, 1, 1)), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.size()
        # group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w

        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        # print('group_x : ',x.shape,group_x.shape)
        x_h = self.highpool(group_x)
        x_w = self.widthpool(group_x).permute(0, 1, 3, 2)

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        x2 = self.msc(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class PReLUConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kerner_size, padding):
        super(PReLUConv2d, self).__init__()
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kerner_size,
                               padding=padding)

    def forward(self, x):
        out = self.conv1(x)
        out = self.prelu(out)
        return out


class DilatedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kerner_size, padding):
        super(DilatedConvolution, self).__init__()
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kerner_size,
                               padding=padding, dilation=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.prelu(out)
        return out


class RC(nn.Module):
    def __init__(self, in_channels, ch3x3red, ch5x5red):
        super(RC, self).__init__()
        self.branch1 = PReLUConv2d(in_channels, ch3x3red, kerner_size=3, padding=1)
        self.branch2 = DilatedConvolution(in_channels, ch5x5red, kerner_size=3, padding=2)
        self.conv128to64 = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1)

    def forward(self, x):
        inputx = x
        branch1 = self.branch1(x)
        branch1 = branch1 + inputx

        branch2 = self.branch2(x)
        branch2 = branch2 + inputx

        output1 = [branch1, branch2]
        out = torch.cat(output1, 1)
        out = self.conv128to64(out)
        return out


class RRAM(nn.Module):
    def __init__(self, in_channal):
        super(RRAM, self).__init__()
        self.msc = RC(in_channels=in_channal, ch3x3red=in_channal, ch5x5red=in_channal)
        self.spatial_attention = RAM(in_channal)
        self.attconv = nn.Conv2d(in_channels=in_channal * 2, out_channels=in_channal, kernel_size=3, padding=1,
                                 stride=1)

    def forward(self, x):
        afb1 = self.msc(x)
        sa1 = self.spatial_attention(afb1)
        sa1 = torch.cat([sa1, x], 1)
        sa1 = self.attconv(sa1)
        return sa1


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.PReLU(num_parameters=1, init=0.25)
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        out = avg_out + max_out

        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = BasicConv(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        # print(self.sigmoid(x).shape)
        return self.sigmoid(x)


class RAM(nn.Module):
    def __init__(self, in_channal):
        super(RAM, self).__init__()
        self.ca = ChannelAttention(in_channal)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x

        return x


class RB(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(RB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3,
                               stride=stride, padding=1)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3,
                               stride=stride, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.prelu(out)
        out = self.conv2(out)
        out = x + out

        return out


class GMR(nn.Module):
    def __init__(self):
        super(GMR, self).__init__()
        self.conv12to64 = nn.Conv2d(in_channels=12, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.prelu = nn.PReLU(num_parameters=1, init=0.25)
        self.bn1 = nn.BatchNorm2d(64)
        self.unshuff = nn.PixelUnshuffle(2)
        self.shuff = nn.PixelShuffle(2)
        self.RAM = RAM(64)
        self.relu = RB(in_channel=64, out_channel=64)
        # self.refconv128to64 = RefConv(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv64to256 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        unshuffx = self.unshuff(x)
        conv12to64x = self.conv12to64(unshuffx)

        RAM1 = self.RAM(conv12to64x)
        Relu1 = self.relu(RAM1 + conv12to64x)
        Relu2 = self.relu(Relu1)
        Relu3 = self.relu(Relu2)
        Relu4 = self.relu(Relu3)
        Relu5 = self.relu(Relu4)
        Relu6 = self.relu(Relu5)
        Relu7 = self.relu(Relu6)
        Relu8 = self.relu(Relu7)
        Relu9 = self.relu(Relu8)
        Relu10 = self.relu(Relu9)
        Relu11 = self.relu(Relu10)
        Relu12 = self.relu(Relu11)

        RAM2 = self.RAM(Relu12)
        out = RAM2 + Relu12
        out = self.conv64to256(out)
        shuffx = self.shuff(out)

        return shuffx


class TRRHA(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=1, act_mode='R', downsample_mode='strideconv',
                 upsample_mode='convtranspose'):
        super(TRRHA, self).__init__()

        down_nonlocal = B.NonLocalBlock2D(nc[2], kernel_size=1, stride=1, padding=0, bias=True, act_mode='B',
                                          downsample=False, downsample_mode='strideconv')
        up_nonlocal = B.NonLocalBlock2D(nc[2], kernel_size=1, stride=1, padding=0, bias=True, act_mode='B',
                                        downsample=False, downsample_mode='strideconv')

        self.m_head = B.conv(in_nc, nc[0], mode='C' + act_mode[-1])

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.conv(nc[0], nc[0], mode='C' + act_mode) for _ in range(nb)],
                                    downsample_block(nc[0], nc[1], mode='2' + act_mode))
        self.m_down2 = B.sequential(*[B.conv(nc[1], nc[1], mode='C' + act_mode) for _ in range(nb)],
                                    downsample_block(nc[1], nc[2], mode='2' + act_mode))
        self.m_down3 = B.sequential(down_nonlocal, *[B.conv(nc[2], nc[2], mode='C' + act_mode) for _ in range(nb)],
                                    downsample_block(nc[2], nc[3], mode='2' + act_mode))
        self.m_body = B.sequential(*[B.conv(nc[3], nc[3], mode='C' + act_mode) for _ in range(nb + 1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], mode='2' + act_mode),
                                  *[B.conv(nc[2], nc[2], mode='C' + act_mode) for _ in range(nb)], up_nonlocal)
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], mode='2' + act_mode),
                                  *[B.conv(nc[1], nc[1], mode='C' + act_mode) for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], mode='2' + act_mode),
                                  *[B.conv(nc[0], nc[0], mode='C' + act_mode) for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, mode='C')

        self.RRAM1 = RRAM(64)
        self.RRAM2 = RRAM(128)
        self.RRAM3 = RRAM(256)
        self.RRAM4 = RRAM(512)
        self.RRAM5 = RRAM(512)
        self.RRAM6 = RRAM(256)
        self.RRAM7 = RRAM(128)
        self.RRAM8 = RRAM(64)

        self.GMR = GMR()

        self.conv192to64 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.m_up3conv = nn.Conv2d(in_channels=512 * 3, out_channels=512, kernel_size=3, padding=1, stride=1)

        self.m_up2conv = nn.Conv2d(in_channels=256 * 3, out_channels=256, kernel_size=3, padding=1, stride=1)
        self.m_up1conv = nn.Conv2d(in_channels=128 * 3, out_channels=128, kernel_size=3, padding=1, stride=1)

        self.EFFM192 = EFFM(192)
        self.EFFM64 = EFFM(64)

    def forward(self, x0):  # x0:Tensor:(1,3,64,64)

        GMR1 = self.GMR(x0)

        x1 = self.m_head(x0)  # x1:Tensor:(1,64,64,64)
        attion1x = self.RRAM1(x1)
        x2 = self.m_down1(attion1x)  # x2:Tensor:(1,128,32,32)
        attion2x = self.RRAM2(x2)
        x3 = self.m_down2(attion2x)  # x3:Tensor:(1,256,16,16)
        attion3x = self.RRAM3(x3)
        x4 = self.m_down3(attion3x)  # x4:Tensor:(1,512,8,8)
        attion4x = self.RRAM4(x4)
        x = self.m_body(attion4x)  # x:Tensor:(1,512,8,8)

        up3convx = self.m_up3conv(torch.cat([x, x4, attion4x], 1))
        attion5x = self.RRAM5(up3convx)

        x = self.m_up3(attion5x)  # x3:Tensor:(1,256,16,16)
        up2convx = self.m_up2conv(torch.cat([x, x3, attion3x], 1))  # x2:Tensor:(1,128,32,32)
        attion6x = self.RRAM6(up2convx)

        x = self.m_up2(attion6x)  # x2:Tensor:(1,128,32,32)
        up1convx = self.m_up1conv(torch.cat([x, x2, attion2x], 1))  # x1:Tensor:(1,64,64,64)
        attion7x = self.RRAM7(up1convx)

        x = self.m_up1(attion7x)  # x1:Tensor:(1,64,64,64)
        attion8x = self.RRAM8(x)

        outputlayer = [x1, attion8x, GMR1]
        out = torch.cat(outputlayer, 1)
        out = self.EFFM192(out)
        out = self.conv192to64(out)
        out = self.EFFM64(out + x1)
        # print('out.shape', out.shape)
        x = self.m_tail(out) + x0  # x0:Tensor:(1,3,64,64)
        return x


if __name__ == '__main__':
    x = torch.rand(1, 3, 64, 64)
    model = TRRHA(3)
    # print(model)
    y = model(x)

    flops, params = get_model_complexity_info(model, (3, 64, 64), as_strings=True, print_per_layer_stat=True)
    print('flops: ', flops, 'params: ', params)
    print(y.shape)
