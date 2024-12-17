## This code is based on the Restormer and NAFNet. Thanks for sharing !
## "Restormer: Efficient Transformer for High-Resolution Image Restoration" (2022 CVPR)
## https://arxiv.org/abs/2111.09881

## NAFNet -> "Simple Baselines for Image Restoration" (2022 ECCV)
## https://arxiv.org/abs/2204.04676


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ptflops import get_model_complexity_info


##########################################################################
# 自定义 LayerNormFunction，继承自 torch.autograd.Function
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps  # 保存 eps 值到上下文中，以便在反向传播时使用
        N, C, H, W = x.size()  # 获取输入张量 x 的形状信息
        mu = x.mean(1, keepdim=True)  # 计算每个通道的均值，保持维度不变
        var = (x - mu).pow(2).mean(1, keepdim=True)  # 计算每个通道的方差，保持维度不变
        y = (x - mu) / (var + eps).sqrt()  # Layer Normalization 公式，对输入进行归一化
        ctx.save_for_backward(y, var, weight)  # 保存 y, var, weight 供反向传播使用
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)  # 应用可学习的缩放和偏移参数
        return y  # 返回归一化后的张量 y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps  # 从上下文中获取保存的 eps 值

        N, C, H, W = grad_output.size()  # 获取梯度张量 grad_output 的形状信息
        y, var, weight = ctx.saved_variables  # 获取保存的 y, var, weight

        g = grad_output * weight.view(1, C, 1, 1)  # 计算关于输出的梯度
        mean_g = g.mean(dim=1, keepdim=True)  # 计算 g 的均值，保持维度不变

        mean_gy = (g * y).mean(dim=1, keepdim=True)  # 计算 g 与 y 的乘积的均值，保持维度不变
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)  # 计算关于输入的梯度
        # 返回输入梯度 gx，权重梯度，偏置梯度，None（因为输入的 weight 和 bias 是可学习的参数，不需要计算梯度）
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


# LayerNorm2d 类继承自 nn.Module
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        # 注册可学习的参数 weight 和 bias，并初始化为指定值
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps  # 设置 eps 为给定的值

    def forward(self, x):
        # 调用 LayerNormFunction 中的 forward 方法执行 Layer Normalization
        # 使用 self.weight 和 self.bias 作为可学习参数，eps 作为归一化操作的常数项
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


''''
这段代码定义了一个名为 SimpleGate 的简单门控（Gate）模块，
它继承自 PyTorch 的 nn.Module 类。在 forward 方法中，它执行了一个简单的门控操作，
将输入张量 x 沿着指定维度进行分割，然后将分割后的两部分相乘并返回结果。
'''


# SimpleGate 类继承自 nn.Module
class SimpleGate(nn.Module):
    def forward(self, x):
        # 将输入张量 x 沿着维度 1 进行分割，分割成两个张量 x1 和 x2
        x1, x2 = x.chunk(2, dim=1)
        # 返回两个张量相乘的结果
        return x1 * x2


## Multi-DConv Head Transposed Self-Attention (MDTA)
# Attention 类继承自 nn.Module，实现了一个多头自注意力机制
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads  # 头的数量
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))  # 用于缩放注意力分数的温度参数

        # qkv 是一个卷积层，用于将输入映射到查询（query）、键（key）和值（value）
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        # qkv_dwconv 是一个深度可分离卷积层，对 qkv 的输出进行深度方向的卷积操作
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        # project_out 是一个卷积层，用于最终输出的投影操作
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入张量 x 的形状信息

        # 对输入 x 执行查询（query）、键（key）、值（value）的卷积计算
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)  # 将 qkv 分成三个部分：q, k, v

        # 重新组织张量的维度结构，将通道维度划分成多个头，并重新排列形状
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 对查询和键执行归一化操作
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # 计算注意力得分，通过缩放温度参数进行缩放
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)  # 对得分执行 softmax 操作，得到归一化的注意力权重

        # 使用注意力权重加权求和值，得到注意力机制的输出
        out = (attn @ v)

        # 重新组织输出张量的形状结构，恢复原始形状
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # 对输出应用最终的投影操作
        out = self.project_out(out)
        return out


# TransformerBlock 类继承自 nn.Module，是 Transformer 模型中的一个基本模块
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()

        # Layer normalization 操作
        self.norm1 = LayerNorm2d(dim)  # 应用在注意力机制前的归一化
        self.attn = Attention(dim, num_heads, bias)  # 自注意力（self-attention）机制

        self.norm2 = LayerNorm2d(dim)  # 应用在前馈神经网络前的归一化
        self.sg = SimpleGate()  # 简单门控（Gate）操作

        ffn_channel = ffn_expansion_factor * dim  # 前馈神经网络通道数
        # 两个卷积层构成前馈神经网络
        self.conv_ff1 = nn.Conv2d(in_channels=dim, out_channels=int(ffn_channel), kernel_size=1, padding=0, stride=1,
                                  groups=1, bias=True)
        self.conv_ff2 = nn.Conv2d(in_channels=int(ffn_channel) // 2, out_channels=dim, kernel_size=1, padding=0,
                                  stride=1, groups=1, bias=True)

        # 可学习的参数用于调节注意力机制和前馈神经网络的重要性
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        # 通过注意力机制和归一化对输入进行处理，并加上调节参数 beta
        y = x + self.attn(self.norm1(x)) * self.beta

        # 通过前馈神经网络进行特征转换
        x = self.conv_ff1(self.norm2(y))  # 第一个卷积层
        x = self.sg(x)  # 简单门控操作
        x = self.conv_ff2(x)  # 第二个卷积层

        # 返回注意力机制和前馈神经网络结果的加权和，用 gamma 参数调节
        return y + x * self.gamma


class Localcnn_block(nn.Module):
    def __init__(self, c, DW_Expand=2, drop_out_rate=0.):
        super().__init__()

        # 计算深度卷积后的通道数
        dw_channel = c * DW_Expand

        # 第一个卷积层，1x1 卷积，用于通道扩展
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        # 第二个卷积层，3x3 深度卷积
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel, bias=True)

        # 第三个卷积层，1x1 卷积，通道缩减
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # 空间上的注意力机制
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True)
        )

        # 简单门控机制
        self.sg = SimpleGate()

        # Layer Normalization 操作
        self.norm1 = LayerNorm2d(c)

        # Dropout 操作
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # 参数 beta，用于调节输出
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        # 对输入进行 Layer Normalization
        x = self.norm1(x)

        # 进行深度卷积
        x = self.conv1(x)
        x = self.conv2(x)

        # 应用简单门控机制
        x = self.sg(x)

        # 空间注意力机制的处理
        x = x * self.sca(x)
        x = self.conv3(x)

        # Dropout 操作
        x = self.dropout1(x)

        # 融合输入和输出结果
        y = inp + x * self.beta

        return y


# Transformer and Conv Block
class ConvTransBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, split_conv_rate, bias):
        super(ConvTransBlock, self).__init__()
        self.dim = dim
        self.rate = split_conv_rate

        # 创建 TransformerBlock，用于处理转换部分的特征
        self.trans_block = TransformerBlock(int(dim - int((dim * split_conv_rate))), num_heads, ffn_expansion_factor,
                                            bias)

        # 创建两个 1x1 的卷积层
        self.conv1 = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)

        # 创建 Localcnn_block，用于处理卷积部分的特征
        # print(int(dim * split_conv_rate))
        self.conv_block = Localcnn_block(int(dim * split_conv_rate))

    def forward(self, x):
        conv_dim = int(self.dim * self.rate)

        # 对输入进行分割，一部分用于卷积处理，一部分用于转换处理
        conv_x0, trans_x = torch.split(self.conv1(x), (conv_dim, self.dim - conv_dim), dim=1)

        # 分别对转换部分和卷积部分进行处理
        # print('trans_x: ',trans_x.shape, 'conv_x0: ', conv_x0.shape)
        trans_x = self.trans_block(trans_x)
        conv_x = self.conv_block(conv_x0)
        # print('trans_x: ', trans_x.shape, 'conv_x: ', conv_x.shape)

        # 将转换处理和卷积处理的结果进行合并
        res = self.conv2(torch.cat((conv_x, trans_x), dim=1))

        # 将输入和处理后的结果相加作为最终输出
        out = x + res

        return out


## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48):
        super(OverlapPatchEmbed, self).__init__()

        # 使用 3x3 的卷积层进行图像嵌入，将输入通道数转换为指定的嵌入维度
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1),  # 3x3 卷积层
            nn.ReLU(inplace=True)  # ReLU 激活函数，inplace=True 表示原地修改，节省内存
        )

    def forward(self, x):
        # 对输入进行图像嵌入操作
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        # 下采样模块：通过卷积和像素逆混洗操作将特征图尺寸缩小
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),  # 3x3 卷积，减小通道数
            nn.PixelUnshuffle(2)  # 像素逆混洗，缩小特征图尺寸
        )

    def forward(self, x):
        return self.body(x)  # 执行下采样操作


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        # 上采样模块：通过卷积和像素混洗操作将特征图尺寸放大
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),  # 3x3 卷积，增加通道数
            nn.PixelShuffle(2)  # 像素混洗，放大特征图尺寸
        )

    def forward(self, x):
        return self.body(x)  # 执行上采样操作


# --------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)  # 计算特征维度缩减后的值

        # 自适应平均池化层用于池化特征图到固定的尺寸（1x1）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 包含一个卷积层和 PReLU 激活函数的序列，用于降低特征维度
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, d, 1, padding=0, bias=bias),  # 1x1 卷积层，降维
            nn.PReLU()
        )

        # 使用 Conv2d 层创建一个长度为 height 的模块列表，用于创建特定数量的卷积操作
        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)  # Softmax 操作用于计算注意力权重

    def forward(self, inp_feats):
        print('inp_feats', inp_feats[0].shape)
        batch_size = inp_feats[0].shape[0]  # 获取批次大小
        n_feats = inp_feats[0].shape[1]  # 获取特征数量

        inp_feats = torch.cat(inp_feats, dim=1)  # 在通道维度上拼接特征
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])  # 重塑特征形状

        feats_U = torch.sum(inp_feats, dim=1)  # 按指定维度对特征进行求和
        feats_S = self.avg_pool(feats_U)  # 对求和结果进行自适应平均池化
        feats_Z = self.conv_du(feats_S)  # 使用卷积和激活函数降维特征

        attention_vectors = [fc(feats_Z) for fc in self.fcs]  # 对降维后的特征执行一系列卷积操作
        attention_vectors = torch.cat(attention_vectors, dim=1)  # 在通道维度上拼接特征
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)  # 重塑特征形状

        attention_vectors = self.softmax(attention_vectors)  # 对卷积后的特征执行 Softmax 操作，获得注意力权重
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)  # 使用注意力权重对特征进行加权求和
        print('feats_V', feats_V.shape)  # 打印特征融合后的形状
        return feats_V  # 返回融合后的特征


# ---------- SKFF_DIBR -----------------------

class AFD_Net(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[2, 6, 8, 10, 4, 3, 2],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2,
                 rate=[0.8, 0.61, 0.4, 0.23],
                 bias=False):
        super(AFD_Net, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)  # 输入图像裁剪为重叠的图块并投影到嵌入维度

        # 编码器层级1：一系列的转换块用于特征提取
        self.encoder_level1 = nn.Sequential(
            *[ConvTransBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             split_conv_rate=rate[0], bias=bias) for i in range(num_blocks[0])])

        # 从层级1到层级2的下采样
        self.down1_2 = Downsample(dim)
        # 编码器层级2：一系列的转换块用于特征提取
        self.encoder_level2 = nn.Sequential(
            *[ConvTransBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             split_conv_rate=rate[1], bias=bias) for i in range(num_blocks[1])])

        # 从层级2到层级3的下采样
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        # 编码器层级3：一系列的转换块用于特征提取
        self.encoder_level3 = nn.Sequential(
            *[ConvTransBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             split_conv_rate=rate[2], bias=bias) for i in range(num_blocks[2])])

        # 从层级3到层级4的下采样
        self.down3_4 = Downsample(int(dim * 2 ** 2))
        # 潜在特征表示
        self.latent = nn.Sequential(*[
            ConvTransBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                           split_conv_rate=rate[3], bias=bias) for i in range(num_blocks[3])])

        # 从层级4到层级3的上采样
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            ConvTransBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                           split_conv_rate=rate[2], bias=bias) for i in range(num_blocks[4])])

        # 从层级3到层级2的上采样
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(
            *[ConvTransBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             split_conv_rate=rate[1], bias=bias) for i in range(num_blocks[5])])

        # 从层级2到层级1的上采样
        self.up2_1 = Upsample(int(dim * 2))
        self.decoder_level1 = nn.Sequential(
            *[ConvTransBlock(dim=int(dim * 2), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             split_conv_rate=rate[0], bias=bias) for i in range(num_blocks[6])])

        ####### SKFF #########
        depth = 4

        self.u4_ = nn.Sequential(nn.Conv2d(8 * dim, dim, 3, 1, 1, bias=bias),
                                 nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False))

        self.u3_ = nn.Sequential(nn.Conv2d(4 * dim, dim, 3, 1, 1, bias=bias),
                                 nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False))
        self.u2_ = nn.Sequential(nn.Conv2d(2 * dim, dim, 3, 1, 1, bias=bias),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.u1_ = nn.Conv2d(dim * 2, dim, 3, 1, 1, bias=bias)

        self.final_ff = SKFF(in_channels=dim, height=depth)

        self.last = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=True)

    def forward(self, inp_img):
        # 编码器层级1：重叠图块嵌入 + 特征提取
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        # 编码器层级2：从层级1到层级2的下采样 + 特征提取
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        # 编码器层级3：从层级2到层级3的下采样 + 特征提取
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        # 编码器层级4：从层级3到层级4的下采样 + 特征提取
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        # 解码器
        u4_ = self.u4_(latent)

        # 层级3解码
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        u3_ = self.u3_(out_dec_level3)

        # 层级2解码
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        u2_ = self.u2_(out_dec_level2)

        # 层级1解码
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        u1_ = self.u1_(out_dec_level1)

        skff_in = [u4_, u3_, u2_, u1_]
        skff_out = self.final_ff(skff_in)  # Selective Kernel Feature Fusion

        output = self.last(skff_out) + inp_img  # 输出 + 输入图像
        return output


if __name__ == "__main__":
    model = AFD_Net()
    # print(model)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    print(out.shape)
    # flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    # print('flops: ', flops, 'params: ', params)
