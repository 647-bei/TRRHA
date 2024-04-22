import torch.nn as nn
from torch.nn import functional as F
import torch

# 定义 RefConv 模型
class RefConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1,
                 map_k=3):
        super(RefConv, self).__init__()

        # 初始化卷积核的原始形状
        self.origin_kernel_shape = (out_channels, in_channels // groups, kernel_size, kernel_size)
        # 将原始卷积核形状注册为模型的缓冲区
        self.register_buffer('weight', torch.zeros(*self.origin_kernel_shape))
        # 计算 2D 卷积核的数量
        G = in_channels * out_channels // (groups ** 2)
        self.num_2d_kernels = out_channels * in_channels // groups
        self.kernel_size = kernel_size
        # 创建用于映射的 2D 卷积操作
        self.convmap = nn.Conv2d(in_channels=self.num_2d_kernels,
                                 out_channels=self.num_2d_kernels, kernel_size=map_k, stride=1, padding=map_k // 2,
                                 groups=G, bias=False)
        # 初始化映射卷积核的权重
        # nn.init.zeros_(self.convmap.weight)
        # 定义偏置为 None，因为在映射卷积核后没有使用
        self.bias = None  # nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        # 设置模型的其他属性
        self.stride = stride
        self.groups = groups
        # 如果没有提供填充参数，将其设置为卷积核大小的一半
        if padding is None:
            padding = kernel_size // 2
        self.padding = padding

    def forward(self, inputs):
        # 将原始卷积核重塑为 (1, num_2d_kernels, kernel_size, kernel_size) 的形状
        origin_weight = self.weight.view(1, self.num_2d_kernels, self.kernel_size, self.kernel_size)
        # 使用映射卷积核对原始卷积核进行映射，得到映射后的卷积核
        # self.weight：这是模型中原始的卷积核参数，是一个形状为 (out_channels, in_channels // groups, kernel_size, kernel_size) 的张量。
        kernel = self.weight + self.convmap(origin_weight).view(*self.origin_kernel_shape)
        # 使用映射后的卷积核进行 2D 卷积操作
        return F.conv2d(inputs, kernel, stride=self.stride, padding=self.padding, dilation=1, groups=self.groups,
                        bias=self.bias)
