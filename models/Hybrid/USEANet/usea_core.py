import torch
import torch.nn as nn
import torch.nn.functional as F

from .pvtv2 import pvt_v2_b0
from .moe import PhysicsMoE

# =============================================================================
# 轻量化核心模块 (inspired by efficient network designs)
# =============================================================================

class DepthwiseConv(nn.Module):
    """深度可分离卷积 - 轻量化核心组件"""
    def __init__(self, dim=768):
        super(DepthwiseConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class LayerNorm2D(nn.Module):
    """2D LayerNorm for efficient computation"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class LightweightConv2d(nn.Module):
    """轻量级卷积模块 - 深度可分离卷积"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(LightweightConv2d, self).__init__()
        
        # 对于1x1卷积，直接使用常规卷积
        if kernel_size == 1:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, 
                                stride=stride, padding=0, bias=False)
        else:
            # 对于大于1x1的卷积，使用深度可分离卷积
            self.conv = nn.Sequential(
                # 深度卷积
                nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, 
                         stride=stride, padding=padding, dilation=dilation,
                         groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.ReLU(inplace=True),
                # 点卷积
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
            )
        
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

# =============================================================================
# 超声图像专用处理模块 - 本工作的核心创新
# =============================================================================

class UltrasoundNoiseReduction(nn.Module):
    """超声图像斑点噪声抑制模块"""
    def __init__(self, in_channels, out_channels):
        super(UltrasoundNoiseReduction, self).__init__()
        self.conv1 = LightweightConv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = LightweightConv2d(out_channels, out_channels, 3, padding=1)
        self.conv3 = LightweightConv2d(out_channels, out_channels, 3, padding=1)
        
        # 如果输入输出通道数不同，需要投影层
        if in_channels != out_channels:
            self.projection = LightweightConv2d(in_channels, out_channels, 1)
        else:
            self.projection = None
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # 多尺度去噪
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        
        # 残差连接
        if self.projection is not None:
            x_res = self.projection(x)
        else:
            x_res = x
            
        return x_res + x3

class EdgeEnhancement(nn.Module):
    """超声图像边缘增强模块"""
    def __init__(self, in_channels, out_channels):
        super(EdgeEnhancement, self).__init__()
        self.conv1 = LightweightConv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = LightweightConv2d(out_channels, out_channels, 3, padding=1)
        
        # 使用分组卷积进行边缘检测
        self.edge_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1, 
                                  groups=out_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        # 初始化边缘检测核
        edge_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32)
        
        # 为每个通道设置相同的边缘检测核
        weight = edge_kernel.unsqueeze(0).unsqueeze(0).repeat(out_channels, 1, 1, 1)
        self.edge_conv.weight.data = weight
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        edge = self.edge_conv(x)
        # 边缘增强
        return x + 0.1 * edge

class ContrastEnhancement(nn.Module):
    """超声图像对比度增强模块"""
    def __init__(self, in_channels, out_channels):
        super(ContrastEnhancement, self).__init__()
        self.conv1 = LightweightConv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = LightweightConv2d(out_channels, out_channels, 3, padding=1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        # 通道注意力增强对比度
        att = self.attention(x)
        return x * att

class MultiBranchFeatureProcessor(nn.Module):
    """多分支特征处理器 - 针对超声图像的专用优化"""
    def __init__(self, in_channel, out_channel):
        super(MultiBranchFeatureProcessor, self).__init__()
        self.relu = nn.ReLU(True)
        
        # 超声图像专用分支
        self.noise_branch = UltrasoundNoiseReduction(in_channel, out_channel)
        self.edge_branch = EdgeEnhancement(in_channel, out_channel)
        self.contrast_branch = ContrastEnhancement(in_channel, out_channel)
        
        # 特征融合
        self.conv_cat = LightweightConv2d(3*out_channel, out_channel, 3, padding=1)
        self.conv_res = LightweightConv2d(in_channel, out_channel, 1)
        
        # 自适应权重学习
        self.weight_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3*out_channel, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):    
        # 超声图像专用分支处理
        x_noise = self.noise_branch(x)
        x_edge = self.edge_branch(x)
        x_contrast = self.contrast_branch(x)
        
        # 特征拼接
        x_cat = torch.cat((x_noise, x_edge, x_contrast), 1)
        
        # 自适应权重融合
        weights = self.weight_conv(x_cat)
        channels_per_branch = x_cat.size(1) // 3
        
        weighted_features = []
        for i in range(3):
            start_idx = i * channels_per_branch
            end_idx = (i + 1) * channels_per_branch
            weighted_features.append(
                x_cat[:, start_idx:end_idx] * weights[:, i:i+1]
            )
        x_weighted = torch.cat(weighted_features, dim=1)
        
        # 最终融合
        x_fused = self.conv_cat(x_weighted)
        x_res = self.conv_res(x)
        
        return self.relu(x_fused + x_res)

# =============================================================================
# 高效注意力机制
# =============================================================================

class EfficientChannelAttention(nn.Module):
    """高效通道注意力机制"""
    def __init__(self, in_channels, reduction=8):
        super(EfficientChannelAttention, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 轻量级通道注意力
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        avg_out = self.channel_attention(self.avg_pool(x))
        max_out = self.channel_attention(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        x = x * spatial_att
        
        return x

class EdgeAwareAttention(nn.Module):
    """边缘感知注意力模块"""
    def __init__(self, in_channels):
        super(EdgeAwareAttention, self).__init__()
        # 使用深度可分离卷积进行边缘检测
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, 
                     groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False)
        )
        
        # 轻量级注意力机制
        reduction = max(4, in_channels // 16)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        edge_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32)
        
        # 为每个通道设置相同的边缘检测核
        weight = edge_kernel.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        self.edge_conv[0].weight.data = weight
        
    def forward(self, x):
        edge = self.edge_conv(x)
        att = self.attention(x)
        return x * (1 + 0.1 * edge) * att

# =============================================================================
# 特征聚合模块
# =============================================================================

class HierarchicalFeatureAggregation(nn.Module):
    """分层特征聚合模块"""
    def __init__(self, channel, num_class):
        super(HierarchicalFeatureAggregation, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 改进的上采样卷积
        self.conv_upsample1 = LightweightConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = LightweightConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = LightweightConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = LightweightConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = LightweightConv2d(2*channel, 2*channel, 3, padding=1)
        
        # 注意力机制
        self.attention2 = EdgeAwareAttention(2*channel)
        self.attention3 = EdgeAwareAttention(3*channel)
        
        self.conv_concat2 = LightweightConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = LightweightConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = LightweightConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5_fg = nn.Conv2d(3*channel, num_class, 1)
        self.conv5_bg = nn.Conv2d(3*channel, num_class, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x1_up = self.upsample(x1)
        x1_up_up = self.upsample(x1_up)
        x2_up = self.upsample(x2)
        x2_1 = self.conv_upsample1(x1_up) * x2
        x3_1 = self.conv_upsample2(x1_up_up) \
               * self.conv_upsample3(x2_up) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(x1_up)), 1)
        x2_2 = self.attention2(self.conv_concat2(x2_2))

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.attention3(self.conv_concat3(x3_2))

        x = self.conv4(x3_2)
        x_fg = self.conv5_fg(x)
        x_bg = self.conv5_bg(x)

        return x_fg, x_bg

# =============================================================================
# 主网络架构
# =============================================================================

class USEANet(nn.Module):
    """Ultrasound Segmentation Enhancement Attention Network (USEANet)
    
    A next-generation lightweight network built around multi-branch feature processing.
    Inspired by modern efficient design principles while specifically optimized 
    for ultrasound image segmentation tasks.
    """
    
    def __init__(self, channel=32, num_classes=2, sem_downsample=1, use_softmax=True, 
                 pretrained_model_path=None):
        super(USEANet, self).__init__()
        self.num_classes = num_classes
        self.sem_downsample = sem_downsample
        self.use_softmax = use_softmax

        if pretrained_model_path is not None:
            pretrained_model_path = f'{pretrained_model_path}/pvt_v2_b0.pth'
        
        # Lightweight Backbone - PVT-B0
        self.backbone = pvt_v2_b0()
        if pretrained_model_path is not None:
            try:
                save_model = torch.load(pretrained_model_path)
                model_dict = self.backbone.state_dict()
                state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
                model_dict.update(state_dict)
                self.backbone.load_state_dict(model_dict)
                print(f"Successfully loaded PVT-B0 pretrained weights from {pretrained_model_path}")
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")
        
        # 多分支特征处理器 - 核心创新
        # PVT-B0: [32, 64, 160, 256]
        # x2 keeps the plain fused branch; x3/x4 use the ultrasound-physics MoE
        # (design §3.3: avoid multi-layer CNN MoE variance, MoE only on semantic layers).
        self.feature_processor_2 = MultiBranchFeatureProcessor(64, channel)
        self.feature_processor_3 = PhysicsMoE(160, channel)
        self.feature_processor_4 = PhysicsMoE(256, channel)
        
        # 分层特征聚合
        self.feature_aggregator = HierarchicalFeatureAggregation(channel, self.num_classes)
        
        # 轻量化解码器 - 大幅减少通道数
        mid_channel_4 = 64   # 极度轻量化
        mid_channel_3 = 24    
        mid_channel_2 = 24
        
        # 第4层解码器
        self.decoder4_conv1 = LightweightConv2d(256, mid_channel_4, kernel_size=1)
        self.decoder4_conv2 = LightweightConv2d(mid_channel_4, mid_channel_4, kernel_size=5, padding=2)
        self.decoder4_conv3 = LightweightConv2d(mid_channel_4, mid_channel_4, kernel_size=5, padding=2)
        self.decoder4_conv4 = LightweightConv2d(mid_channel_4, mid_channel_4, kernel_size=5, padding=2)
        self.decoder4_conv5_fg = LightweightConv2d(mid_channel_4, num_classes, kernel_size=1)
        self.decoder4_conv5_bg = LightweightConv2d(mid_channel_4, num_classes, kernel_size=1)
        
        # 第3层解码器
        self.decoder3_conv1 = LightweightConv2d(160, mid_channel_3, kernel_size=1)
        self.decoder3_conv2 = LightweightConv2d(mid_channel_3, mid_channel_3, kernel_size=3, padding=1)
        self.decoder3_conv3 = LightweightConv2d(mid_channel_3, mid_channel_3, kernel_size=3, padding=1)
        self.decoder3_conv4_fg = LightweightConv2d(mid_channel_3, num_classes, kernel_size=3, padding=1)
        self.decoder3_conv4_bg = LightweightConv2d(mid_channel_3, num_classes, kernel_size=3, padding=1)
        
        # 第2层解码器
        self.decoder2_conv1 = LightweightConv2d(64, mid_channel_2, kernel_size=1)
        self.decoder2_conv2 = LightweightConv2d(mid_channel_2, mid_channel_2, kernel_size=3, padding=1)
        self.decoder2_conv3 = LightweightConv2d(mid_channel_2, mid_channel_2, kernel_size=3, padding=1)
        self.decoder2_conv4_fg = LightweightConv2d(mid_channel_2, num_classes, kernel_size=3, padding=1)
        self.decoder2_conv4_bg = LightweightConv2d(mid_channel_2, num_classes, kernel_size=3, padding=1)
        
        # 高效注意力模块
        self.decoder4_attention = EfficientChannelAttention(mid_channel_4, reduction=4)
        self.decoder3_attention = EfficientChannelAttention(mid_channel_3, reduction=4) 
        self.decoder2_attention = EfficientChannelAttention(mid_channel_2, reduction=4)

    def forward(self, x, segSize=None):
        # 编码器阶段 - 轻量化backbone
        x1, x2, x3, x4 = self.backbone(x)
        
        # 多分支特征处理
        x2_processed = self.feature_processor_2(x2)
        x3_processed = self.feature_processor_3(x3)
        x4_processed = self.feature_processor_4(x4)
        
        # 特征聚合
        agg_feat_fg, agg_feat_bg = self.feature_aggregator(x4_processed, x3_processed, x2_processed)
        lateral_map_5_fg = F.interpolate(agg_feat_fg, scale_factor=8/self.sem_downsample, mode='bilinear')
        lateral_map_5_bg = F.interpolate(agg_feat_bg, scale_factor=8/self.sem_downsample, mode='bilinear')
        
        # 轻量化解码器 - 第4层
        crop_4_fg = F.interpolate(agg_feat_fg, scale_factor=0.25, mode='bilinear')
        crop_4_bg = F.interpolate(agg_feat_bg, scale_factor=0.25, mode='bilinear')
        
        x = self.decoder4_conv1(x4)
        x = F.relu(self.decoder4_conv2(x))
        x = F.relu(self.decoder4_conv3(x))
        x = F.relu(self.decoder4_conv4(x))
        
        # 应用高效注意力
        x = self.decoder4_attention(x)
        
        decoder4_feat_fg = self.decoder4_conv5_fg(x)
        decoder4_feat_bg = self.decoder4_conv5_bg(x)
        
        if self.use_softmax:
            decoder4_feat_fg = decoder4_feat_fg + decoder4_feat_fg.mul(F.softmax(crop_4_fg-crop_4_bg, dim=1))
        else:
            decoder4_feat_fg = decoder4_feat_fg + decoder4_feat_fg.mul(crop_4_fg-crop_4_bg)
        
        lateral_map_4_fg = F.interpolate(decoder4_feat_fg, scale_factor=32/self.sem_downsample, mode='bilinear')
        lateral_map_4_bg = F.interpolate(decoder4_feat_bg, scale_factor=32/self.sem_downsample, mode='bilinear')
        
        # 轻量化解码器 - 第3层
        crop_3_fg = F.interpolate(decoder4_feat_fg, scale_factor=2, mode='bilinear')
        crop_3_bg = F.interpolate(decoder4_feat_bg, scale_factor=2, mode='bilinear')
        
        x = self.decoder3_conv1(x3)
        x = F.relu(self.decoder3_conv2(x))
        x = F.relu(self.decoder3_conv3(x))
        
        x = self.decoder3_attention(x)
        
        decoder3_feat_fg = self.decoder3_conv4_fg(x)
        decoder3_feat_bg = self.decoder3_conv4_bg(x)
        
        if self.use_softmax:
            decoder3_feat_fg = decoder3_feat_fg + decoder3_feat_fg.mul(F.softmax(crop_3_fg-crop_3_bg, dim=1))
        else:
            decoder3_feat_fg = decoder3_feat_fg + decoder3_feat_fg.mul(crop_3_fg-crop_3_bg)
        
        lateral_map_3_fg = F.interpolate(decoder3_feat_fg, scale_factor=16/self.sem_downsample, mode='bilinear')
        lateral_map_3_bg = F.interpolate(decoder3_feat_bg, scale_factor=16/self.sem_downsample, mode='bilinear')
        
        # 轻量化解码器 - 第2层
        crop_2_fg = F.interpolate(decoder3_feat_fg, scale_factor=2, mode='bilinear')
        crop_2_bg = F.interpolate(decoder3_feat_bg, scale_factor=2, mode='bilinear')
        
        x = self.decoder2_conv1(x2)
        x = F.relu(self.decoder2_conv2(x))
        x = F.relu(self.decoder2_conv3(x))
        
        x = self.decoder2_attention(x)
        
        decoder2_feat_fg = self.decoder2_conv4_fg(x)
        decoder2_feat_bg = self.decoder2_conv4_bg(x)
        
        if self.use_softmax:
            decoder2_feat_fg = decoder2_feat_fg + decoder2_feat_fg.mul(F.softmax(crop_2_fg-crop_2_bg, dim=1))
        else:
            decoder2_feat_fg = decoder2_feat_fg + decoder2_feat_fg.mul(crop_2_fg-crop_2_bg)
        
        lateral_map_2_fg = F.interpolate(decoder2_feat_fg, scale_factor=8/self.sem_downsample, mode='bilinear')
        lateral_map_2_bg = F.interpolate(decoder2_feat_bg, scale_factor=8/self.sem_downsample, mode='bilinear')
        
        return (lateral_map_2_fg, lateral_map_3_fg, lateral_map_4_fg, lateral_map_5_fg,
                lateral_map_2_bg, lateral_map_3_bg, lateral_map_4_bg, lateral_map_5_bg)


def create_useanet(**kwargs):
    """创建USEANet模型的工厂函数"""
    return USEANet(**kwargs)