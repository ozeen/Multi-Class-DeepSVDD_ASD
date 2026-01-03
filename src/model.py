import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)

Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 128, 2, 2],
    [4, 128, 2, 2],
    [4, 128, 2, 2],
]

class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super(TgramNet, self).__init__()
        # if "center=True" of stft, padding = win_len / 2
        self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                # 313(10) , 63(2), 126(4)
                nn.LayerNorm(313),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for idx in range(num_layer)])

    def forward(self, x):
        out = self.conv_extrctor(x)
        out = self.conv_encoder(out)
        return out




# MobileFaceNet BackBone
class MobileFaceNet(nn.Module):
    def __init__(self,
                 num_class,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 in_channels = 1
                 ):
        super(MobileFaceNet, self).__init__()

        self.conv1 = ConvBlock(in_channels, 64, 3, 2, 1)

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)
        # 20(10), 4(2), 8(4)
        self.linear7 = ConvBlock(512, 512, (8, 20), 1, 0, dw=True, linear=True)
        # self.linear7 = ConvBlock(512, 512, (4, 10), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        self.fc_out = nn.Linear(128, num_class)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        feature = x.view(x.size(0), -1)
        out = self.fc_out(feature)
        return out, feature


# Multi-Class SVDD
class ClassVDD_Tgram(nn.Module):
    """
    Backbone: MobileNetV2 (首层改 1 通道) -> GAP -> Linear(z_dim) -> Activation -> Linear(num_classes)
    - SVDD 使用激活后的中间特征 z
    - 分类使用最后线性层输出的 logits
    """

    def __init__(
            self,
            num_classes,
            device,
            z_dim: int = 128,
            eps_center: float = 0.01,
            leak_slope: float = 0.2,
            in_ch: int = 1,  # 输入通道，默认为 1
            mobilenet_width_mult: float = 1.0,
            bottleneck_setting = Mobilefacenet_bottleneck_setting,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.device = device
        self.eps_center = eps_center
        self.leak_slope = leak_slope

        # ===== MobileNetV2 backbone=====
        self.mobilefacenet = MobileFaceNet(num_class=num_classes+z_dim,
                                           bottleneck_setting=bottleneck_setting,
                                           in_channels=2)
        # ===== TGram Feature Extractor =====
        self.tgramnet = TgramNet(mel_bins=128, win_len=1024, hop_len=512)
       
        self.register_buffer('c', torch.zeros(num_classes, z_dim))

        # Soft Edge
        self.nu = 0.1  # 0.05~0.2

        # Trainable Radius for each class
        self.R_raw = nn.Parameter(torch.full((num_classes,), -10.0))



    def forward(self, x_mel: torch.Tensor, x_wav: torch.Tensor,return_logits: bool = False):
        """
        x: (B,1,H,W)
        return:
          logits: [B, num_classes]
          z: 激活后的中间特征，shape [B, z_dim]
        """
        x_t = self.tgramnet(x_wav).unsqueeze(1)
        x = torch.cat((x_mel, x_t), dim=1)
      
       
        out, feature = self.mobilefacenet(x)

        # latent for svdd and logits for classification
        z = out[:, :self.z_dim]
        logits = out[:, self.z_dim:]
        return logits, z

    # ---------------------- initialize center ----------------------
    def set_c(self, dataloader):
        """
        随机初始化每类中心 c[k]，与标签无关；不扫描数据。
        仍保留原函数名与调用方式，方便兼容你现有训练代码。
        """
        c = torch.randn(self.num_classes, self.z_dim).to(self.device)
        # 可选：行归一化 + eps 保护
        # c = F.normalize(c, dim=1)
        eps = self.eps_center
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c > 0)] = eps
        if 'c' not in self._buffers:
            self.register_buffer('c', c)
        else:
            self.c.copy_(c)


    # ---------------------- Multi-Class SVDD Loss (Hard Edge) ----------------------
    def compute_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        """
        DeepSVDD 损失：样本到其对应类别中心的距离平方；使用激活后的中间特征 z
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        _,z = self.forward(x_mel,x_wav)              # [B, z_dim]
        cz = self.c[label]                   # [B, z_dim]（y: 0..K-1 的 LongTensor）
        loss_svdd = torch.mean(torch.sum((z - cz) ** 2, dim=1))
        return loss_svdd

    # -------------------- One-Class SVDD Loss (Hard Edge) -------------------
    def compute_oneclass_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        DeepSVDD 损失：样本到其对应类别中心的距离平方；使用激活后的中间特征 z
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        _, z = self.forward(x_mel,x_wav)  # [B, z_dim]
        cz = torch.mean(self.c,dim=0,keepdim=True)  # [B, z_dim]（y: 0..K-1 的 LongTensor）
        loss_svdd = torch.mean(torch.sum((z - cz) ** 2, dim=1))
        return loss_svdd


    # ---------------------- Multi-Class SVDD Loss (Soft Edge) ----------------------
    def compute_soft_svdd_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor):
        """
        dist = ||z - c_y||^2
        R_y = softplus(R_raw[y]) >= 0
        loss = mean(R_y^2) + (1/nu) * mean( max(0, dist - R_y^2) )
        返回: (loss_soft, scores)；scores=dist-R_y^2（越大越异常）
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")

        logits, z = self.forward(x_mel,x_wav)  # z: [B, z_dim]
        cz = self.c[label]  # [B, z_dim]
        dist = torch.sum((z - cz) ** 2, dim=1)  # [B]

        R = F.softplus(self.R_raw)  # [K]
        R_y = R[label]  # [B]
        scores = dist - R_y ** 2

        loss_R = torch.mean(R_y ** 2)
        loss_hinge = torch.mean(torch.clamp(scores, min=0.0))
        loss_soft = loss_R + (1.0 / self.nu) * loss_hinge
        return loss_soft, scores



    # ---------------------- Classification Loss  ----------------------
    def compute_classification_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor ,label: torch.Tensor):
        """
        交叉熵分类损失；使用最后线性层输出 logits
        返回：loss_cls, logits
        """
        logits, z = self.forward(x_mel, x_wav, return_logits=True)
        loss_cls = F.cross_entropy(logits, label)
        return loss_cls, logits

    # ---------------------- One-Class SVDD Anomaly Score ----------------------
    def compute_oneclass_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel,x_wav)  # [B, z_dim]
        # d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B, K]
        cz = torch.mean(self.c,dim=0,keepdim=True)
        score = torch.sum((z - cz) ** 2)

        # score, _ = torch.min(d2, dim=1)
        return score

    # ---------------------- Multi-Class SVDD Anomaly Score ----------------------
    def compute_anomaly_score(self, x_mel: torch.Tensor,x_wav: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel,x_wav)  # [B, z_dim]
        #d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B, K]
        cz = self.c[label]
        score = torch.sum((z - cz) ** 2)

        #score, _ = torch.min(d2, dim=1)
        return score



    # ---------------------- 考虑分类错误的异常分数计算 -----------------------
    def compute_anomaly_score_with_classification_weight(
            self,
            x_mel: torch.Tensor,
            x_wav: torch.Tensor,
            label: torch.Tensor,
            gamma: float = 1.0,  # 映射陡峭度：越大，偏离越敏感
            coef_max: float = 10.0,  # 系数上限，防极端值
            eps: float = 1e-12
    ) -> torch.Tensor:
        """
        若最近中心==真实标签: score = ||z - c_label||^2
        否则: score = ||z - c_label||^2 * alpha, 其中 alpha = (p_true + eps)^(-gamma) ∈ [1, coef_max]
        返回: [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")

        logits, z = self.forward(x_mel, x_wav)  # logits: [B,K], z: [B,D]
        # 与各中心的距离平方
        d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B,K]
        pred = torch.argmin(d2, dim=1)  # [B]

        # 到真实类中心的距离（原逻辑的基准分数）
        d_true = torch.sum((z - self.c[label]) ** 2, dim=1)  # [B]

        # 用分类 logits 计算真类概率 p_true
        p = torch.softmax(logits, dim=1)  # [B,K]
        p_true = p.gather(1, label.view(-1, 1)).squeeze(1)  # [B]

        # 将 p_true -> 系数 alpha，保证 >=1，并加上上限避免爆炸
        alpha = torch.clamp((p_true + eps).pow(-gamma), min=1.0, max=coef_max)  # [B]

        # 仅在预测错误时应用放大系数
        mismatch = (pred != label)
        score = d_true.clone()
        score[mismatch] = score[mismatch] * alpha[mismatch]

        return score
    
    # ---------------------- Multi-Class SVDD  Anomaly Score (Soft)----------------------
    def soft_boundary_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor = None):
        """
        若提供 label：返回 dist - R_y^2 （[B]）
        若不提供 label：返回 min_k(dist_k - R_k^2) （[B]）
        """
        logits, z = self.forward(x_mel,x_wav)
        R = F.softplus(self.R_raw)  # [K]

        if label is not None:
            cz = self.c[label]  # [B, z_dim]
            dist = torch.sum((z - cz) ** 2, dim=1)  # [B]
            return dist - R[label] ** 2
        else:
            d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B,K]
            scores = d2 - R.view(1, -1) ** 2  # [B,K]
            return torch.min(scores, dim=1)[0]


    # ---------------------- Classification Anomaly Score ----------------------
    def compute_classification_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel, x_wav)  # [B, z_dim]

        return logits

    @torch.no_grad()
    def _l2_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) if self.contrastive_l2_normalize else x






# Multi-Class SVDD mel spectrogram only 比采用tgram略微差一点
class ClassVDD_newmobile_mel_only(nn.Module):
    """
    Backbone: MobileNetV2 (首层改 1 通道) -> GAP -> Linear(z_dim) -> Activation -> Linear(num_classes)
    - SVDD 使用激活后的中间特征 z
    - 分类使用最后线性层输出的 logits
    """

    def __init__(
            self,
            num_classes,
            device,
            nu: float = 0.2,
            z_dim: int = 128,
            eps_center: float = 0.01,
            leak_slope: float = 0.2,
            in_ch: int = 1,  # 输入通道，默认为 1
            mobilenet_width_mult: float = 1.0,
            bottleneck_setting = Mobilefacenet_bottleneck_setting,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.device = device
        self.eps_center = eps_center
        self.leak_slope = leak_slope

        # ===== MobileNetV2 backbone=====
        self.mobilefacenet = MobileFaceNet(num_class=num_classes+z_dim,
                                           bottleneck_setting=bottleneck_setting,
                                           in_channels=1
                                           )
        
        
        
       
        # ===== TGram Feature Extractor =====
        # self.tgramnet = TgramNet(mel_bins=128, win_len=1024, hop_len=512)
        
      
        
        self.register_buffer('c', torch.zeros(num_classes, z_dim))

        # Soft Edge
        self.nu = nu  # 0.05~0.2

        # Trainable Radius for each class
        self.R_raw = nn.Parameter(torch.full((num_classes,), -10.0))



    def forward(self, x_mel: torch.Tensor, x_wav: torch.Tensor,return_logits: bool = False):
        """
        x: (B,1,H,W)
        return:
          logits: [B, num_classes]
          z: 激活后的中间特征，shape [B, z_dim]
        """
        
        
        out, feature = self.mobilefacenet(x_mel)

        # latent for svdd and logits for classification
        z = out[:, :self.z_dim]
        logits = out[:, self.z_dim:]
        return logits, z

    # ---------------------- initialize center ----------------------
    def set_c(self, dataloader):
        """
        随机初始化每类中心 c[k]，与标签无关；不扫描数据。
        仍保留原函数名与调用方式，方便兼容你现有训练代码。
        """
        c = torch.randn(self.num_classes, self.z_dim).to(self.device)
        # 可选：行归一化 + eps 保护
        # c = F.normalize(c, dim=1)
        eps = self.eps_center
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c > 0)] = eps
        if 'c' not in self._buffers:
            self.register_buffer('c', c)
        else:
            self.c.copy_(c)


    # ---------------------- Multi-Class SVDD Loss (Hard Edge) ----------------------
    def compute_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        """
        DeepSVDD 损失：样本到其对应类别中心的距离平方；使用激活后的中间特征 z
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        _,z = self.forward(x_mel,x_wav)              # [B, z_dim]
        cz = self.c[label]                   # [B, z_dim]（y: 0..K-1 的 LongTensor）
        loss_svdd = torch.mean(torch.sum((z - cz) ** 2, dim=1))
        return loss_svdd

    # -------------------- One-Class SVDD Loss (Hard Edge) -------------------
    def compute_oneclass_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        DeepSVDD 损失：样本到其对应类别中心的距离平方；使用激活后的中间特征 z
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        _, z = self.forward(x_mel,x_wav)  # [B, z_dim]
        cz = torch.mean(self.c,dim=0,keepdim=True)  # [B, z_dim]（y: 0..K-1 的 LongTensor）
        loss_svdd = torch.mean(torch.sum((z - cz) ** 2, dim=1))
        return loss_svdd


    # ---------------------- Multi-Class SVDD Loss (Soft Edge) ----------------------
    def compute_soft_svdd_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor):
        """
        dist = ||z - c_y||^2
        R_y = softplus(R_raw[y]) >= 0
        loss = mean(R_y^2) + (1/nu) * mean( max(0, dist - R_y^2) )
        返回: (loss_soft, scores)；scores=dist-R_y^2（越大越异常）
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")

        logits, z = self.forward(x_mel,x_wav)  # z: [B, z_dim]
        cz = self.c[label]  # [B, z_dim]
        dist = torch.sum((z - cz) ** 2, dim=1)  # [B]

        R = F.softplus(self.R_raw)  # [K]
        R_y = R[label]  # [B]
        scores = dist - R_y ** 2

        loss_R = torch.mean(R_y ** 2)
        loss_hinge = torch.mean(torch.clamp(scores, min=0.0))
        loss_soft = loss_R + (1.0 / self.nu) * loss_hinge
        return loss_soft, scores



    # ---------------------- Classification Loss  ----------------------
    def compute_classification_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor ,label: torch.Tensor):
        """
        交叉熵分类损失；使用最后线性层输出 logits
        返回：loss_cls, logits
        """
        logits, z = self.forward(x_mel, x_wav, return_logits=True)
        loss_cls = F.cross_entropy(logits, label)
        return loss_cls, logits

    # ---------------------- One-Class SVDD Anomaly Score ----------------------
    def compute_oneclass_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel,x_wav)  # [B, z_dim]
        # d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B, K]
        cz = torch.mean(self.c,dim=0,keepdim=True)
        score = torch.sum((z - cz) ** 2)

        # score, _ = torch.min(d2, dim=1)
        return score

    # ---------------------- Multi-Class SVDD Anomaly Score ----------------------
    def compute_anomaly_score(self, x_mel: torch.Tensor,x_wav: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel,x_wav)  # [B, z_dim]
        #d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B, K]
        cz = self.c[label]
        score = torch.sum((z - cz) ** 2)

        #score, _ = torch.min(d2, dim=1)
        return score

    
    # ---------------------- 考虑分类错误的异常分数计算 -----------------------
    def compute_anomaly_score_with_classification_weight(
            self,
            x_mel: torch.Tensor,
            x_wav: torch.Tensor,
            label: torch.Tensor,
            gamma: float = 1.0,  # 映射陡峭度：越大，偏离越敏感
            coef_max: float = 10.0,  # 系数上限，防极端值
            eps: float = 1e-12
    ) -> torch.Tensor:
        """
        若最近中心==真实标签: score = ||z - c_label||^2
        否则: score = ||z - c_label||^2 * alpha, 其中 alpha = (p_true + eps)^(-gamma) ∈ [1, coef_max]
        返回: [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")

        logits, z = self.forward(x_mel, x_wav)  # logits: [B,K], z: [B,D]
        # 与各中心的距离平方
        d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B,K]
        pred = torch.argmin(d2, dim=1)  # [B]

        # 到真实类中心的距离（原逻辑的基准分数）
        d_true = torch.sum((z - self.c[label]) ** 2, dim=1)  # [B]

        # 用分类 logits 计算真类概率 p_true
        p = torch.softmax(logits, dim=1)  # [B,K]
        p_true = p.gather(1, label.view(-1, 1)).squeeze(1)  # [B]

        # 将 p_true -> 系数 alpha，保证 >=1，并加上上限避免爆炸
        alpha = torch.clamp((p_true + eps).pow(-gamma), min=1.0, max=coef_max)  # [B]

        # 仅在预测错误时应用放大系数
        mismatch = (pred != label)
        score = d_true.clone()
        score[mismatch] = score[mismatch] * alpha[mismatch]

        return score


    
    # ---------------------- Multi-Class SVDD  Anomaly Score (Soft)----------------------
    def soft_boundary_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor = None):
        """
        若提供 label：返回 dist - R_y^2 （[B]）
        若不提供 label：返回 min_k(dist_k - R_k^2) （[B]）
        """
        logits, z = self.forward(x_mel,x_wav)
        R = F.softplus(self.R_raw)  # [K]

        if label is not None:
            cz = self.c[label]  # [B, z_dim]
            dist = torch.sum((z - cz) ** 2, dim=1)  # [B]
            return dist - R[label] ** 2
        else:
            d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B,K]
            scores = d2 - R.view(1, -1) ** 2  # [B,K]
            return torch.min(scores, dim=1)[0]


    # ---------------------- Classification Anomaly Score ----------------------
    def compute_classification_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel, x_wav)  # [B, z_dim]

        return logits

    @torch.no_grad()
    def _l2_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) if self.contrastive_l2_normalize else x



# Sinc backbone
class SincConv_fast(nn.Module):
    """Sinc-based 1D convolution with softplus-constrained frequency parameters."""

    @staticmethod
    def to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel: np.ndarray) -> np.ndarray:
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels: int, # 输出滤波器数量
        kernel_size: int, # 每个滤波器的长度
        sample_rate: int = 16000, # 输入信号的采样率
        in_channels: int = 1, # 输入通道数1
        stride: int = 1, # Conv1d 的步长
        padding: int = 0, # Conv1d 的填充长度。
        dilation: int = 1, # Conv1d 的膨胀系数。
        bias: bool = False,
        groups: int = 1,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
    ):
        super().__init__()

        if in_channels != 1:
            raise ValueError(f"SincConv only supports one input channel (got in_channels={in_channels}).")

        self.out_channels = out_channels
        self.kernel_size = int(kernel_size) | 1          # 强制奇数长度
        self.sample_rate = int(sample_rate)
        self.stride, self.padding, self.dilation = stride, padding, dilation
        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups != 1:
            raise ValueError("SincConv does not support groups.")

        self.min_low_hz = float(min_low_hz)
        self.min_band_hz = float(min_band_hz)

        # ===== 初始化滤波器频率参数 =====
        low_hz = 30.0
        high_hz = self.sample_rate / 2.0 - (self.min_low_hz + self.min_band_hz)

        # mel频率
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), out_channels + 1)

        # 对应mel的真实频率
        hz = self.to_hz(mel)

        # 低通频率变为可学习参数
        self.low_hz_ = nn.Parameter(torch.tensor(hz[:-1], dtype=torch.float32).view(-1, 1))

        # 带通频率也为可学习参数
        self.band_hz_ = nn.Parameter(torch.tensor(np.diff(hz), dtype=torch.float32).view(-1, 1))

        # Hamming 半窗与半时间轴
        half = self.kernel_size // 2
        n_lin = torch.linspace(0, half - 1, steps=half)
        window_half = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)
        self.register_buffer("window_half", window_half) # window half是固定参数
        n = (self.kernel_size - 1) / 2.0
        t_half = 2 * math.pi * torch.arange(-n, 0) / self.sample_rate
        self.register_buffer("t_half", t_half.view(1, -1))
        self.register_buffer("filters", torch.empty(out_channels, 1, self.kernel_size))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        dtype, device = waveforms.dtype, waveforms.device
        t_half = self.t_half.to(dtype=dtype, device=device)
        window_half = self.window_half.to(dtype=dtype, device=device)

        # ===== softplus 约束 =====
        low = self.min_low_hz + F.softplus(self.low_hz_)              # 代替 abs
        high = torch.clamp(
            low + self.min_band_hz + F.softplus(self.band_hz_),      # 代替 abs
            min=self.min_low_hz,
            max=self.sample_rate / 2.0,
        )
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, t_half)
        f_times_t_high = torch.matmul(high, t_half)

        denom = (t_half / 2.0).to(dtype=dtype, device=device)
        eps = torch.finfo(dtype).tiny
        denom = torch.where(torch.abs(denom) < eps, torch.full_like(denom, eps), denom)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / denom) * window_half
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_safe = torch.where(band <= 0, torch.full_like(band, 1.0), band)
        band_pass = band_pass / (2 * band_safe[:, None])

        self.filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )

# Multi-Class SVDD with Sinc Backbone 似乎有提升 有点希望
class ClassVDD_newmobile_sinc(nn.Module):
    """
    Backbone: MobileNetV2 (首层改 1 通道) -> GAP -> Linear(z_dim) -> Activation -> Linear(num_classes)
    - SVDD 使用激活后的中间特征 z
    - 分类使用最后线性层输出的 logits
    """

    def __init__(
            self,
            num_classes,
            device,
            z_dim: int = 128,
            eps_center: float = 0.01,
            leak_slope: float = 0.2,
            in_ch: int = 1,  # 输入通道，默认为 1
            mobilenet_width_mult: float = 1.0,
            bottleneck_setting = Mobilefacenet_bottleneck_setting,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.device = device
        self.eps_center = eps_center
        self.leak_slope = leak_slope

        # ===== MobileNetV2 backbone=====
        self.mobilefacenet = MobileFaceNet(num_class=num_classes+z_dim,
                                           bottleneck_setting=bottleneck_setting,
                                           in_channels=2)
        # ===== Sinc Feature Extractor =====
        self.sinc = SincConv_fast(
            out_channels=128,  # 滤波器数量
            kernel_size=1025,  # 每个滤波器长度
            sample_rate=16000,  # 采样率
            stride=512,
            padding=512,
            min_low_hz=20,
        )
        
        self.register_buffer('c', torch.zeros(num_classes, z_dim))

        # Soft Edge
        self.nu = 0.1  # 0.05~0.2

        # Trainable Radius for each class
        self.R_raw = nn.Parameter(torch.full((num_classes,), -10.0))



    def forward(self, x_mel: torch.Tensor, x_wav: torch.Tensor,return_logits: bool = False):
        """
        x: (B,1,H,W)
        return:
          logits: [B, num_classes]
          z: 激活后的中间特征，shape [B, z_dim]
        """
        x_t = self.sinc(x_wav).unsqueeze(1)
        x = torch.cat((x_mel, x_t), dim=1)
       

        out, feature = self.mobilefacenet(x)

        # latent for svdd and logits for classification
        z = out[:, :self.z_dim]
        logits = out[:, self.z_dim:]
        return logits, z

    # ---------------------- initialize center ----------------------
    def set_c(self, dataloader):
        """
        随机初始化每类中心 c[k]，与标签无关；不扫描数据。
        仍保留原函数名与调用方式，方便兼容你现有训练代码。
        """
        c = torch.randn(self.num_classes, self.z_dim).to(self.device)
        # 可选：行归一化 + eps 保护
        # c = F.normalize(c, dim=1)
        eps = self.eps_center
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c > 0)] = eps
        if 'c' not in self._buffers:
            self.register_buffer('c', c)
        else:
            self.c.copy_(c)


    # ---------------------- Multi-Class SVDD Loss (Hard Edge) ----------------------
    def compute_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        """
        DeepSVDD 损失：样本到其对应类别中心的距离平方；使用激活后的中间特征 z
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        _,z = self.forward(x_mel,x_wav)              # [B, z_dim]
        cz = self.c[label]                   # [B, z_dim]（y: 0..K-1 的 LongTensor）
        loss_svdd = torch.mean(torch.sum((z - cz) ** 2, dim=1))
        return loss_svdd

    # -------------------- One-Class SVDD Loss (Hard Edge) -------------------
    def compute_oneclass_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        DeepSVDD 损失：样本到其对应类别中心的距离平方；使用激活后的中间特征 z
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        _, z = self.forward(x_mel,x_wav)  # [B, z_dim]
        cz = torch.mean(self.c,dim=0,keepdim=True)  # [B, z_dim]（y: 0..K-1 的 LongTensor）
        loss_svdd = torch.mean(torch.sum((z - cz) ** 2, dim=1))
        return loss_svdd

    # ---------------------- 考虑分类错误的异常分数计算 -----------------------
    def compute_anomaly_score_with_classification_weight(
            self,
            x_mel: torch.Tensor,
            x_wav: torch.Tensor,
            label: torch.Tensor,
            gamma: float = 1.0,  # 映射陡峭度：越大，偏离越敏感
            coef_max: float = 10.0,  # 系数上限，防极端值
            eps: float = 1e-12
    ) -> torch.Tensor:
        """
        若最近中心==真实标签: score = ||z - c_label||^2
        否则: score = ||z - c_label||^2 * alpha, 其中 alpha = (p_true + eps)^(-gamma) ∈ [1, coef_max]
        返回: [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")

        logits, z = self.forward(x_mel, x_wav)  # logits: [B,K], z: [B,D]
        # 与各中心的距离平方
        d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B,K]
        pred = torch.argmin(d2, dim=1)  # [B]

        # 到真实类中心的距离（原逻辑的基准分数）
        d_true = torch.sum((z - self.c[label]) ** 2, dim=1)  # [B]

        # 用分类 logits 计算真类概率 p_true
        p = torch.softmax(logits, dim=1)  # [B,K]
        p_true = p.gather(1, label.view(-1, 1)).squeeze(1)  # [B]

        # 将 p_true -> 系数 alpha，保证 >=1，并加上上限避免爆炸
        alpha = torch.clamp((p_true + eps).pow(-gamma), min=1.0, max=coef_max)  # [B]

        # 仅在预测错误时应用放大系数
        mismatch = (pred != label)
        score = d_true.clone()
        score[mismatch] = score[mismatch] * alpha[mismatch]

        return score

    # ---------------------- Multi-Class SVDD Loss (Soft Edge) ----------------------
    def compute_soft_svdd_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor):
        """
        dist = ||z - c_y||^2
        R_y = softplus(R_raw[y]) >= 0
        loss = mean(R_y^2) + (1/nu) * mean( max(0, dist - R_y^2) )
        返回: (loss_soft, scores)；scores=dist-R_y^2（越大越异常）
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")

        logits, z = self.forward(x_mel,x_wav)  # z: [B, z_dim]
        cz = self.c[label]  # [B, z_dim]
        dist = torch.sum((z - cz) ** 2, dim=1)  # [B]

        R = F.softplus(self.R_raw)  # [K]
        R_y = R[label]  # [B]
        scores = dist - R_y ** 2

        loss_R = torch.mean(R_y ** 2)
        loss_hinge = torch.mean(torch.clamp(scores, min=0.0))
        loss_soft = loss_R + (1.0 / self.nu) * loss_hinge
        return loss_soft, scores



    # ---------------------- Classification Loss  ----------------------
    def compute_classification_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor ,label: torch.Tensor):
        """
        交叉熵分类损失；使用最后线性层输出 logits
        返回：loss_cls, logits
        """
        logits, z = self.forward(x_mel, x_wav, return_logits=True)
        loss_cls = F.cross_entropy(logits, label)
        return loss_cls, logits

    # ---------------------- One-Class SVDD Anomaly Score ----------------------
    def compute_oneclass_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel,x_wav)  # [B, z_dim]
        # d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B, K]
        cz = torch.mean(self.c,dim=0,keepdim=True)
        score = torch.sum((z - cz) ** 2)

        # score, _ = torch.min(d2, dim=1)
        return score

    # ---------------------- Multi-Class SVDD Anomaly Score ----------------------
    def compute_anomaly_score(self, x_mel: torch.Tensor,x_wav: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel,x_wav)  # [B, z_dim]
        #d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B, K]
        cz = self.c[label]
        score = torch.sum((z - cz) ** 2)

        #score, _ = torch.min(d2, dim=1)
        return score

    # ---------------------- Multi-Class SVDD  Anomaly Score (Soft)----------------------
    def soft_boundary_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor = None):
        """
        若提供 label：返回 dist - R_y^2 （[B]）
        若不提供 label：返回 min_k(dist_k - R_k^2) （[B]）
        """
        logits, z = self.forward(x_mel,x_wav)
        R = F.softplus(self.R_raw)  # [K]

        if label is not None:
            cz = self.c[label]  # [B, z_dim]
            dist = torch.sum((z - cz) ** 2, dim=1)  # [B]
            return dist - R[label] ** 2
        else:
            d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B,K]
            scores = d2 - R.view(1, -1) ** 2  # [B,K]
            return torch.min(scores, dim=1)[0]


    # ---------------------- Classification Anomaly Score ----------------------
    def compute_classification_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel, x_wav)  # [B, z_dim]

        return logits

    @torch.no_grad()
    def _l2_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) if self.contrastive_l2_normalize else x



#------------------------------------ dprnn model---------------------------------------------------------------


# 编码器 1维卷积+激活函数，同时有对输入的维度扩充，带输入的序列的dim1上插入了1个维度，从[B,T]--->[B,1,T],然后送入到卷积层中
class Encoder(nn.Module):
    '''
       Conv-Tasnet Encoder part
       kernel_size: the length of filters
       out_channels: the number of filters
    '''

    def __init__(self, kernel_size ,stride, padding, out_channels=64, audio_channels=1,):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=audio_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, groups=1, bias=False)

    def forward(self, x):
        """
          Input:
              x: [B, 8, T], B is batch size, T is times
              源代码需要对输入序列插入一个维度，而新的序列本身有通道数，因此不插入
              输出的形状不发生变化，还是B C T_out
          Returns:
              x: [B, C, T_out]
              T_out is the number of time steps
        """
        # B x T -> B x 1 x T
        #x = torch.unsqueeze(x, dim=1)
        # B x 1 x T -> B x C x T_out
        x = self.conv1d(x)
        x = F.relu(x)
        return x

# 解码器，没有激活函数，单纯的反卷积；如果输入x的维度不是2或3，则报错
class Decoder(nn.ConvTranspose1d):
    '''
        Decoder of the TasNet
        This module can be seen as the gradient of Conv1d with respect to its input.
        It is also known as a fractionally-strided convolution
        or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: [B, N, L]
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        # if torch.squeeze(x).dim() == 1:
        #     x = torch.squeeze(x, dim=1)
        # else:
        #     x = torch.squeeze(x)
        return x


# 全局层归一化函数
class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
          input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        return x

# 累计层归一化函数
class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
           x = x.permute(0, 2, 3, 1).contiguous()
           # N x K x S x C == only channel norm
           x = super().forward(x)
           # N x C x K x S
           x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x

# 选取归一化函数
def select_norm(norm, dim, shape):
    if norm == 'gln':
        return GlobalLayerNorm(dim, shape, elementwise_affine=True) # shape是输入特征图的维度数
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)

# DPRNN模块
"""
序列切块，块内RNN一次，块间RNN一次
"""
class Dual_RNN_Block(nn.Module):
    '''
       Implementation of the intra-RNN and the inter-RNN
       input:
            in_channels: 输入序列x的特征维数 The number of expected features in the input x
            out_channels: 输出序列的特征维数 The number of features in the hidden state h
            rnn_type: RNN类型 RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs
                     of each LSTM layer except the last layer,
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    '''

    def __init__(self, out_channels,
                 hidden_channels, rnn_type='LSTM', norm='ln',
                 dropout=0, bidirectional=False, num_spks=2):
        super(Dual_RNN_Block, self).__init__()
        # RNN model
        # 其实就是两个一模一样的RNN
        # rnn的输出维数就是hidden_size，双向rnn则是2*hidden_size
        self.intra_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.inter_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        # Norm
        self.intra_norm = select_norm(norm, out_channels, 4)
        self.inter_norm = select_norm(norm, out_channels, 4)
        # Linear
        self.intra_linear = nn.Linear(
            hidden_channels * 2 if bidirectional else hidden_channels, out_channels)
        self.inter_linear = nn.Linear(
            hidden_channels * 2 if bidirectional else hidden_channels, out_channels)

    def forward(self, x):
        '''
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        '''
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)  # 批量数*切块数量 切块长度 特征维度
        # [BS, K, H]
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        # [BS, K, N]
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B * S * K, -1)).view(B * S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)

        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)  # 批量数*切块长度 切块数量 特征维度
        # [BK, S, H]
        inter_rnn, _ = self.inter_rnn(inter_rnn)
        # [BK, S, N]
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B * S * K, -1)).view(B * K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out


# DPRNN网络，DPRNN网络包含多个DPRNN模块以及其他一些模块如 2d卷积和门控层
class Dual_Path_RNN(nn.Module):
    '''
       Implementation of the Dual-Path-RNN model
       input:
            in_channels: 输入序列的特征维度
            out_channels: hidden state h
            rnn_type: rnn网络的类型 RNN, LSTM, GRU

            norm: 归一化的类型 gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs
                     of each LSTM layer except the last layer,
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False

            num_layers:  The  Number of Dual-Path-Block 的数量
            K: the length of chunk 切块的长度
            num_spks: the number of speakers 说话人数量
    '''

    def __init__(self, in_channels, out_channels, hidden_channels,
                 rnn_type='LSTM', norm='ln', dropout=0,
                 bidirectional=False, num_layers=4, K=200):
        super(Dual_Path_RNN, self).__init__()
        self.K = K # 序列切块大小



        self.num_layers = num_layers
        self.norm = select_norm(norm, in_channels, 3)# 选择归一化函数，shape是输入特征图的维度数（我的好像是4）

        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        # 经过编码器后，还要经过一次归一化、以及一次 1维卷积；然后 送入到DPRNN模块中
        # self.dual_rnn = nn.ModuleList([])
        # for i in range(num_layers):
        #     # 添加双路径RNN模块
        #     self.dual_rnn.append(Dual_RNN_Block(out_channels, hidden_channels,
        #                              rnn_type=rnn_type, norm=norm, dropout=dropout,
        #                              bidirectional=bidirectional))
        self.dual_rnn = nn.Sequential()
        for i in range(num_layers):
            # 添加双路径RNN模块
            self.dual_rnn.add_module('dual_rnn_block_%d' % i,
                                     Dual_RNN_Block(out_channels, hidden_channels,
                                                    rnn_type=rnn_type, norm=norm, dropout=dropout,
                                                    bidirectional=bidirectional))
        # 2d卷积
        self.conv2d = nn.Conv2d(
            out_channels, out_channels, kernel_size=1)

        # 1d卷积
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()

        self.activation = nn.ReLU()

         # gated output layer 门控层：1维卷积+反曲激活函数
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1),
                                    nn.Tanh()
                                    )
        # gated output layer 门控层：1维卷积+sigmoid激活函数
        self.output_gate = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1),
                                         nn.Sigmoid()
                                         )

    def forward(self, x):
        '''
           x: [B, N, L]
           N 是卷积层扩充后的输出通道数

        '''
        # [B, N, L]
        x = self.norm(x)
        # [B, N, L]
        x = self.conv1d(x)
        # 输入的序列先归一化,(上一步卷积中没有归一化，放到这其实不大对)


        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)


        # [B, N*spks, K, S]
        # 这里为什么要有一个循环？
        # for i in range(self.num_layers):
        #     x = self.dual_rnn[i](x)

        x = self.dual_rnn(x)
        x = self.prelu(x)
        x = self.conv2d(x) # 这里将通道数扩充为说话人数*原通道数，方便后续的说话人分离


        # [B*spks, N, K, S]
        B, _, K, S = x.shape
        x = x.view(B,-1, K, S)
        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x)*self.output_gate(x)

        # [spks*B, N, L]
        x = self.end_conv1x1(x)

        # [B*spks, N, L] -> [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, N, L)
        x = self.activation(x)


        return x

    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


# DPRNN 整体网络模型
class Dual_RNN_model(nn.Module):
    '''
       model of Dual Path RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            hidden_channels: The hidden size of RNN
            kernel_size: Encoder and Decoder Kernel size
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs
                     of each LSTM layer except the last layer,
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
            num_layers: number of Dual-Path-Block
            K: the length of chunk
            num_spks: the number of speakers
    '''

    def __init__(self, in_channels, out_channels, hidden_channels,
                 kernel_size=2, rnn_type='LSTM', norm='ln', dropout=0,
                 bidirectional=False, num_layers=2, K=200, audio_channels=1):
        super(Dual_RNN_model, self).__init__()
        self.encoder = Encoder(kernel_size=1024, stride=512, padding=512, out_channels=in_channels, audio_channels=1)
        self.separation = Dual_Path_RNN(in_channels, out_channels, hidden_channels,
                                        rnn_type=rnn_type, norm=norm, dropout=dropout,
                                        bidirectional=bidirectional, num_layers=num_layers, K=K)
        self.decoder = Decoder(in_channels=in_channels, out_channels=audio_channels, kernel_size=kernel_size,
                               stride=kernel_size // 2, bias=False)


    def forward(self, x):
        '''
           x: [B, C, L]
        '''

        """       
        输入是[B, C, L]，输出是[B,out_channels,L-1]
        这里进行了通道的扩充
        """
        e = self.encoder(x)  # 1维卷积 + 激活函数
        s = self.separation(e) # DPRNN # DPRNN输出的就是Mask

        return s




# Multi-Class SVDD with dprnn
class ClassVDD_newmobile_dprnn(nn.Module):
    """
    Backbone: MobileNetV2 (首层改 1 通道) -> GAP -> Linear(z_dim) -> Activation -> Linear(num_classes)
    - SVDD 使用激活后的中间特征 z
    - 分类使用最后线性层输出的 logits
    """

    def __init__(
            self,
            num_classes,
            device,
            z_dim: int = 128,
            eps_center: float = 0.01,
            leak_slope: float = 0.2,
            in_ch: int = 1,  # 输入通道，默认为 1
            mobilenet_width_mult: float = 1.0,
            bottleneck_setting = Mobilefacenet_bottleneck_setting,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.device = device
        self.eps_center = eps_center
        self.leak_slope = leak_slope

        # ===== MobileNetV2 backbone=====
        self.mobilefacenet = MobileFaceNet(num_class=num_classes+z_dim,
                                           bottleneck_setting=bottleneck_setting,
                                           in_channels=2)
        # ===== TGram Feature Extractor =====
        self.dprnn = Dual_RNN_model(128,64, 128, bidirectional=True, norm='ln', num_layers=1)

        self.register_buffer('c', torch.zeros(num_classes, z_dim))

        # Soft Edge
        self.nu = 0.1  # 0.05~0.2

        # Trainable Radius for each class
        self.R_raw = nn.Parameter(torch.full((num_classes,), -10.0))



    def forward(self, x_mel: torch.Tensor, x_wav: torch.Tensor,return_logits: bool = False):
        """
        x: (B,1,H,W)
        return:
          logits: [B, num_classes]
          z: 激活后的中间特征，shape [B, z_dim]
        """
        x_t = self.dprnn(x_wav).unsqueeze(1)
        x = torch.cat((x_mel, x_t), dim=1)
        

        out, feature = self.mobilefacenet(x)

        # latent for svdd and logits for classification
        z = out[:, :self.z_dim]
        logits = out[:, self.z_dim:]
        return logits, z

    # ---------------------- initialize center ----------------------
    def set_c(self, dataloader):
        """
        随机初始化每类中心 c[k]，与标签无关；不扫描数据。
        仍保留原函数名与调用方式，方便兼容你现有训练代码。
        """
        c = torch.randn(self.num_classes, self.z_dim).to(self.device)
        # 可选：行归一化 + eps 保护
        # c = F.normalize(c, dim=1)
        eps = self.eps_center
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c > 0)] = eps
        if 'c' not in self._buffers:
            self.register_buffer('c', c)
        else:
            self.c.copy_(c)


    # ---------------------- Multi-Class SVDD Loss (Hard Edge) ----------------------
    def compute_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        """
        DeepSVDD 损失：样本到其对应类别中心的距离平方；使用激活后的中间特征 z
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        _,z = self.forward(x_mel,x_wav)              # [B, z_dim]
        cz = self.c[label]                   # [B, z_dim]（y: 0..K-1 的 LongTensor）
        loss_svdd = torch.mean(torch.sum((z - cz) ** 2, dim=1))
        return loss_svdd

    # -------------------- One-Class SVDD Loss (Hard Edge) -------------------
    def compute_oneclass_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        DeepSVDD 损失：样本到其对应类别中心的距离平方；使用激活后的中间特征 z
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        _, z = self.forward(x_mel,x_wav)  # [B, z_dim]
        cz = torch.mean(self.c,dim=0,keepdim=True)  # [B, z_dim]（y: 0..K-1 的 LongTensor）
        loss_svdd = torch.mean(torch.sum((z - cz) ** 2, dim=1))
        return loss_svdd


    # ---------------------- Multi-Class SVDD Loss (Soft Edge) ----------------------
    def compute_soft_svdd_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor):
        """
        dist = ||z - c_y||^2
        R_y = softplus(R_raw[y]) >= 0
        loss = mean(R_y^2) + (1/nu) * mean( max(0, dist - R_y^2) )
        返回: (loss_soft, scores)；scores=dist-R_y^2（越大越异常）
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")

        logits, z = self.forward(x_mel,x_wav)  # z: [B, z_dim]
        cz = self.c[label]  # [B, z_dim]
        dist = torch.sum((z - cz) ** 2, dim=1)  # [B]

        R = F.softplus(self.R_raw)  # [K]
        R_y = R[label]  # [B]
        scores = dist - R_y ** 2

        loss_R = torch.mean(R_y ** 2)
        loss_hinge = torch.mean(torch.clamp(scores, min=0.0))
        loss_soft = loss_R + (1.0 / self.nu) * loss_hinge
        return loss_soft, scores



    # ---------------------- Classification Loss  ----------------------
    def compute_classification_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor ,label: torch.Tensor):
        """
        交叉熵分类损失；使用最后线性层输出 logits
        返回：loss_cls, logits
        """
        logits, z = self.forward(x_mel, x_wav, return_logits=True)
        loss_cls = F.cross_entropy(logits, label)
        return loss_cls, logits

    # ---------------------- One-Class SVDD Anomaly Score ----------------------
    def compute_oneclass_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel,x_wav)  # [B, z_dim]
        # d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B, K]
        cz = torch.mean(self.c,dim=0,keepdim=True)
        score = torch.sum((z - cz) ** 2)

        # score, _ = torch.min(d2, dim=1)
        return score

    # ---------------------- Multi-Class SVDD Anomaly Score ----------------------
    def compute_anomaly_score(self, x_mel: torch.Tensor,x_wav: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel,x_wav)  # [B, z_dim]
        #d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B, K]
        cz = self.c[label]
        score = torch.sum((z - cz) ** 2)

        #score, _ = torch.min(d2, dim=1)
        return score

    # ---------------------- Multi-Class SVDD  Anomaly Score (Soft)----------------------
    def soft_boundary_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor = None):
        """
        若提供 label：返回 dist - R_y^2 （[B]）
        若不提供 label：返回 min_k(dist_k - R_k^2) （[B]）
        """
        logits, z = self.forward(x_mel,x_wav)
        R = F.softplus(self.R_raw)  # [K]

        if label is not None:
            cz = self.c[label]  # [B, z_dim]
            dist = torch.sum((z - cz) ** 2, dim=1)  # [B]
            return dist - R[label] ** 2
        else:
            d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B,K]
            scores = d2 - R.view(1, -1) ** 2  # [B,K]
            return torch.min(scores, dim=1)[0]


    # ---------------------- Classification Anomaly Score ----------------------
    def compute_classification_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel, x_wav)  # [B, z_dim]

        return logits

    @torch.no_grad()
    def _l2_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) if self.contrastive_l2_normalize else x




















# -----------------------------------------attention block -------------------------------------------------

class AxialAttentionLastDim(nn.Module):
    """
    单通道输入，仅沿最后一维 H 进行轴向注意力：
    - 不进行任何池化或降采样
    - 可选局部窗口注意力以降低复杂度（默认启用）
    复杂度：
      全局：O(B * W * H^2 * d)
      局部：O(B * W * H * (2*win+1) * d)
    形状：
      in/out: [B, 1, W, H]
    """
    def __init__(self,in_channels=1, dim: int = 16, window_size: int = 32, use_bias: bool = False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # None 或 0/>=H 时为全局

        # 升/降维均使用 1x1 卷积，空间尺寸不变
        self.in_proj  = nn.Conv2d(in_channels, dim, kernel_size=1, bias=use_bias)
        self.to_q     = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_k     = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_v     = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(dim, in_channels, kernel_size=1, bias=use_bias)

        # 轻量归一化和非线性 + 残差
        self.norm = nn.GroupNorm(1, dim)
        self.act  = nn.GELU()
        self.gamma = nn.Parameter(torch.tensor(1.0))  # 残差缩放

        self.scale = dim ** -0.5

    @staticmethod
    def _make_band_mask(L: int, window_size: int, device, dtype):
        """生成带状注意力 mask（True 表示可见，False 表示屏蔽）"""
        idx = torch.arange(L, device=device)
        dist = (idx[None, :] - idx[:, None]).abs()  # [L, L]
        band = (dist <= window_size)
        return band.to(dtype=torch.bool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, W, H] -> [B, 1, W, H]
        """
        B, C, W, H = x.shape
        # assert C == 1, "该模块假定输入通道为 1"

        feat = self.in_proj(x)  # [B, dim, W, H]
        z = self.act(self.norm(feat))

        q = self.to_q(z)  # [B, dim, W, H]
        k = self.to_k(z)
        v = self.to_v(z)

        # 仅沿最后一维 H 做注意力：将每个 W 位置视作一条长度 H 的序列
        # 组织为 N=B*W 条序列，长度 L=H，通道 d=dim
        N = B * W
        q = q.permute(0, 2, 3, 1).contiguous().view(N, H, self.dim)  # [N, L, d]
        k = k.permute(0, 2, 3, 1).contiguous().view(N, H, self.dim)  # [N, L, d]
        v = v.permute(0, 2, 3, 1).contiguous().view(N, H, self.dim)  # [N, L, d]

        # 注意力 logits: [N, L, L]
        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))

        # 局部窗口注意力（可选）
        if self.window_size is not None and self.window_size > 0 and self.window_size < H:
            band = self._make_band_mask(H, self.window_size, attn.device, attn.dtype)
            # 将带外位置置为 -inf（用一个足够小的数）
            very_neg = torch.finfo(attn.dtype).min if attn.dtype.is_floating_point else -1e9
            attn = attn.masked_fill(~band, very_neg)

        attn = F.softmax(attn, dim=-1)  # [N, L, L]
        out  = torch.matmul(attn, v)    # [N, L, d]

        # 还原回 [B, dim, W, H]
        out = out.view(B, W, H, self.dim).permute(0, 3, 1, 2).contiguous()  # [B, dim, W, H]

        # 残差、标准化与输出
        out = feat + self.gamma * out
        out = self.out_proj(self.act(self.norm(out)))  # [B, 1, W, H]
        return out


class AxialSelfAttention1D(nn.Module):
    """
    在单一轴向（宽或高）上做自注意力：
    - axis='w'：对每一行，做长度为W的序列注意力（共B*H个序列）
    - axis='h'：对每一列，做长度为H的序列注意力（共B*W个序列）
    输入/输出形状：B, C(=dim), W, H  <->  B, C, W, H
    """
    def __init__(self, dim: int, heads: int = 2, axis: str = 'w'):
        super().__init__()
        assert axis in ('w', 'h')
        self.axis = axis
        self.heads = heads
        self.dim = dim
        head_dim = dim // heads
        assert dim % heads == 0, "dim需能被heads整除"

        self.scale = head_dim ** -0.5
        # 线性映射（用1x1卷积实现）到Q,K,V
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        # 输出映射
        self.to_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, W, H = x.shape
        q = self.to_q(x)   # B,C,W,H
        k = self.to_k(x)
        v = self.to_v(x)

        # 重排为序列： [批次*并行条数, 序列长, 头数, 维度]
        if self.axis == 'w':
            # 每行一条序列：序列长=W，并行条数=B*H
            q = q.permute(0, 3, 2, 1).contiguous().view(B*H, W, C)  # (B*H, W, C)
            k = k.permute(0, 3, 2, 1).contiguous().view(B*H, W, C)
            v = v.permute(0, 3, 2, 1).contiguous().view(B*H, W, C)
            L, N = W, B*H
        else:
            # 每列一条序列：序列长=H，并行条数=B*W
            q = q.permute(0, 2, 3, 1).contiguous().view(B*W, H, C)  # (B*W, H, C)
            k = k.permute(0, 2, 3, 1).contiguous().view(B*W, H, C)
            v = v.permute(0, 2, 3, 1).contiguous().view(B*W, H, C)
            L, N = H, B*W

        # 拆分多头: (N, L, heads, head_dim) -> (N, heads, L, head_dim)
        q = q.view(N, L, self.heads, C // self.heads).permute(0, 2, 1, 3)
        k = k.view(N, L, self.heads, C // self.heads).permute(0, 2, 1, 3)
        v = v.view(N, L, self.heads, C // self.heads).permute(0, 2, 1, 3)

        # 注意力: (N, heads, L, L)
        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (N, heads, L, head_dim)

        # 合并头并还原形状
        out = out.permute(0, 2, 1, 3).contiguous().view(N, L, C)  # (N, L, C)

        if self.axis == 'w':
            out = out.view(B, H, W, C).permute(0, 3, 2, 1)  # B,C,W,H
        else:
            out = out.view(B, W, H, C).permute(0, 3, 1, 2)  # B,C,W,H

        return self.to_out(out)  # B,C,W,H


class SingleChannelCorrelationAttention(nn.Module):
    """
    轻量全局相关性模块（单通道输入）：
    1) 先用1x1把C=1提升到小的隐维dim（默认32）
    2) 做宽向+高向两次轴向注意力（近似全局相关）
    3) 融合并压回单通道，输出尺寸仍为[B,1,W,H]

    复杂度：O(B*dim*H*W*(H+W))，显著低于完整二维自注意力的 O(B*(HW)^2)
    """
    def __init__(self, dim: int = 32, heads: int = 2, fuse: str = 'sum'):
        super().__init__()
        assert fuse in ('sum', 'avg', 'gate')
        self.fuse = fuse

        self.in_proj = nn.Conv2d(1, dim, kernel_size=1, bias=False)
        self.attn_w = AxialSelfAttention1D(dim, heads=heads, axis='w')
        self.attn_h = AxialSelfAttention1D(dim, heads=heads, axis='h')

        if fuse == 'gate':
            # 可学习融合权重
            self.alpha = nn.Parameter(torch.tensor(0.5))
        self.out_proj = nn.Conv2d(dim, 1, kernel_size=1, bias=False)
        # 轻量归一化 + 非线性，帮助稳定
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,1,W,H]  ->  y: [B,1,W,H]
        """
        B, C, W, H = x.shape
        assert C == 1, "该模块假定输入通道为1"
        feat = self.in_proj(x)                 # B,dim,W,H
        # 两个轴向注意力
        y_w = self.attn_w(feat)                # B,dim,W,H
        y_h = self.attn_h(feat)                # B,dim,W,H

        if self.fuse == 'sum':
            y = y_w + y_h
        elif self.fuse == 'avg':
            y = 0.5 * (y_w + y_h)
        else:  # gate
            y = self.alpha * y_w + (1 - self.alpha) * y_h

        y = self.act(self.norm(y + feat))      # 残差+归一化
        out = self.out_proj(y)                 # 压回1通道
        return out




# SE channel attention
class SE2D(nn.Module):
    def __init__(self, channels=2, r=2):
        super().__init__()
        hidden = max(1, channels // r)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):  # x:[B,2,H,W]
        b, c, _, _ = x.shape
        w = self.avg(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

# ECA channel attention
class ECA2D(nn.Module):
    def __init__(self, channels=2, k_size=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False)
        self.act = nn.Sigmoid()
    def forward(self, x):  # [B,2,H,W]
        y = self.avg(x)                 # [B,2,1,1]
        y = y.squeeze(-1).transpose(1,2)  # [B,1,2]
        y = self.conv(y)                  # [B,1,2]
        y = self.act(y).transpose(1,2).unsqueeze(-1)  # [B,2,1,1]
        return x * y

# gated attention
class GatedFusion2to2(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.gate = nn.Conv2d(in_channels, 2, kernel_size=1, bias=True)  # 生成两个gate

    def forward(self, x):  # [B,2,H,W]
        x1, x2 = x[:, 0:1], x[:, 1:2]
        g = torch.sigmoid(self.gate(x))  # [B,2,H,W]
        g1, g2 = g[:, 0:1], g[:, 1:2]
        y1 = g1 * x1 + (1 - g1) * x2
        y2 = g2 * x2 + (1 - g2) * x1
        return torch.cat([y1, y2], dim=1)


# depthwise attention
class DWConvFusion(nn.Module):
    def __init__(self, in_channels=2, mid=8):
        super().__init__()
        self.pw1 = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.dw  = nn.Conv2d(mid, mid, kernel_size=3, padding=1, groups=mid, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.pw2 = nn.Conv2d(mid, in_channels, kernel_size=1, bias=False)
        self.bn  = nn.BatchNorm2d(in_channels)
    def forward(self, x):  # [B,2,H,W]
        y = self.pw1(x)
        y = self.dw(y)
        y = self.act(y)
        y = self.pw2(y)
        y = self.bn(y)
        return x + y        # 残差，形状不变


# SpatialAttention
class SpatialAttentionLite(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        p = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=p, bias=False)
        self.act = nn.Sigmoid()
    def forward(self, x):  # [B,2,H,W]
        avg = torch.mean(x, dim=1, keepdim=True)          # [B,1,H,W]
        mx  = torch.max(x, dim=1, keepdim=True)[0]        # [B,1,H,W]
        a = torch.cat([avg, mx], dim=1)                   # [B,2,H,W]
        w = self.act(self.conv(a))                        # [B,1,H,W]
        return x * w






if __name__ == "__main__":
    # ===== 创建测试输入 =====
    batch_size = 4  # 批大小
    n_channels = 1  # 输入通道（SincConv 只支持 1）
    signal_length = 160000  # 每个样本长度（1 秒，16 kHz）

    x = torch.randn(batch_size, n_channels, signal_length)  # [B, 1, T]

    # ===== 创建 SincConv 层 =====
    model = SincConv_fast(
        out_channels=128,  # 滤波器数量
        kernel_size=1025,  # 每个滤波器长度
        sample_rate=16000,  # 采样率
        stride=512,
        padding=512
    )

    # ===== 前向传播 =====
    y = model(x)

    # ===== 输出信息 =====
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Conv params : out_channels={model.out_channels}, kernel_size={model.kernel_size}")






