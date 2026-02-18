import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ---------------------------------------------------------
# 1. GeM Pooling (Senior Tip: ดีกว่า AvgPool สำหรับ Image Retrieval)
# ---------------------------------------------------------
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channel, H, W)
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

# ---------------------------------------------------------
# 2. CBAM Attention (ตัวช่วยให้โมเดล "เพ่ง" รายละเอียด)
# ---------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

# ---------------------------------------------------------
# 3. ArcFace Head (เหมือนเดิม เพราะดีอยู่แล้ว)
# ---------------------------------------------------------
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m); self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1 - cosine**2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

# ---------------------------------------------------------
# 4. PillModel Ultimate (รวมร่าง)
# ---------------------------------------------------------
class PillModel(nn.Module):
    def __init__(self, num_classes, model_name='convnext_small', embed_dim=512, dropout=0.0):
        super().__init__()
        # 1. Load Backbone without Pooling (เราจะทำเอง)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        
        # Auto-detect channels
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy) # (B, C, H, W)
            in_channels = features.shape[1]
            
        # 2. Add Attention (ช่วยแยกยารูปร่างเหมือนกัน)
        self.attention = CBAM(in_channels)
        
        # 3. Add GeM Pooling (ดึง Feature เด่น)
        self.pooling = GeM()
        
        self.bn = nn.BatchNorm1d(in_channels)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, embed_dim)
        self.bn_emb = nn.BatchNorm1d(embed_dim)
        
        # 4. ArcFace Head
        self.head = ArcMarginProduct(embed_dim, num_classes)

    def forward(self, x, labels=None):
        # Backbone -> (B, C, H, W)
        feat_map = self.backbone(x)
        
        # Attention Refinement -> มองหารอยปั๊มบนยา
        feat_map = self.attention(feat_map)
        
        # Pooling -> (B, C, 1, 1) -> (B, C)
        feat = self.pooling(feat_map).flatten(1)
        
        feat = self.bn(feat)
        feat = self.drop(feat)
        
        # Embedding
        emb = self.bn_emb(self.fc(feat))
        
        if labels is not None:
            return self.head(emb, labels)
        
        return emb