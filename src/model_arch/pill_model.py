import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm 

# ---------------------------------------------------------
# 2. ArcFace Head (ยกมาจาก architecture.py เดิมเป๊ะๆ)
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
# 3. PillModel (Main Class)
# ---------------------------------------------------------
class PillModel(nn.Module):
    def __init__(self, num_classes=1000, model_name='convnext_small', embed_dim=512, dropout=0.0):
        super().__init__()
        # Load Pretrained Backbone (timm)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # Auto-detect input features dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            in_features = self.backbone(dummy).shape[1]
            
        self.bn = nn.BatchNorm1d(in_features)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, embed_dim)
        self.bn_emb = nn.BatchNorm1d(embed_dim)
        
        # ArcFace Head
        self.head = ArcMarginProduct(embed_dim, num_classes)

    def forward(self, x, labels=None):
        # 1. Feature Extraction
        feat = self.backbone(x)
        feat = self.bn(feat)
        feat = self.drop(feat)
        
        # 2. Embedding (Vector 512)
        emb = self.bn_emb(self.fc(feat))
        
        # 3. Logic แยกโหมด
        if labels is not None:
            return self.head(emb, labels)
        
        # Inference Mode: Return Vector 🔥
        return emb