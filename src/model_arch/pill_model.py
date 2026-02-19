import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# --- 1. Class เสริม (เก็บไว้เผื่ออนาคตอยากใช้ แต่รอบนี้ยังไม่ได้ใช้) ---
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class CBAM(nn.Module):
    # (เก็บ Class CBAM ไว้เหมือนเดิม เผื่ออนาคต)
    def __init__(self, planes):
        super(CBAM, self).__init__()
        # ... (ใส่ Code CBAM เดิมของคุณตรงนี้ หรือถ้าไม่ได้ใช้ ลบออกก็ได้) ...
        pass 

# --- 2. The FINAL PillModel (ตรงปก 100%) ---
class PillModel(nn.Module):
    def __init__(self, num_classes=1000, model_name='convnext_small', embed_dim=512, use_cbam=False):
        super(PillModel, self).__init__()
        
        # 1. Load Backbone
        self.backbone = timm.create_model(model_name, pretrained=True)
        
        # Check num_features
        if hasattr(self.backbone, 'num_features'):
            n_features = self.backbone.num_features
        else:
            n_features = self.backbone.fc.in_features 
            
        # Remove original head
        self.backbone.reset_classifier(0)
        
        # 2. CBAM (ปิดไว้ตาม Default)
        self.use_cbam = use_cbam
        if self.use_cbam:
            self.attention = CBAM(n_features) # ต้องมี Class CBAM ถ้ารันบรรทัดนี้
        
        # 3. 🔥 จุดแก้: เปลี่ยนกลับเป็น Standard Pooling (AvgPool)
        # เพราะไฟล์ Weight คุณไม่มีค่า p ของ GeM
        self.pooling = nn.AdaptiveAvgPool2d((1, 1)) 
        
        # 4. Projection Layers (ตัวแปลงร่าง)
        self.bn = nn.BatchNorm1d(n_features)
        self.fc = nn.Linear(n_features, embed_dim)
        self.bn_emb = nn.BatchNorm1d(embed_dim)
        
        # Head (มีไว้กัน Error แต่ไม่ได้ใช้)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        
        if self.use_cbam:
            features = self.attention(features)
            
        features = self.pooling(features).flatten(1)
        
        features = self.bn(features)
        features = self.fc(features)
        features = self.bn_emb(features)
        
        return features