import torch

# ViT 모델 로드
model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
model.eval()

# 모든 레이어 이름 출력
for name, module in model.named_modules():
    print(name)
