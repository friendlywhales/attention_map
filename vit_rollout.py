import torch
import numpy as np


# extract_attention_map: 주어진 모듈에서 Attention 맵을 추출하는 함수 정의.
def extract_attention_map(module, input, output):
    # qkv: 출력으로부터 쿼리, 키, 값을 포함한 텐서를 할당함. 텐서 크기는 (batch_size, num_patches + 1, 3 * embed_dim)
    qkv = output
    # qkv 텐서의 크기를 분리하여 배치 크기, 패치 수, 총 임베딩 차원수를 각각 저장
    batch_size, num_patches, total_dim = qkv.size()
    # 총 임베딩 차원을 3으로 나누어 각 쿼리, 키, 값의 임베딩 차원을 구하기
    embed_dim = total_dim // 3
    # 주어진 Attention 헤드 수를 12로 설정
    num_heads = 12
    # 각 헤드의 차원을 계산
    head_dim = embed_dim // num_heads

    # qkv 텐서를 쿼리(q), 키(k), 값(v)으로 분리
    q, k, v = qkv[:, :, :embed_dim], qkv[:, :, embed_dim:2*embed_dim], qkv[:, :, 2*embed_dim:]
    # q와 k를 재구성하고 차원 순서를 변경
    q = q.reshape(batch_size, num_patches, num_heads, head_dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, num_patches + 1, head_dim)
    k = k.reshape(batch_size, num_patches, num_heads, head_dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, num_patches + 1, head_dim)

    # 쿼리와 키의 점곱으로 Attention 스코어를 계산한다. 각 스코어는 head_dim의 제곱근으로 나눈다
    attention_scores = torch.einsum("bhqd, bhkd -> bhqk", q, k) / (head_dim ** 0.5)  # (batch_size, num_heads, num_patches + 1, num_patches + 1)
    # Softmax 함수를 사용하여 Attention 확률을 계산함
    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

    # 계산된 Attention 확률을 반환
    return attention_probs


# rollout: Attention 맵을 누적하는 함수 정의.
def rollout(attentions, discard_ratio, head_fusion):
    # 결과 텐서를 초기화합니다. 초기값은 단위 행렬입니다.
    result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)

    # torch.no_grad() 블록 안에서 텐서의 연산 그래프 추적을 비활성화합니다.
    with torch.no_grad():
        # Attention 리스트 순회
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise ValueError("Attention head fusion type not supported")

            # 주어진 head_fusion 방법에 따라 Attention 헤드를 병합
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            flat[0, indices] = 0

            # Attention 값을 평탄화하고 가장 낮은 Attention 값을 선택적으로 무시
            I = torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device)
            a = (attention_heads_fused + I) / 2
            a = a / a.sum(dim=-1, keepdim=True)

            # Attention 값과 단위 행렬의 평균을 계산하고 정규화
            result = torch.matmul(a, result)

            # 결과 텐서와 현재 Attention 값을 행렬 곱셈합니다.
            mask = result[0, 1:].mean(0)  # Average over all heads, skip CLS token
            print("Mask shape before reshape:", mask.shape)  # 디버깅용 출력
            mask = mask[:196]  # 첫 번째 [CLS] 토큰을 제외하고 196개의 패치를 가져옴 (14x14 라서 196개)
            width = int(mask.size(0) ** 0.5)
            mask = mask.reshape(width, width).cpu().numpy()
            mask = mask / np.max(mask)
        return mask

# VITAttentionRollout 클래스 생성자를 정의합니다. 모델, 헤드 병합 방법, 디스카드 비율을 설정합니다.
class VITAttentionRollout:
    def __init__(self, model, head_fusion="mean", discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self.attentions = []

        # 모델의 모든 모듈을 순회하면서 qkv 이름을 가진 torch.nn.Linear 모듈에 대해 Forward Hook을 등록합니다.
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'qkv' in name:
                module.register_forward_hook(self.get_attention)
               #print(f"Hook registered to layer: {name}")

    # Attention 맵을 추출하고 리스트에 추가하는 get_attention 함수를 정의합니다.
    def get_attention(self, module, input, output):
        attention_probs = extract_attention_map(module, input, output)
        self.attentions.append(attention_probs.detach().cpu())

    # 모델을 실행하고 Attention 맵을 누적하여 반환하는 __call__ 메서드를 정의합니다.
    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            _ = self.model(input_tensor)

        if not self.attentions:
            raise ValueError("No attention values were collected. Check if the attention layer name is correct.")

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)
