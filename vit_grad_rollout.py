import torch
import numpy as np

def grad_rollout(attentions, gradients, discard_ratio):
    result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
    with torch.no_grad():
        for attention, grad in zip(attentions, gradients):
            weights = grad
            attention_heads_fused = (attention * weights).mean(axis=1)
            attention_heads_fused[attention_heads_fused < 0] = 0

            # Drop the lowest attentions, but don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device)
            a = (attention_heads_fused + I) / 2
            a = a / a.sum(dim=-1, keepdim=True)

            result = torch.matmul(result, a)

    # Look at the total attention between the class token and the image patches
    mask = result[0, 1:].mean(0)  # Average over all heads, skip CLS token
    print("Mask shape before reshape:", mask.shape)  # 디버깅용 출력
    mask = mask[:196]  # 첫 번째 [CLS] 토큰을 제외하고 196개의 패치를 가져옴
    width = int(mask.size(0) ** 0.5)
    mask = mask.reshape(width, width).cpu().numpy()
    mask = mask / np.max(mask)
    return mask

class VITAttentionGradRollout:
    def __init__(self, model, attention_layer_name='attn_drop', discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio

        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)
                module.register_backward_hook(self.get_attention_gradient)

        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def get_attention_gradient(self, module, grad_input, grad_output):
        self.attention_gradients.append(grad_input[0].cpu())

    def __call__(self, input_tensor, category_index):
        self.attentions = []
        self.attention_gradients = []
        self.model.zero_grad()
        output = self.model(input_tensor)
        category_mask = torch.zeros(output.size()).to(output.device)
        category_mask[:, category_index] = 1
        loss = (output * category_mask).sum()
        loss.backward()

        return grad_rollout(self.attentions, self.attention_gradients, self.discard_ratio)
