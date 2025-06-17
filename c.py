import torch

path = "/root/.cache/huggingface/hub/models--lukeysong--Llama-4-Scout-17B-16E-Eagle3/snapshots/fe6c99094bf16dcc74923b1d22084b6eac1303f3/draft_outputs.pt"




def rms_norm(x: torch.Tensor, eps: float = 1e-6):
    """
    This is just a functional RMSNorm without the trainable scale parameter.

    Args:
        x (torch.Tensor): input tensor to normalize
        eps (float): small value to avoid division by zero. Default: 1e-6

    Returns:
        torch.Tensor: The normalized tensor having the same shape as ``x``.

    """
    x_fp32 = x.float()
    x_normed = (
        x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    ).type_as(x)
    return x_normed

path = "/tmp/torchtune/llama4_17Bx16E/draft/input_embeds.pt"
input_embeds_rms = torch.load(path)
input_embeds_rms = input_embeds_rms.to(torch.float32)
print(rms_norm(input_embeds_rms))
