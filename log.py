import torch


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


path = "/root/.cache/huggingface/hub/models--lukeysong--Llama-4-Scout-17B-16E-Eagle3/snapshots/b1d8a533361fcf7cfef112fcea6be9113f18b071/input_embeds.pt"


input_embeds_rms = torch.load(path)
squeezed_tensor = input_embeds_rms.squeeze()
first_two_rows = squeezed_tensor[:2]  # 只取前两行
print("First two rows:", first_two_rows)

print("input fuck  ", rms_norm(first_two_rows))


