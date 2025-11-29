import torch

# 示例数据
N = 5
is_in_pipe = torch.tensor([[True], [False], [True], [False], [True]])
r_squared = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
pipe_inner_radius = 2.5

# 计算
output = torch.where(
    is_in_pipe,
    torch.abs(r_squared - pipe_inner_radius).unsqueeze(1),
    torch.tensor(-1, device='cpu')
)

print(output.shape)
