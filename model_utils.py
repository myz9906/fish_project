import torch
from models import DynamicsModel

def load_dynamics_model(path, device='cpu'):
    """
    从 path 加载 DynamicsModel 权重 和 normalize 统计量，
    并挂到 model.stats 属性上。返回 model。
    """
    ckpt = torch.load(path, map_location=device)
    model = DynamicsModel(input_dim=4, hidden_dim=64).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # 如果包含 normalize stats，就挂到 model.stats
    stats = {}
    for key in ('in_mean','in_std','out_mean','out_std'):
        if key in ckpt:
            stats[key] = torch.tensor(ckpt[key], device=device)
    model.stats = stats
    return model