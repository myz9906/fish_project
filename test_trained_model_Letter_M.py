import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------------------------
#  配置
# -------------------------------------------------------------------
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'DynamicsModel_without_theta.pth'          # <- 改成你的模型路径
csv_path = Path('fish_data') / 'Fish-system-identification' / 'Step_angle_0.csv'
trail_id   =  47                          # 选哪一条 trail
dt         = 1.0 / 100.0                  # 100 Hz 数据

# -------------------------------------------------------------------
#  helper：直接拷贝你现成的 M 字生成函数
# -------------------------------------------------------------------
def make_letter_target_M(H, dt, device):
    """
    生成 M 字的 (H,4) 目标张量 [x, y, vx, vy]，
    全程在线性段上插值（不再依赖三次样条）。
    """
    # 0) 在 [0,1] 上等距离取 H 个点
    u = torch.linspace(0.0, 1.0, H, device=device)

    # 1) 准备存放 x,y
    xs = torch.empty(H, device=device)
    ys = torch.empty(H, device=device)

    # 2) 四段拼接（每段长度 0.25）
    mask1 = u <= 0.25
    t1 = u[mask1] / 0.25
    xs[mask1] = -1.0 + 0.5 * t1
    ys[mask1] =     0.0 + 1.0 * t1

    mask2 = (u > 0.25) & (u <= 0.50)
    t2 = (u[mask2] - 0.25) / 0.25
    xs[mask2] = -0.5 + 0.5 * t2
    ys[mask2] =  1.0 - 1.0 * t2

    mask3 = (u > 0.50) & (u <= 0.75)
    t3 = (u[mask3] - 0.50) / 0.25
    xs[mask3] =  0.0 + 0.5 * t3
    ys[mask3] =  0.0 + 1.0 * t3

    mask4 = u > 0.75
    t4 = (u[mask4] - 0.75) / 0.25
    xs[mask4] =  0.5 + 0.5 * t4
    ys[mask4] =  1.0 - 1.0 * t4

    # 3) 缩放到 0.1 大小
    xs *= 0.1
    ys *= 0.1

    # 4) 计算每步的速度向量 vx, vy
    dpos = torch.stack([xs, ys], dim=1)           # (H,2)
    # 差分得到位移，再除以 dt 得到速度向量
    vels = (dpos[1:] - dpos[:-1]) / dt            # (H-1,2)
    zero2 = torch.zeros(1, 2, device=device)      # 第一步速度用 0
    vels_full = torch.cat([zero2, vels], dim=0)   # (H,2)

    # 5) 构造 (H,4) 目标张量：x, y, vx, vy
    targ = torch.zeros(H, 4, device=device)
    targ[:,0] = xs
    targ[:,1] = ys
    targ[:,2:] = vels_full

    return targ



# -------------------------------------------------------------------
# 3) 加载你训练好的“数据驱动”DynamicsModel
# -------------------------------------------------------------------
class LearnedDynamicsModel:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        from model_utils import load_dynamics_model # <- 改成实际模块名
        self.model = load_dynamics_model(model_path, device=device)
        self.model.eval()

    def step(self, s_v, s_r, a_v, dt):
        from sim import step_dynamics_without_theta      # <- 改成实际模块名
        return step_dynamics_without_theta(s_v, s_r, a_v, self.model, dt)

model = LearnedDynamicsModel(model_path, device=device)

# -------------------------------------------------------------------
#  4) 下面这一段切换到 M 字轨迹
# -------------------------------------------------------------------
H = 500  # 你想跑多少步就设多大
traj_M = make_letter_target_M(H, dt, device)  # (H,4)

# 初始化
sv      = torch.zeros(4, device=device)  # [x,y,vx,vy,theta]
sr_pred = torch.zeros(4, device=device)
sr_pred[0] = -0.1
sr_pred[1] = -0.03

# 先把 sv 的初始位置放到 M 字第 0 点
sv[0:2] = traj_M[0,0:2]
# θ 初始化为 0 就好
pred_xy    = []
virtual_xy = []

# 滚动预测
for t in range(H-1):
    # 取出当前命令：x*,y*,v* 以及（可选的）theta*
    x_cmd, y_cmd, vx_cmd, vy_cmd = traj_M[t]
    a_v = torch.tensor([vx_cmd, vy_cmd], device=device)
    # 推网络
    sv, sr_pred = model.step(sv, sr_pred, a_v, dt)
    pred_xy.append([sr_pred[0].item(), sr_pred[1].item()])
    virtual_xy.append([sv[0].item(),    sv[1].item()])

pred_xy    = np.array(pred_xy)
virtual_xy = np.array(virtual_xy)


# -------------------------------------------------------------------
#  5) 可视化
# -------------------------------------------------------------------
plt.figure(figsize=(6,5))
# M 字虚拟鱼
plt.plot(virtual_xy[:,0], virtual_xy[:,1], 'r-',
         label='Virtual Fish (M)')
# 网络预测的真鱼
plt.plot(pred_xy[:,0], pred_xy[:,1], 'b--',
         label='Predicted Real Fish')
plt.legend()
plt.xlabel('x'); plt.ylabel('y'); plt.grid(True)
plt.title('M Letter Tracking by Learned Dynamics')
plt.show()