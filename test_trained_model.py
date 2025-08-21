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
trail_id   =  3                          # 选哪一条 trail
dt         = 1.0 / 100.0                  # 100 Hz 数据

# -------------------------------------------------------------------
#  1) 读取并挑出单条 trail
# -------------------------------------------------------------------
df = pd.read_csv(csv_path)
sub = df[df['trail']==trail_id].reset_index(drop=True)
xv, yv = sub['Virtualfish x'].values, sub['Virtualfish y'].values
xr, yr = sub['Realfish x'].values,    sub['Realfish y'].values
N = len(xv)

# -------------------------------------------------------------------
#  2) 计算速度和朝向（最简单的差分+atan2）
# -------------------------------------------------------------------
# 一阶差分速度
vvx = np.empty_like(xv)
vvy = np.empty_like(yv)
vrx = np.empty_like(xr)
vry = np.empty_like(yr)
vvx[0] = (xv[1] - xv[0]) / dt
vvy[0] = (yv[1] - yv[0]) / dt
vrx[0] = (xr[1] - xr[0]) / dt
vry[0] = (yr[1] - yr[0]) / dt
vvx[1:] = (xv[1:] - xv[:-1]) / dt
vvy[1:] = (yv[1:] - yv[:-1]) / dt
vrx[1:] = (xr[1:] - xr[:-1]) / dt
vry[1:] = (yr[1:] - yr[:-1]) / dt

# 2) 滑动平均（window=5）
window = 5
kernel = np.ones(window, dtype=np.float32) / window
def smooth(arr):
    conv = np.convolve(arr, kernel, mode='valid')
    prefix = np.cumsum(arr[:window-1]) / np.arange(1, window)
    return np.concatenate([prefix, conv])
vvx_s = smooth(vvx)
vvy_s = smooth(vvy)
vrx_s = smooth(vrx)
vry_s = smooth(vry)



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
# 4) 从 t=0 开始，多步 rollout，记录每一步网络预测的 真鱼(sr_pred)
# -------------------------------------------------------------------
sv      = torch.zeros(4, device=device)   # 虚拟鱼状态占位
sr_pred = torch.zeros(4, device=device)   # 预测的真鱼状态占位

pred_xy = []
real_xy = []
virtual_xy = []

for t in range(N-1):
    # 构造当前状态向量： [x, y, vx, vy, θ]
    sv = torch.tensor([xv[t], yv[t], vvx_s[t], vvy_s[t]], dtype=torch.float32,
                      device=device)
    sr_pred = torch.tensor([xr[t], yr[t], vrx_s[t], vry_s[t]], dtype=torch.float32,
                           device=device)
    a_v = torch.tensor([vvx_s[t], vvy_s[t]], dtype=torch.float32, device=device)

    # 网络一步预测
    sv, sr_pred = model.step(sv, sr_pred, a_v, dt)

    # 存 t+1 时刻的预测 pos & 真实 pos
    pred_xy.append([sr_pred[0].item(), sr_pred[1].item()])
    real_xy.append([xr[t+1], yr[t+1]])
    virtual_xy.append([sv[0].item(), sv[1].item()])

pred_xy = np.array(pred_xy)
real_xy = np.array(real_xy)
virtual_xy = np.array(virtual_xy)

# -------------------------------------------------------------------
# 5) 可视化对比
# -------------------------------------------------------------------
# plt.figure(figsize=(6,5))
# plt.plot(real_xy[:,0], real_xy[:,1], 'o-', label='Real Fish Trajectory')
# plt.plot(pred_xy[:,0], pred_xy[:,1], 'x--', label='Predicted Real Fish Trajectory')
# plt.plot(virtual_xy[:,0], virtual_xy[:,1], 'd-.', label='Virtual Fish Trajectory')

# # plt.title(f'Trail {trail_id}：数据驱动模型预测 vs 真实')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)
# plt.show()


# 用热力图表示轨迹随时间的变化情况
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

# 假设 real_xy, pred_xy, virtual_xy 都是 (N,2) 的 numpy array
N = real_xy.shape[0]
# 用于 LineCollection 的段着色：N-1 条线段
timesteps_seg = np.arange(N-1)
# 用于 scatter 的点着色：N 个点
timesteps_pts = np.arange(N)

# 1) 准备统一的 cmap 和 norm
cmap = plt.get_cmap('viridis')
norm = plt.Normalize(vmin=0, vmax=N-1)

# 2) 将每条轨迹拆成线段
def make_segments(xy):
    pts = xy.reshape(-1, 1, 2)                # (N,1,2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)  # (N-1,2,2)

seg_real = make_segments(real_xy)
seg_pred = make_segments(pred_xy)
seg_virt = make_segments(virtual_xy)

# 3) 各自创建 LineCollection，共用 cmap 和 norm，用不同 linestyle 区分
lc_real = LineCollection(seg_real, cmap=cmap, norm=norm,
                         linewidth=2, linestyle='-')
lc_pred = LineCollection(seg_pred, cmap=cmap, norm=norm,
                         linewidth=2, linestyle='--')
lc_virt = LineCollection(seg_virt, cmap=cmap, norm=norm,
                         linewidth=2, linestyle='-.')

# 将时间步映射到颜色
lc_real.set_array(timesteps_seg)
lc_pred.set_array(timesteps_seg)
lc_virt.set_array(timesteps_seg)

# 4) 画图
fig, ax = plt.subplots(figsize=(6,5))
ax.add_collection(lc_real)
ax.add_collection(lc_pred)
ax.add_collection(lc_virt)

# Overlay markers (同样用 cmap/norm 来着色)
ax.scatter(real_xy[:,0], real_xy[:,1],
           c=timesteps_pts, cmap=cmap, norm=norm,
           marker='o', s=60, edgecolor='k')
ax.scatter(pred_xy[:,0], pred_xy[:,1],
           c=timesteps_pts, cmap=cmap, norm=norm,
           marker='x', s=60)
ax.scatter(virtual_xy[:,0], virtual_xy[:,1],
           c=timesteps_pts, cmap=cmap, norm=norm,
           marker='d', s=60, edgecolor='k')

# 6) 画 proxy artists 做图例
legend_items = [
    Line2D([0],[0], color='gray', marker='o', linestyle='', label='Real Fish'),
    Line2D([0],[0], color='gray', marker='x', linestyle='', label='Predicted Fish'),
    Line2D([0],[0], color='gray', marker='d', linestyle='', label='Virtual Fish'),
]
ax.legend(handles=legend_items, loc='upper left')

# 7) 轴设置
all_x = np.hstack([real_xy[:,0], pred_xy[:,0], virtual_xy[:,0]])
all_y = np.hstack([real_xy[:,1], pred_xy[:,1], virtual_xy[:,1]])
# ax.set_xlim(all_x.min(), all_x.max())
# ax.set_ylim(all_y.min(), all_y.max())
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)

# 8) 统一 colorbar
cbar = fig.colorbar(lc_real, ax=ax)
cbar.set_label('Time step')

plt.show()