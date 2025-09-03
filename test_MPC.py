import torch
import math
import matplotlib.pyplot as plt
import time

import numpy as np
from scipy.interpolate import CubicSpline

# 本地示例：测试 ActionPlanner 是否能追踪简单目标
# 请先确保 planner.py 中的 ActionPlanner 已经按 Torch 张量版实现

start_time = time.time()

# 1. 定义一个最简单的线性动力学模型，这个里面 real fish的位置就是virtual fish的位置
class LinearDynamicsModel:
    def step(self, s_v, s_r, a_v, dt):
        # leader 更新
        pos = s_v[:2] + a_v * dt
        vel = a_v
        theta = s_v[4]   # 这个theta 貌似不会更新，一直都是0
        sv = torch.cat([pos, vel, theta.unsqueeze(0)], dim=0)
        # follower 完全跟随 leader
        sr = sv.clone()
        return sv, sr


# 用neural network 设计出来的 dynamics model 

from model_utils import load_dynamics_model
from sim import step_dynamics  # 原来用于 NN 的推进函数


class LearnedDynamicsModel:
    """ 原来加载 .pt 模型 的实现 """
    def __init__(self, model_path: str, rx=0.07, ry=0.07, device: str = 'cpu'):
        self.device = device
        self.model  = load_dynamics_model(model_path, device=device)
        self.model.eval()
        self.rx, self.ry = rx, ry

    def step(self, s_v, s_r, a_v, dt):
        # 直接调用原来的 step_dynamics
        return step_dynamics(s_v, s_r, a_v, self.model, dt)
    


    
# BioPD model 
class BioPDDynamicsModel:
    """
    Leader 由 a_v 推进；Follower 用 BioPD 控制律：
      e = x_F - x_L
      de = (e - prev_e) / dt
      v = -(Kp*e + Kd*de) * exp( - e^2 / (2 r^2) )
      cutoff: if |e|>r then v=0
    """
    def __init__(self, Kp=1.0, Kd=0.5, rx=0.1, ry=0.1, device='cpu'):
        self.Kp, self.Kd = Kp, Kd
        self.rx, self.ry = rx, ry
        self.device = device
        # prev error for x and y (initialized zero)
        self.prev_e = torch.zeros(2, device=self.device)
        # 预先计算“最大可达步长” (一阶积分近似)，用于 reachable set
        # BioPD 的峰值速度 ≈ Kp * r * exp(-½)
        # 把 -0.5 包成 Tensor
        half = torch.tensor(-0.5, device=device)
        self.v_xmax = self.Kp * self.rx * torch.exp(half)
        self.v_ymax = self.Kp * self.ry * torch.exp(half)

    def step(self, s_v: torch.Tensor, s_r: torch.Tensor, a_v: torch.Tensor, dt: float):
        # —— 1) 更新 leader —— #
        # state = [x, y, vx, vy, θ]
        sv = s_v.clone()
        # apply commanded velocity a_v
        vx, vy = a_v[0], a_v[1]
        sv[2:4] = a_v

        sv[0:2] = sv[0:2] + a_v * dt


        # sv[4]   = torch.atan2(a_v[1], a_v[0])     # 不能直接atan2，因为会nan，需要分情况考虑

        # 然后计算朝向：如果 vx ≠ 0 用 atan2，一旦 vx=0 则直接 π/2·sign(vy)，
        # 如果 vx=vy=0 再设为 0
        # 注意 eps 防止浮点比较问题
        eps = 1e-8
        vx_zero = vx.abs() < eps
        vy_zero = vy.abs() < eps

        # # 用 atan2 计算初始角度（会在 0/0 时产生 nan）
        # theta_raw = torch.atan2(vy, vx)

        # # 当 vx=0 且 vy≠0 时，用 ±π/2
        # theta_x0 = torch.sign(vy) * (math.pi / 2)

        # # 把这三种情况合并
        # theta = torch.where(vx_zero,     # vx=0, vy≠0
        #                     theta_x0,
        #                     theta_raw)              # else 用 atan2


        theta = torch.zeros_like(vx)
        # 非零方向
        idx = (vx.abs()) > eps
        theta[idx] = torch.atan2(vy[idx], vx[idx])
        # vx≈0, vy≠0
        idx2 = (vx.abs() < eps)
        theta[idx2] = torch.sign(vy[idx2]) * (math.pi/2)
        # 其余（0,0）保持 0


        sv[4] = theta



        # —— 2) BioPD 更新 follower —— #
        # current positions
        xL, yL = sv[0], sv[1]
        xF, yF = s_r[0], s_r[1]

        # error and its derivative
        e = torch.stack([xF - xL, yF - yL])           # (2,)
        de = (e - self.prev_e) / dt
        # save for next step (detach to avoid huge graph)
        self.prev_e = e.detach()

        # compute raw PD term
        pd = self.Kp * e + self.Kd * de               # (2,)
        # gaussian attenuation
        atten = torch.exp(-0.5 * (e**2) / torch.tensor([self.rx**2, self.ry**2], device=self.device))
        v_r = - pd * atten                             # (2,)

        # v_r = - pd

        # cutoff beyond r_x / r_y
        
        # mask = torch.tensor([abs(e[0]) <= self.rx,
        #                      abs(e[1]) <= self.ry],
        #                     device=self.device)
        # v_r = v_r * mask.to(v_r.dtype)

        # build new follower state
        sr = s_r.clone()
        sr[2:4] = v_r
        sr[0:2] = sr[0:2] + v_r * dt

        # follower 朝向同样处理
        vx2, vy2 = v_r[0], v_r[1]
        vx_zero2 = vx2.abs() < eps
        vy_zero2 = vy2.abs() < eps


        # theta2_raw = torch.atan2(vy2, vx2)
        # theta2_x0  = torch.sign(vy2) * (math.pi / 2)

        # theta2 = torch.where(vx_zero2, theta2_x0, theta2_raw)

        theta2 = torch.zeros_like(vx2)
        # 非零方向
        idx = (vx2.abs()) > eps
        theta2[idx] = torch.atan2(vy2[idx], vx2[idx])
        # vx≈0, vy≠0
        idx2 = (vx2.abs() < eps)
        theta2[idx2] = torch.sign(vy2[idx2]) * (math.pi/2)

        sr[4] = theta2
        

        return sv, sr
    

# 2. 实例化 Planner
from planner import ActionPlanner  # 确保 planner.py 在同一目录
dt = 1/100.

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# model = LinearDynamicsModel()
model = BioPDDynamicsModel(Kp=2.3, Kd=0.56, rx=0.07, ry=0.07, device=device)  # 原版模型   
# model = BioPDDynamicsModel(Kp=10.3, Kd=0.56, rx=0.05, ry=0.05, device=device)  # 自己用来测试的模型
# model = LearnedDynamicsModel(model_path = 'DynamicsModel.pth')

# 这个是MPC的planner
planner = ActionPlanner(
    dynamics_model=model,
    device=device,
    horizon=500,
    mpc_iters=500,      # 增加MPC的iteration 貌似能够让误差变小
    MPC_step = 500,
    lr=1e-1,
    v_max=10.0,
    dt=dt
)

# 这个是CEM的planner:
# action_planner = ActionPlanner(
#     model_path   = dynamics_model_path,
#     device       = device,
#     horizon      = args.planner_horizon,
#     pop_size     = 300,
#     elite_frac   = 0.2,
#     iterations   = 5,
#     dt           = 1/30.
# )

# 初始状态

s_v = torch.zeros(5, device=device)
s_r = torch.zeros(5, device=device)
# s_r[0] = -0.1   # x 方向偏移


# # 这个是 一下子算出所有的action 

def test_tracking(traj_target):
    a0, full_seq = planner.plan(s_v, s_r, traj_target)
    sv, sr = s_v.clone(), s_r.clone()

    hist_v = [[sv[0].item(), sv[1].item()]]
    hist_r = [[sr[0].item(), sr[1].item()]]
    for k in range(planner.MPC_step):
        # 关键：把 numpy -> torch 的时候指定 device
        a_v = torch.tensor(full_seq[k], device=device, dtype=torch.float32)
        sv, sr = model.step(sv, sr, a_v, dt)
        hist_v.append([sv[0].item(), sv[1].item()])
        hist_r.append([sr[0].item(), sr[1].item()])
    return np.array(hist_v), np.array(hist_r)


# 这个是 receding-horizon ，用滚动时域的方法求轨迹，目前没有考虑 warm-start，先这样看下情况，后续用 warm-start来整
# warm-start（用上一步的 a_seq 作为这次 plan 的初值），可以把 plan 签名改成接收一个上次的 a_seq_init，然后在每次循环结束后保存下来，下次传给它。这样可以大幅加速收敛。

# def test_tracking(traj_target):
#     """
#     traj_target: (T,4)      —— 整条轨迹，长度 T
#     planner.MPC_step: int        —— 总共想执行多少步（可以 = T，也可以 <T）
#     """
#     sv, sr = s_v.clone(), s_r.clone()
#     hist_v = [(sv[0].item(), sv[1].item())]
#     hist_r = [(sr[0].item(), sr[1].item())]

#     for t in range(planner.MPC_step):
#         # 1) 构造本次优化要看的“未来 H 步”目标
#         #    如果剩余不够 H，就截到末尾
#         start = t
#         end   = min(t + planner.H, traj_target.shape[0])
#         target_slice = traj_target[start:end]

#         # 2) MPC 只看这段 target_slice，返回第一个动作 a0
#         a0, _ = planner.plan(sv, sr, target_slice)

#         # 3) 只执行 a0
#         a_v = torch.tensor(a0, device=device, dtype=torch.float32)
#         sv, sr = model.step(sv, sr, a_v, dt)

#         # 4) 记录
#         hist_v.append((sv[0].item(), sv[1].item()))
#         hist_r.append((sr[0].item(), sr[1].item()))

#     return np.array(hist_v), np.array(hist_r)

# # 3 任务. 固定点追踪

# H = planner.H
# fixed_target = torch.tensor([[1.0, 1.0, 0.0, 0.0]] * H)
# print("\n-- Fixed Point Test --")
# a0_fixed, pos_fixed_v,  pos_fixed_r= test_tracking(fixed_target)

# # # 4. 直线追踪：从 (0,0) 到 (1,0)
# line_x = torch.linspace(1/H, 1.0, H, device=device)
# # y 坐标全是 y0
# line_y = torch.full((H,), 2, device=planner.device)  # 2指的是y坐标为2

# zeros  = torch.zeros((H,),        device=device)

# # 直接沿维度 1 堆成 (H,4)
# line_target = torch.stack([line_x,      # x
#                            line_y,      # y
#                            zeros,       # v=0
#                            zeros], dim=1)  # θ=0
# print("\n-- Straight Line Test --")
# a0_line, pos_line_v,  pos_line_r = test_tracking(line_target)

# # 5. 正弦曲线追踪： x 轴 0->1, y = sin(pi * x)
# sine_x = torch.linspace(1/H, 1.0, H, device=device)
# sine_y = torch.sin(math.pi * sine_x) + 0.5

# # 2) 零张量
# zeros = torch.zeros(H, device=device)

# # 3) 直接沿第 1 维堆成 (H,4)
# sine_target = torch.stack([sine_x,     # x
#                            sine_y,     # y
#                            zeros,      # v = 0
#                            zeros],     # θ = 0
#                           dim=1)       # 变成 (H,4)

# print("\n-- Sine Wave Test --")
# a0_sine, pos_sine_v,  pos_sine_r = test_tracking(sine_target)


# elapsed = time.time() - start_time
# print(f"Training completed in {elapsed:.1f} seconds.")

# # 6. 绘图比较
# cases = [
#     ("Fixed Point", fixed_target, pos_fixed_r),
#     ("Straight Line", line_target, pos_line_r),
#     ("Sine Wave", sine_target, pos_sine_r),
# ]

# for name, target, positions in cases:
#     xs = [p[0] for p in positions]
#     ys = [p[1] for p in positions]
#     # 目标
#     if hasattr(target, 'cpu'):
#         tx = target[:,0].cpu().numpy()
#         ty = target[:,1].cpu().numpy()
#     else:
#         tx = target[:,0]
#         ty = target[:,1]

#     plt.figure()
#     plt.plot(xs, ys, 'o-', label=f'{name} Track')
#     plt.plot(tx, ty, 'k--', label=f'{name} Target')
#     plt.title(f"MPC Tracking: {name}")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.legend()
#     plt.grid(True)
#     plt.show()


# # 画跟踪误差
# # 时间轴
# times = np.arange(pos_line_v.shape[0]) * dt

# # 跟踪直线的误差

# # 1) 误差随时间
# errors = np.linalg.norm(pos_line_v - pos_line_r, axis=1)
# plt.figure()
# plt.plot(times, errors, '-')
# plt.xlabel('Time (s)')
# plt.ylabel('Euclidean error')
# plt.title('Leader–Follower Position Error vs Time')
# plt.grid(True)
# plt.show()

# # 2) x 方向位置随时间
# plt.figure()
# plt.plot(times, pos_line_v[:,0], label='Virtual Fish (x)')
# plt.plot(times, pos_line_r[:,0], label='Real Fish    (x)')
# plt.xlabel('Time (s)')
# plt.ylabel('x position')
# plt.title('x Position vs Time')
# plt.legend()
# plt.grid(True)
# plt.show()

# # 3) y 方向位置随时间
# plt.figure()
# plt.plot(times, pos_line_v[:,1], label='Virtual Fish (y)')
# plt.plot(times, pos_line_r[:,1], label='Real Fish    (y)')
# plt.xlabel('Time (s)')
# plt.ylabel('y position')
# plt.title('y Position vs Time')
# plt.legend()
# plt.grid(True)
# plt.show()

# # 跟踪正弦曲线的误差：

# # 1) 误差随时间
# errors = np.linalg.norm(pos_sine_v - pos_sine_r, axis=1)
# plt.figure()
# plt.plot(times, errors, '-')
# plt.xlabel('Time (s)')
# plt.ylabel('Euclidean error')
# plt.title('Leader–Follower Position Error vs Time')
# plt.grid(True)
# plt.show()

# # 2) x 方向位置随时间
# plt.figure()
# plt.plot(times, pos_sine_v[:,0], label='Virtual Fish (x)')
# plt.plot(times, pos_sine_r[:,0], label='Real Fish    (x)')
# plt.xlabel('Time (s)')
# plt.ylabel('x position')
# plt.title('x Position vs Time')
# plt.legend()
# plt.grid(True)
# plt.show()

# # 3) y 方向位置随时间
# plt.figure()
# plt.plot(times, pos_sine_v[:,1], label='Virtual Fish (y)')
# plt.plot(times, pos_sine_r[:,1], label='Real Fish    (y)')
# plt.xlabel('Time (s)')
# plt.ylabel('y position')
# plt.title('y Position vs Time')
# plt.legend()
# plt.grid(True)
# plt.show()



# 任务：生成不同的字母

def make_letter_target(letter, H, dt, device):
    """
    生成 M/P 字母的 (H,4) 目标张量 [x,y,v,θ].
    """
    # 1) 控制点
    if letter == 'M':
        pts = [(-1.0,0.0), (-0.5,1.0), (0.0,0.0), (0.5,1.0), (1.0,0.0)]
    elif letter == 'P':
        pts = [
        (-1.0, 0.0),    # 底部
        (-1.0, 1.0),    # 往上到接近圈的起点
        (-0.5, 1.0),    # 圈的左侧
        (-0.5, 0.5),    # 圈的右侧
        (-1.0, 0.5),    # 回到圈的起点
        (-1.0, 0.0)     # 回到底部
    ]
    else:
        raise ValueError("只支持 'M' 或 'P'")

    # 2) 缩放、样条
    scale = 0.1
    xs_knots = np.array([scale*x for x,y in pts])
    ys_knots = np.array([scale*y for x,y in pts])
    t_knots  = np.linspace(0, 1, len(pts))
    csx = CubicSpline(t_knots, xs_knots)
    csy = CubicSpline(t_knots, ys_knots)

    zero = torch.zeros(1, device=device, dtype=torch.float32)

    # 3) 离散化到 H 点
    us = np.linspace(0, 1, H)
    xs = torch.tensor(csx(us), device=device, dtype=torch.float32)
    ys = torch.tensor(csy(us), device=device, dtype=torch.float32)

    # 4) 计算每步速度 v_t = distance / dt
    #    dpos shape = (H-1,2)
    dpos = torch.stack([xs, ys], dim=1)[1:] - torch.stack([xs, ys], dim=1)[:-1]
    speeds = torch.norm(dpos, dim=1) / dt           # (H-1,)
    v_t = torch.cat([zero, speeds], dim=0)    # (H,)

    # 5) 构造目标张量 (H,4): [x, y, v_t, θ_t]
    #    这里我们先不填 θ_t，留 0
    targ = torch.zeros(H, 4, device=device, dtype=torch.float32)
    targ[:,0] = xs
    targ[:,1] = ys
    # targ[:,2] = v_t
    # 可选：让 θ = atan2(dy,dx)
    # dxy = torch.cat([dpos, dpos[-1:]], dim=0)
    # targ[:,3] = torch.atan2(dxy[:,1], dxy[:,0])

    return targ

#  用sin 函数的方式 生成字母 M
def make_letter_target_M(H, dt, device):
    """
    生成 M 字的 (H,4) 目标张量 [x, y, v, θ=0]，全程在线性段上插值，
    仿照之前 sine 曲线的写法，不依赖样条。
    - H: 规划步数
    - dt: 时间步长
    - device: torch.device
    """
    # 0) 在 [0,1] 上等距离取 H 个点
    u = torch.linspace(0.0, 1.0, H, device=device)

    # 1) 准备存放 x,y
    xs = torch.empty(H, device=device)
    ys = torch.empty(H, device=device)

    # 2) 四段拼接
    # 段长 0.25
    # seg1: (-1,0)->(-0.5,1)
    mask1 = u <= 0.25
    t1 = u[mask1] / 0.25
    xs[mask1] = -1.0 + 0.5 * t1
    ys[mask1] =    0.0 + 1.0 * t1

    # seg2: (-0.5,1)->( 0,0)
    mask2 = (u > 0.25) & (u <= 0.50)
    t2 = (u[mask2] - 0.25) / 0.25
    xs[mask2] = -0.5 + 0.5 * t2
    ys[mask2] =  1.0 - 1.0 * t2

    # seg3: ( 0,0)->( 0.5,1)
    mask3 = (u > 0.50) & (u <= 0.75)
    t3 = (u[mask3] - 0.50) / 0.25
    xs[mask3] =  0.0 + 0.5 * t3
    ys[mask3] =  0.0 + 1.0 * t3

    # seg4: (0.5,1)->(1,0)
    mask4 = u > 0.75
    t4 = (u[mask4] - 0.75) / 0.25
    xs[mask4] =  0.5 + 0.5 * t4
    ys[mask4] =  1.0 - 1.0 * t4

    # 3) 缩放到你的字母大小（和 spline 版本 0.1 保持一致）
    xs = xs * 0.1
    ys = ys * 0.1

    # 4) 计算每步速度 v = dist/ dt
    dpos = torch.stack([xs, ys], dim=1)
    dpos = dpos[1:] - dpos[:-1]           # (H-1,2)
    speeds = torch.norm(dpos, dim=1) / dt # (H-1,)
    # 首速度可以填 0
    zero   = torch.tensor(0.0, device=device)
    v_t    = torch.cat([zero.unsqueeze(0), speeds], dim=0)  # (H,)

    # 5) 构造 (H,4) 目标张量：x,y,v,θ
    targ = torch.zeros(H, 4, device=device)
    targ[:,0] = xs
    targ[:,1] = ys
    # targ[:,2] = v_t
    # θ = 0（你也可以后面根据需要填 atan2(dy,dx)）
    # targ[:,3] = ...

    return targ



simulation_step = planner.MPC_step
device=planner.device

# —— 测试 M 字追踪 —— #
# traj_M = make_letter_target('M', H, dt, device)
traj_M = make_letter_target_M(simulation_step, dt, device)  # 这里的 H是规划的步数
print("\n-- Letter M Test --")
hist_v_M, hist_r_M = test_tracking(traj_M)


# # —— 测试 P 字追踪 —— #
# traj_P = make_letter_target('P', simulation_step, dt, device)
# print("\n-- Letter P Test --")
# hist_v_P,  hist_r_P = test_tracking(traj_P)



cases = [
    ("traj_M", traj_M, hist_r_M, hist_v_M),
    # ("traj_P", traj_P, hist_r_P, hist_v_P),
]

# for name, target, positions, virtual_fish_positions in cases:
#     xs = [p[0] for p in positions]
#     ys = [p[1] for p in positions]
#     # 目标
#     if hasattr(target, 'cpu'):
#         tx = target[:,0].cpu().numpy()
#         ty = target[:,1].cpu().numpy()
#     else:
#         tx = target[:,0]
#         ty = target[:,1]

#     plt.figure()
#     plt.plot(xs, ys, 'o-', label=f'{name} Track')
#     plt.plot(tx, ty, 'k--', label=f'{name} Target')
#     plt.title(f"MPC Tracking: {name}")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

for name, target, real_pos, virt_pos in cases:
    # unpack real‐fish history
    xs = [p[0] for p in real_pos]
    ys = [p[1] for p in real_pos]

    # unpack virtual‐fish history
    vxs = [p[0] for p in virt_pos]
    vys = [p[1] for p in virt_pos]

    # target letter
    if hasattr(target, 'cpu'):
        tx = target[:,0].cpu().numpy()
        ty = target[:,1].cpu().numpy()
    else:
        tx = target[:,0]
        ty = target[:,1]

    plt.figure()
    # plot target
    plt.plot(tx, ty,   'k--',   label=f'{name} Target')
    # plot real fish
    plt.plot(xs, ys,   'o-',    label=f'{name} Real Fish')
    # plot virtual fish in a different color/marker
    plt.plot(vxs, vys, '^-',    label=f'{name} Virtual Fish')

    plt.title(f"MPC Tracking: {name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


# 画real fish 和virtual fish的跟踪误差

# 时间轴
times = np.arange(hist_v_M.shape[0]) * dt

# 1) 误差随时间
errors = np.linalg.norm(hist_v_M - hist_r_M, axis=1)
plt.figure()
plt.plot(times, errors, '-')
plt.xlabel('Time (s)')
plt.ylabel('Euclidean error')
plt.title('Leader–Follower Position Error vs Time')
plt.grid(True)
plt.show()

# 2) x 方向位置随时间
plt.figure()
plt.plot(times, hist_v_M[:,0], label='Virtual Fish (x)')
plt.plot(times, hist_r_M[:,0], label='Real Fish    (x)')
plt.xlabel('Time (s)')
plt.ylabel('x position')
plt.title('x Position vs Time')
plt.legend()
plt.grid(True)
plt.show()

# 3) y 方向位置随时间
plt.figure()
plt.plot(times, hist_v_M[:,1], label='Virtual Fish (y)')
plt.plot(times, hist_r_M[:,1], label='Real Fish    (y)')
plt.xlabel('Time (s)')
plt.ylabel('y position')
plt.title('y Position vs Time')
plt.legend()
plt.grid(True)
plt.show()