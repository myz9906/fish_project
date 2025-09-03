import numpy as np
import math
import time
from scipy.interpolate import CubicSpline
import torch
from models import DynamicsModel
from model_utils import load_dynamics_model


# MOD: 导入公共 step 函数
from sim import step_dynamics

class LetterPlanner:
    def __init__(self, letter='M', total_steps=500, dt=1/30., device='cpu', dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.dt     = dt
        self.T      = total_steps

        # 1) 定义控制点并缩放
        if letter == 'M':
            x_pts = [-1.0, -0.5, 0.0, 0.5, 1.0]
            y_pts = [ 0.0,  1.0, 0.0, 1.0, 0.0]
        elif letter == 'P':
            x_pts = [-1.0, -1.0, 0.0, 0.0]
            y_pts = [ 0.0,  1.0, 1.0, 0.0]
        else:
            x_pts = [0.0, 0.0]
            y_pts = [0.0, 0.0]

        x_pts = [0.1 * x for x in x_pts]
        y_pts = [0.1 * y for y in y_pts]

        # 2) 三次样条插值
        t_knots = np.linspace(0, 1, len(x_pts))
        cs_x = CubicSpline(t_knots, x_pts)
        cs_y = CubicSpline(t_knots, y_pts)

        # 3) 离散化轨迹并存成 Tensor (T,4)
        rel = []
        prev_x = prev_y = None
        for i in range(self.T):
            u = i / (self.T - 1)
            x = float(cs_x(u))
            y = float(cs_y(u))
            if i == 0:
                vx = vy = 0.0
            else:
                vx = (x - prev_x) / self.dt
                vy = (y - prev_y) / self.dt
            rel.append([x, y, vx, vy])
            prev_x, prev_y = x, y
        self.traj = torch.tensor(rel, dtype=torch.float32, device=self.device)  # (T,4)

        # 缓存第一个相对点
        self.dx0, self.dy0 = self.traj[0,0], self.traj[0,1]

        # 状态
        self.step_count = 0
        self.offset     = torch.zeros(2, dtype=torch.float32, device=self.device)

    def reset(self, init_position=None):
        """
        重置步数，并把 traj[0] 对齐到 init_position。
        init_position: (x0, y0) 或 None -> (0,0)
        """
        self.step_count = 0
        if init_position is None:
            init = torch.zeros(2, device=self.device)
        else:
            init = torch.tensor(init_position, dtype=torch.float32, device=self.device)
        self.offset = init - self.traj[0,:2]  # (2,)

    def get_target(self, obs=None):
        """
        返回当前 step_count 的绝对目标 (x, y, vx, vy)，
        类型：torch.Tensor(4,)
        """
        rel = self.traj[self.step_count]           # (4,)
        xy  = self.offset + rel[:2]                # (2,)
        return torch.cat([xy, rel[2:4]], dim=0)     # (4,)

    def get_future_targets(self, H):
        """
        返回形状 (H,4) 的 torch.Tensor，表示从下一步开始连续 H 步目标。
        """
        base = self.step_count
        # [base+1, base+2, ..., base+H], clamp 在 [0, T-1]
        idxs = torch.arange(base+1, base+1+H, device=self.device).clamp(max=self.T-1)
        future = self.traj[idxs]                   # (H,4)
        # 位置维度加上 offset
        pos   = future[:,:2] + self.offset.unsqueeze(0)  # (H,2)
        vel   = future[:,2:4]                      # (H,2)
        return torch.cat([pos, vel], dim=1)        # (H,4)

    def step(self, x_r, y_r):
        
        self.offset = torch.tensor([x_r-self.dx0, y_r-self.dy0], device=self.device, dtype=self.dtype)
        # 指针前移
        self.step_count = min(self.step_count + 1, self.T - 1)  # 最终的输出是numpy类型



# # 这个 action planner是 CEM model 
# class ActionPlanner:
#     def __init__(self, model_path, device='cpu', horizon=10,
#                  pop_size=200, elite_frac=0.2, iterations=5, dt=1/30.):
#         self.device = device
#         self.model = load_dynamics_model(model_path, device=device)
#         self.H  = horizon
#         self.K  = pop_size
#         self.M  = int(self.K * elite_frac)
#         self.I  = iterations
#         self.dt = dt

#     # MOD: 新增参数 traj_target，用于传入未来 H 步的目标 [x,y,v,theta]
#     def plan(self, s_v, s_r, traj_target):
#         """
#         s_v, s_r: 当前 (5,) 状态
#         traj_target: numpy array, shape (H,4)，每行 (x*, y*, v*, theta*)
#         返回：最优动作 mean[0] 以及整条动作序列 mean (H,2)
#         """
#         mean = np.zeros((self.H, 2), dtype=np.float32)
#         std  = np.ones ((self.H, 2), dtype=np.float32) * 0.5

#         for it in range(self.I):
#             cands   = np.random.randn(self.K, self.H, 2) * std + mean
#             rewards = np.zeros(self.K, dtype=np.float32)

#             for k in range(self.K):
#                 sv, sr = s_v.clone(), s_r.clone()
#                 rsum = 0.0

#                 for t in range(self.H):
#                     a = torch.tensor(cands[k, t], device=self.device)
#                     # 调用公共 step 函数推进
#                     sv, sr = step_dynamics(sv, sr, a, self.model, self.dt)

#                     # MOD: 根据 traj_target[t] 计算 reward
#                     x_tgt, y_tgt, v_tgt, theta_tgt = traj_target[t]
#                     # 位置误差
#                     pos_err = torch.norm(sr[:2] - torch.tensor([x_tgt, y_tgt], device=self.device))
#                     # 速度误差
#                     vel    = torch.norm(sr[2:4])
#                     vel_err = torch.abs(vel - v_tgt)
#                     # 可加方向误差 term，如果需要:
#                     # dir_err = 1 - torch.cos(sr[4] - theta_tgt)
#                     # rsum += -(pos_err + 0.1*vel_err + 0.1*dir_err)

#                     rsum += -(pos_err + 0.1 * vel_err)

#                 rewards[k] = rsum

#             # 保留精英，更新 mean/std
#             elite_idx = rewards.argsort()[-self.M:]
#             elites    = cands[elite_idx]
#             mean      = elites.mean(axis=0)
#             std       = elites.std (axis=0) + 1e-6

#         return mean[0], mean   # 最终的输出是numpy类型
    

# 这个 是基于梯度MPC 的 ActionPlanner

# class ActionPlanner:
#     """
#     基于梯度型 MPC 的 ActionPlanner

#     参数:
#       model_path:  训练好的 DynamicsModel 权重文件路径
#       device:      运行设备 ('cpu' or 'cuda')
#       horizon:     MPC 预测步长 H
#       mpc_iters:   每次滚动优化的迭代次数
#       lr:          优化器学习率
#       lambda_v:    速度跟踪权重
#       lambda_delta:动作平滑权重
#       v_max:       虚拟鱼速度的最大绝对值，用于 clamp
#     """
#     def __init__(self,
#                  dynamics_model,
#                  device='cpu',
#                  horizon=10,
#                  mpc_iters=20,
#                  lr=1e-2,
#                  lambda_v=0.1,
#                  lambda_delta=0.0,
#                  v_max=10.0,
#                  dt=1/30.):
#         self.device = device
#         self.dynamics_model = dynamics_model
#         self.H      = horizon
#         self.iters  = mpc_iters
#         self.lr     = lr
#         self.lambda_v     = lambda_v
#         self.lambda_delta = lambda_delta
#         self.v_max  = v_max
#         self.dt     = dt

#     def plan(self, s_v: torch.Tensor, s_r: torch.Tensor, traj_target: np.ndarray):
#         # 初始化可优化动作序列
#         a_seq = torch.zeros(self.H,2,device=self.device,requires_grad=True)
#         optimizer = torch.optim.Adam([a_seq], lr=self.lr)

#         for it in range(self.iters):
#             optimizer.zero_grad()
#             sv, sr = s_v.clone(), s_r.clone()
#             loss = 0.0

#             for k in range(self.H):
#                 sv, sr = self.dynamics_model.step(sv, sr, a_seq[k], self.dt)
#                 x_t, y_t, *_ = traj_target[k]
#                 loss += (sr[0]-x_t)**2 + (sr[1]-y_t)**2
#                 # vel   = torch.norm(sr[2:4])
#                 # vel_t = traj_target[k, 2]           # 从 traj_target 传进来的目标速度
#                 # loss += 0.1 * (vel - vel_t)**2
#                 if k>0:
#                     loss += self.lambda_delta * torch.norm(a_seq[k]-a_seq[k-1])**2

#             # 打印一下 loss
#             # print(f"iter {it} loss = {loss.item():.4f}")

#             loss.backward()
#             # 打印梯度范数
#             # print(" ||a_seq.grad|| =", a_seq.grad.norm().item())

#             optimizer.step()
#             with torch.no_grad():
#                 a_seq.clamp_(-self.v_max, self.v_max)
#             # 打印更新后 a_seq[0]
#             # print(" a_seq[0] =", a_seq.detach().cpu().numpy())

#         return a_seq.detach().cpu().numpy()[0], a_seq.detach().cpu().numpy()

# 考虑速度等误差的MPC planner:
class ActionPlanner:
    def __init__(self,
                 dynamics_model,
                 device='cpu',
                 horizon=10,
                 mpc_iters=20,
                 MPC_step = 100,
                 lr=1e-2,
                 w_pos=40.0,        # 位置误差权重
                 w_vel=0,        # 速度误差权重
                 w_dir=0,        # 朝向误差权重
                 w_delta=0,     # 动作平滑权重
                 w_act=0,       # 动作幅度惩罚
                 w_dist = 50.0,
                # w_dist = 0.0,
                 w_term=1.0,       # 终端位置惩罚
                 v_max=10.0,
                 dt=1/30.):
        self.device  = device
        self.model   = dynamics_model
        self.H       = horizon
        self.iters   = mpc_iters
        self.lr      = lr
        self.w_pos   = w_pos
        self.w_vel   = w_vel
        self.w_dir   = w_dir
        self.w_delta = w_delta
        self.w_act   = w_act
        self.w_dist  = w_dist
        self.w_term  = w_term
        self.v_max   = v_max
        self.dt      = dt
        self.MPC_step = MPC_step

    # 用 reachable set的办法，限制 leader 也就是virtual fish的运动范围
    def _project_to_reachable(self, x_t: torch.Tensor, y_t: torch.Tensor,
                              sr_x: torch.Tensor, sr_y: torch.Tensor):
        """
        如果 model 定义了 v_xmax/v_ymax，就把 (x_t,y_t) 投影到
        跟鱼 sr 在一小步 dt 内的可达矩形:
         |Δx| ≤ v_xmax*dt, |Δy| ≤ v_ymax*dt
        否则原样返回。
        """
        # 只有 BioPD model 拥有 v_xmax / v_ymax
        if not (hasattr(self.model, 'rx') and hasattr(self.model, 'ry')):
            return x_t, y_t

        # 误差向量
        dx = x_t - sr_x
        dy = y_t - sr_y
        # 最大步长

        # dx_max = self.model.v_xmax * self.dt
        # dy_max = self.model.v_ymax * self.dt

        rx, ry = self.model.rx, self.model.ry

        # print("dx_max", dx_max, "dy_max", dy_max)
        # clamp

        # dx = torch.clamp(dx, -dx_max, dx_max)    # 目前这个是根据 real fish的速度来确定 上下限，还有一个思路是直接根据 cutoff 距离来确定，直接把移动距离限制在cutoff的距离之内：dx = torch.clamp(dx, -rx, rx)
        # dy = torch.clamp(dy, -dy_max, dy_max)

        # 直接用距离做cutoff 

        dx = torch.clamp(dx, -rx, rx)    # 目前这个是根据 real fish的速度来确定 上下限，还有一个思路是直接根据 cutoff 距离来确定，直接把移动距离限制在cutoff的距离之内：dx = torch.clamp(dx, -rx, rx)
        dy = torch.clamp(dy, -ry, ry)

        # 返回投影后的位置
        return sr_x + dx, sr_y + dy
    
    def plan(self, s_v: torch.Tensor, s_r: torch.Tensor, traj_target: torch.Tensor):
        """
        traj_target: (H,4) 张量，每行 (x*, y*, v*, θ*)
        """
        T = traj_target.shape[0]
        H_eff = min(self.H, T)    # 真正要跑的步数


        # warm‐start：如果你之前保存了 self.prev_seq，可以在这用它初始化 a_seq
        a_seq = torch.zeros(self.H, 2,
                            device=self.device,
                            requires_grad=True)  #为a_seq 构建计算图，方便以后计算梯度
        optimizer = torch.optim.Adam([a_seq], lr=self.lr)  # 把 a_seq 放到一个 Adam 优化器里，等于告诉优化器“我的待优化变量就是这个张量”。

        for _ in range(self.iters):
            optimizer.zero_grad()
            sv, sr = s_v.clone(), s_r.clone()
            # 保存“真实鱼当前状态”用来第一次投影
            x_r0 = s_r[0].clone()
            y_r0 = s_r[1].clone()
            loss = 0.0

            for k in range(H_eff):
                # rollout 一步
                sv, sr = self.model.step(sv, sr, a_seq[k], self.dt)
                # print("sv", sv)
                # print("sr", sr)
                # time.sleep(1)  # 暂停 1 秒

                # 目标
                x_t, y_t, v_t, th_t = traj_target[k]
                # —— 可达集投影 —— #

                # if k == 0:
                    # x_t, y_t = self._project_to_reachable(x_t, y_t, x_r0, y_r0)

                # 1) 位置误差
                pos_err = (sr[0] - x_t)**2 + (sr[1] - y_t)**2
                loss += self.w_pos * pos_err

                # # 2) 速度误差
                # vel    = torch.norm(sr[2:4])
                # vel_err = (vel - v_t)**2
                # loss += self.w_vel * vel_err

                # # 3) 方向误差
                # dir_err = 1 - torch.cos(sr[4] - th_t)
                # loss += self.w_dir * dir_err

                # # 4) 动作平滑
                # if k > 0:
                #     loss += self.w_delta * torch.norm(a_seq[k] - a_seq[k-1])**2

                # # 5) 动作幅度惩罚（防止猛冲）
                # loss += self.w_act * torch.norm(a_seq[k])**2

                # # leader 与 follower 相对距离的惩罚， 让双方不要离得太远

                if hasattr(self.model, 'rx') and hasattr(self.model, 'ry'):
                    rx, ry = self.model.rx, self.model.ry
                    dx = torch.abs(sr[0] - sv[0]) - rx
                    dy = torch.abs(sr[1] - sv[1]) - ry
                    # penalize only when outside the cutoff
                    loss += self.w_dist * torch.relu(dx)**2
                    loss += self.w_dist * torch.relu(dy)**2

            # 6) 终端位置惩罚
            xH, yH, _, _ = traj_target[-1]
            term_err = (sr[0] - xH)**2 + (sr[1] - yH)**2
            loss += self.w_term * term_err

            loss.backward()
            optimizer.step()          # 用反向传播 + 梯度下降的方式进行优化
            with torch.no_grad():
                a_seq.clamp_(-self.v_max, self.v_max)

        full_seq = a_seq.detach().cpu().numpy()
        return full_seq[0].copy(), full_seq