# planner_RL.py
import torch
import numpy as np

class TrajectoryPlanner:
    """
    封装了轨迹生成、旋转对齐和平移对齐的导航模块。
    """
    def __init__(self, shape: str, H: int, dt: float, device: str = 'cpu'):
        self.device = device
        self.dt = dt
        self.total_steps = H
        self.shape = shape # 记录形状名称以便重新生成
        
        # 1. 生成【原始】基础轨迹 (标准朝向，例如 M 字底边平行于 X 轴)
        # 我们把这个作为 "template" (模板)，永远不修改它
        self.template_trajectory = self._generate_template(shape, H, dt, device)

        self.base_trajectory = self.template_trajectory
        
        # 2. 实际使用的轨迹 (会被旋转和平移)
        # 初始化时先等于模板
        self.active_trajectory = self.template_trajectory.clone()

        # 3. 变换参数 (初始化为无变换)
        self.rotation_matrix = torch.eye(2, device=device)
        self.traj_origin = torch.zeros(2, device=device)
        self.traj_angle = 0.0
        
        self.current_step = 0

    def _generate_template(self, shape, H, dt, device):
        """
        内部工厂函数：生成标准朝向的轨迹模板
        """
        shape_upper = shape.upper()
        if shape_upper == 'M':
            return make_letter_target_M(H, dt, device)
        elif shape_upper == 'LINE':
            return make_letter_target_line(H, dt, device)
        elif shape_upper == 'SINE':
            return make_target_sine(H, dt, device)
        else:
            raise ValueError(f"Shape '{shape}' not supported.")

    def reset(self, initial_pos: torch.Tensor, initial_angle: float):
        """
        初始化轨迹，计算并缓存所有变换矩阵。
        Args:
            initial_pos: (2,) Global 坐标系下的起点
            initial_angle: Global 坐标系下的期望朝向 (弧度)
        """
        self.current_step = 0
        self.traj_origin = initial_pos.clone()
        self.traj_angle = initial_angle

        # --- 1. 计算旋转矩阵 ---
        # 获取模板的自然朝向 (Local)
        p0 = self.template_trajectory[0, :2]
        # p1 = self.template_trajectory[1, :2]
        # template_angle = torch.atan2(p1[1] - p0[1], p1[0] - p0[0])
        # 简化版：假设模板就是水平向右的 (0度)
        template_angle = 0.0 
        
        # 需要旋转的角度
        delta_angle = initial_angle - template_angle
        
        cos_t = torch.cos(torch.tensor(delta_angle, device=self.device))
        sin_t = torch.sin(torch.tensor(delta_angle, device=self.device))
        
        # Global -> Local (逆旋转)
        # 这里的矩阵用于把 Global 向量转回 Local
        # Local = R^T * (Global - Origin)
        # 所以我们存 R (Local -> Global 的正向旋转矩阵)
        self.rotation_matrix = torch.tensor([
            [cos_t, -sin_t],
            [sin_t,  cos_t]
        ], device=self.device)

        # --- 2. 生成 active_trajectory (用于渲染和物理参考) ---
        # 这一步是为了 Render 方便，直接算出 Global 轨迹
        centered_traj = self.template_trajectory[:, :2] - p0
        rotated_pos = torch.matmul(centered_traj, self.rotation_matrix.T) # (N,2) * (2,2)
        final_pos = rotated_pos + self.traj_origin
        
        self.active_trajectory = self.template_trajectory.clone()
        self.active_trajectory[:, 0] = final_pos[:, 0]
        self.active_trajectory[:, 1] = final_pos[:, 1]
        
        return self.get_current_target_global()
    
    def transform_to_local(self, global_pos, global_vel):
        """把 Global 状态转成 Local (给 RL Model 看)"""
        # 位置: 先平移，再逆旋转 (R^T * (P - Origin))
        rel_pos = global_pos - self.traj_origin
        local_pos = torch.matmul(self.rotation_matrix.T, rel_pos)
        
        # 速度: 只逆旋转 (R^T * V)
        local_vel = torch.matmul(self.rotation_matrix.T, global_vel)
        
        return local_pos, local_vel

    def transform_to_global(self, local_action):
        """把 Local 动作转成 Global (给物理引擎用)"""
        # 动作通常是速度矢量: 正旋转 (R * Action)
        global_action = torch.matmul(self.rotation_matrix, local_action)
        return global_action

    def get_next_target_relative_local(self, s_v_global: torch.Tensor) -> torch.Tensor:
        """
        直接返回 Local 坐标系下的相对目标位置 (给 Observation 用)
        """
        # 1. 获取 Global 里的当前位置
        current_pos_global = s_v_global[:2]
        
        # 2. 获取 Global 里的下一个目标点
        next_idx = np.clip(self.current_step + 1, 0, self.total_steps - 1)
        target_pos_global = self.active_trajectory[next_idx, :2]
        
        # 3. 计算 Global 相对距离
        diff_global = target_pos_global - current_pos_global
        
        # 4. 转成 Local 相对距离 (只旋转，不平移)
        diff_local = torch.matmul(self.rotation_matrix.T, diff_global)
        
        return diff_local
    
    def get_current_target_global(self):
        idx = np.clip(self.current_step, 0, self.total_steps - 1)
        return self.active_trajectory[idx]





    def step(self):
        if self.current_step < self.total_steps - 1:
            self.current_step += 1
            
    def _get_target_by_step(self, step_index: int) -> torch.Tensor:
        step_index = np.clip(step_index, 0, self.total_steps - 1)
        # 直接从处理好的 active_trajectory 取值，不需要再加 offset 了
        return self.active_trajectory[step_index]

    def get_current_target(self) -> torch.Tensor:
        return self._get_target_by_step(self.current_step)

    def get_next_target_relative_pos(self, s_v: torch.Tensor) -> torch.Tensor:
        next_step_index = self.current_step + 1
        target_pos = self._get_target_by_step(next_step_index)[:2]
        current_pos = s_v[:2]
        return target_pos - current_pos

# ==================================================================
# 你提供的代码 (原封不动地放进来，或者你也可以单独 import)  原版的轨迹是针对0.1 dt 的，目前的频率变成了0.01，所以把轨迹长度变短来应对
# ==================================================================
def make_letter_target_M(H: int, dt: float, device: str) -> torch.Tensor:
    """
    生成 M 字的 (H,4) 目标张量 [x, y, v, θ=0]。
    x 轴从 0 移动到 10。
    y 轴振幅为 5 (与 X 轴保持良好比例)。
    - H: 规划步数
    - dt: 时间步长
    - device: torch.device
    """
    
    # 1) 定义轨迹参数
    amplitude = 3.0 # M 字的高度
    
    # 2) 准备 x,y
    # u 在 [0, 1] 上均匀取 H 个点
    u = torch.linspace(0.0, 1.0, H, device=device)
    xs = torch.empty(H, device=device)
    ys = torch.empty(H, device=device)

    # 3) 四段拼接 (每段占 0.25)
    # 段1: (0, 0) -> (2.5, 5)
    mask1 = u <= 0.25
    t1 = u[mask1] / 0.25  # t1 从 0 -> 1
    xs[mask1] = 0.0 + 2.5 * t1
    ys[mask1] = 0.0 + amplitude * t1

    # 段2: (2.5, 5) -> (5.0, 0)
    mask2 = (u > 0.25) & (u <= 0.50)
    t2 = (u[mask2] - 0.25) / 0.25 # t2 从 0 -> 1
    xs[mask2] = 2.5 + 2.5 * t2
    ys[mask2] = amplitude - amplitude * t2

    # 段3: (5.0, 0) -> (7.5, 5)
    mask3 = (u > 0.50) & (u <= 0.75)
    t3 = (u[mask3] - 0.50) / 0.25 # t3 从 0 -> 1
    xs[mask3] = 5.0 + 2.5 * t3
    ys[mask3] = 0.0 + amplitude * t3

    # 段4: (7.5, 5) -> (10.0, 0)
    mask4 = u > 0.75
    t4 = (u[mask4] - 0.75) / 0.25 # t4 从 0 -> 1
    xs[mask4] = 7.5 + 2.5 * t4
    ys[mask4] = amplitude - amplitude * t4

    # 4) 计算每步速度 v = dist / dt (保持格式一致性)
    dpos = torch.stack([xs, ys], dim=1)
    dpos = dpos[1:] - dpos[:-1]           # (H-1, 2)
    speeds = torch.norm(dpos, dim=1) / dt # (H-1,)
    
    # 首速度可以填 0
    zero_speed = torch.tensor(0.0, device=device)
    v_t = torch.cat([zero_speed.unsqueeze(0), speeds], dim=0)  # (H,)

    # 5) 构造 (H,4) 目标张量：x,y,v,θ
    targ = torch.zeros(H, 4, device=device)
    targ[:, 0] = 0.09 * xs
    targ[:, 1] = 0.4 * ys   # 这个是适配实际场景的字母M大小， 宽 0.09, 高 0.12
    
    # targ[:, 2] = v_t (可选)

    targ = 0.1 * targ 

    return targ

def make_letter_target_line(H: int, dt: float, device: str) -> torch.Tensor:
    """
    生成一条水平直线轨迹 (H,4) 目标张量 [x, y, v, θ=0]。
    从 (-0.1, 0) 移动到 (0.1, 0)。
    - H: 规划步数
    - dt: 时间步长
    - device: torch.device
    """
    
    # 1) 定义起点和终点
    x_start = -5
    x_end = 5
    y_pos = 0.0

    # 2) 准备存放 x,y
    # 在 [x_start, x_end] 上等距离取 H 个点
    xs = torch.linspace(x_start, x_end, H, device=device)
    ys = torch.full((H,), y_pos, device=device) # Y 轴始终为 0

    # 3) 计算每步速度 v = dist / dt
    # (这和你的 M 函数的逻辑完全一致)
    dpos = torch.stack([xs, ys], dim=1)
    dpos = dpos[1:] - dpos[:-1]           # (H-1, 2)
    speeds = torch.norm(dpos, dim=1) / dt # (H-1,)
    
    # 首速度可以填 0
    zero   = torch.tensor(0.0, device=device)
    v_t    = torch.cat([zero.unsqueeze(0), speeds], dim=0)  # (H,)

    # 4) 构造 (H,4) 目标张量：x,y,v,θ
    # 注意：你的 M 函数只用了 (H, 4)，且 v 和 θ 都是 0
    # 我们这里也保持一致，只填充 x, y
    targ = torch.zeros(H, 4, device=device)
    targ[:, 0] = xs
    targ[:, 1] = ys

    targ = 0.1 * targ
    
    # (如果你希望 v 和 θ 也被填充，可以取消下面两行的注释)
    # targ[:, 2] = v_t 
    # targ[:, 3] = 0.0 # θ 始终为 0
    return targ

def make_target_sine(H: int, dt: float, device: str) -> torch.Tensor:
    """
    生成一条 Sine 波形轨迹 (H,4) 目标张量 [x, y, v, θ=0]。
    x 轴从 -5 移动到 5 (与 'LINE' 保持一致)。
    y 轴完成一个完整的 sin 周期 (振幅为 1)。
    
    Args:
        H: 规划步数
        dt: 时间步长
        device: torch.device
    """
    
    # 1) 定义轨迹参数
    x_start = -5.0
    x_end = 5.0
    amplitude = 1.0  # Y 轴振幅
    num_cycles = 1.0   # 在 x 轴范围内完成 1 个完整周期

    # 2) 准备存放 x,y
    # x 轴
    xs = torch.linspace(x_start, x_end, H, device=device)
    
    # y 轴
    # 将 x 的范围 [x_start, x_end] 映射到 [0, 2*pi*num_cycles]
    angle = (xs - x_start) / (x_end - x_start) * (2 * torch.pi * num_cycles)
    ys = amplitude * torch.sin(angle)
    # (轨迹将从 (-5, 0) 开始, 到 (5, 0) 结束)

    # 3) 计算每步速度 v = dist / dt (和你的 M 函数逻辑一致)
    dpos = torch.stack([xs, ys], dim=1)
    dpos = dpos[1:] - dpos[:-1]           # (H-1, 2)
    speeds = torch.norm(dpos, dim=1) / dt # (H-1,)
    
    # 首速度可以填 0
    zero_speed = torch.tensor(0.0, device=device)
    v_t = torch.cat([zero_speed.unsqueeze(0), speeds], dim=0)  # (H,)

    # 4) 构造 (H,4) 目标张量：x,y,v,θ
    # 遵循你的格式, 只填充 x, y
    targ = torch.zeros(H, 4, device=device)
    targ[:, 0] = xs
    targ[:, 1] = ys

    targ = 0.1 * targ
    
    # (如果你希望 v 也被填充，可以取消下面一行的注释)
    # targ[:, 2] = v_t 
    
    return targ