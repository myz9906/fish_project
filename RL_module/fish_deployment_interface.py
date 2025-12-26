# fish_deployment_interface.py
import numpy as np
import torch
import pandas as pd  # <--- [新增 1] 导入 pandas
from stable_baselines3 import PPO

# 1. 导入环境类 (为了获取参数)
from env_RL import CustomFishEnv
# 2. 导入规划器
from planner_RL import TrajectoryPlanner


class FishDeploymentAgent:
    """
    部署接口类：
    负责连接 真实世界(Global Frame) 与 RL大脑(Local Frame)。
    它利用 Planner 进行坐标系的实时转换。
    """
    def __init__(self, model_path, device='cpu'):
        self.device = device

        # --- 1. 加载配置 (Single Source of Truth) ---
        self.dt = CustomFishEnv.DT
        self.max_speed = CustomFishEnv.MAX_SPEED
        self.rx = CustomFishEnv.REACHABLE_RX
        self.ry = CustomFishEnv.REACHABLE_RY
        # self.history_window = 1
        self.history_window = CustomFishEnv.VEL_SMOOTH_WINDOW

        self.constraint_type = 'box'

        self.max_steps = CustomFishEnv.MAX_STEPS
        
        # print(f"Deployment Config Loaded:")
        # print(f" - dt: {self.dt} (Freq: {1/self.dt:.1f}Hz)")
        # print(f" - Max Speed: {self.max_speed}")
        # print(f" - Reachable Set: x={self.rx}, y={self.ry}")

        # # --- 2. 加载模型 ---
        # print(f"Loading model from {model_path}...")
        self.model = PPO.load(model_path, device=self.device)
        
        # --- 3. 初始化规划器 ---
        # 规划器现在负责所有的几何计算和坐标变换
        self.planner = TrajectoryPlanner(
            shape='M', 
            H=self.max_steps, 
            dt=self.dt, 
            device=self.device
        )

        # --- 4. 运行时状态缓存 ---
        self.pos_r_history = []  # 真实鱼位置历史 (用于计算平滑速度)
        self.pos_v_last = None   # 上一帧虚拟鱼位置 (Global)
        self.last_action_vel_global = np.array([0.0, 0.0], dtype=np.float32) # 上一帧动作 (Global)

        self.s_v = None # Global State Tensor
        self.s_r = None # Global State Tensor

        # 数据记录
        self.data_log = []
        self.step_count = 0

    def reset(self, init_r_pos, init_v_pos, init_angle=0.0):
        """
        重置接口。
        Args:
            init_r_pos: (x, y) 真实鱼初始位置 (Global)
            init_v_pos: (x, y) 虚拟鱼初始位置 (Global)
            init_angle: float, 期望的轨迹朝向 (弧度, Global)
        """
        # 1. 初始化历史缓存
        self.pos_r_history = [init_r_pos] * (self.history_window + 2)
        self.pos_v_last = np.array(init_v_pos, dtype=np.float32)
        self.last_action_vel_global = np.array([0.0, 0.0], dtype=np.float32)

        # 2. 初始化 Global 状态 Tensor
        self.s_r = torch.tensor([init_r_pos[0], init_r_pos[1], 0.0, 0.0], dtype=torch.float32, device=self.device)
        self.s_v = torch.tensor([init_v_pos[0], init_v_pos[1], 0.0, 0.0], dtype=torch.float32, device=self.device)

        # 3. [关键] 重置 Planner (建立坐标系)
        # Planner 会在内部计算好旋转矩阵和平移向量
        self.planner.reset(initial_pos=self.s_r[:2], initial_angle=init_angle)

        # 4. 重置日志
        self.step_count = 0
        # self.data_log.append({
        #     "step": 0,
        #     "real_x": float(init_r_pos[0]), 
        #     "real_y": float(init_r_pos[1]),
        #     "real_vx": 0.0, 
        #     "real_vy": 0.0,
        #     "virtual_x": float(init_v_pos[0]), 
        #     "virtual_y": float(init_v_pos[1]),
        #     "action_vx": 0.0, 
        #     "action_vy": 0.0
        # })
        
        print(f"Agent Reset: Start Pos={init_r_pos}, Angle={init_angle:.2f}")

    def step(self, real_x, real_y):
        """
        核心步进函数。
        输入: 真实鱼的 Global 坐标
        输出: 虚拟鱼下一帧的 Global 坐标
        """
        self.step_count += 1

        # =========================================
        # 1. 更新 Global 状态 (物理世界)
        # =========================================
        
        # A. 计算真实鱼速度 (差分+平滑)
        self.pos_r_history.append([real_x, real_y])
        if len(self.pos_r_history) > self.history_window * 3:
            self.pos_r_history.pop(0)
        vel_r = self._calculate_real_velocity_smooth(self.pos_r_history)

        # B. 获取虚拟鱼状态
        # 速度直接取自上一帧 Agent 的 Global 输出 (零延迟)
        vel_v = self.last_action_vel_global
        curr_v_pos = self.pos_v_last

        # C. 更新 Tensor (Global Frame)
        self.s_r = torch.tensor([real_x, real_y, vel_r[0], vel_r[1]], dtype=torch.float32, device=self.device)
        self.s_v = torch.tensor([curr_v_pos[0], curr_v_pos[1], vel_v[0], vel_v[1]], dtype=torch.float32, device=self.device)

        # =========================================
        # 2. 构造 Observation (Global -> Local)
        # =========================================
        
        # 利用 Planner 转换状态到 Local Frame (Agent 只懂 Local)
        sv_pos_local, sv_vel_local = self.planner.transform_to_local(self.s_v[:2], self.s_v[2:])
        sr_pos_local, sr_vel_local = self.planner.transform_to_local(self.s_r[:2], self.s_r[2:])
        
        # 计算 Local 相对状态
        # (真实鱼 Local - 虚拟鱼 Local)
        rel_pos_local = sr_pos_local - sv_pos_local
        rel_vel_local = sr_vel_local - sv_vel_local
        
        # 获取 Local 相对目标
        rel_target_local = self.planner.get_next_target_relative_local(self.s_v) # 输入Global，内部自动转Local

        # 拼接 Observation
        obs_tensor = torch.cat([
            sv_pos_local, sv_vel_local,  # s_v (Local)
            rel_pos_local, rel_vel_local, # s_r - s_v (Local)
            rel_target_local             # target (Local)
        ], dim=0)
        
        obs_np = obs_tensor.cpu().detach().numpy()

        # =========================================
        # 3. RL 推理 & 动作转换 (Local -> Global)
        # =========================================
        
        # A. 获取 Local Action (Agent 以为自己在跑水平 M)
        action_local, _ = self.model.predict(obs_np, deterministic=True)
        action_local_tensor = torch.tensor(action_local, device=self.device, dtype=torch.float32)
        
        # B. 转换为 Global Action (物理世界实际需要的速度方向)
        action_global_tensor = self.planner.transform_to_global(action_local_tensor)
        
        # C. 缩放速度
        action_scaled_global = action_global_tensor.cpu().detach().numpy() * self.max_speed


        if np.linalg.norm(action_scaled_global) > 1e-6:
            self.current_orientation = np.arctan2(action_scaled_global[1], action_scaled_global[0])
        else:
            # 如果 RL 输出速度极小（几乎停下），则保持上一帧的朝向，避免鱼头乱转
            # self.current_orientation 保持不变
            pass

        # =========================================
        # 4. 物理积分 & 约束 (Global Frame)
        # =========================================
        
        # 积分
        next_v_x = curr_v_pos[0] + action_scaled_global[0] * self.dt
        next_v_y = curr_v_pos[1] + action_scaled_global[1] * self.dt

        # 可达集约束 (在 Global Frame 下做 Box Constraint)
        # 注意：这里和 env_RL 保持一致，使用 Global 矩形
        rel_x = next_v_x - real_x
        rel_y = next_v_y - real_y
        # =========================================
        # 【核心修改】根据 constraint_type 选择不同的截断逻辑
        # =========================================
        clamped_x, clamped_y = 0.0, 0.0

        if self.constraint_type == 'box':
            # --- 方案 A: 矩形约束 (原逻辑) ---
            # 这里的 rx, ry 分别是矩形的长宽限制
            clamped_x = np.clip(rel_x, -self.rx, self.rx)
            clamped_y = np.clip(rel_y, -self.ry, self.ry)

        elif self.constraint_type == 'circle':
            # --- 方案 B: 圆形约束 (新逻辑) ---
            # 1. 计算距离 (L2 Norm)
            dist = np.sqrt(rel_x**2 + rel_y**2)
            
            # 2. 定义最大半径
            # 按照你的代码逻辑：max_radius = 2 * rx
            # (这里假设你传进来的 rx 是某种基准长度)
            max_radius = 2 * self.rx 
            
            # 3. 截断判断
            if dist > max_radius:
                # 缩放回圆内 (保持方向，压缩长度)
                scale = max_radius / (dist + 1e-8) # 防止除0
                clamped_x = rel_x * scale
                clamped_y = rel_y * scale
            else:
                # 在圆内，不处理
                clamped_x = rel_x
                clamped_y = rel_y
        
        final_v_x = real_x + clamped_x
        final_v_y = real_y + clamped_y

        # =========================================
        # 5. 更新缓存
        # =========================================
        self.last_action_vel_global = action_scaled_global # 存 Global 速度
        self.pos_v_last = np.array([final_v_x, final_v_y], dtype=np.float32)
        self.planner.step()

        # 记录日志
        # self._log_data(real_x, real_y, vel_r, final_v_x, final_v_y, action_scaled_global)

        return final_v_x, final_v_y, self.current_orientation

    def _calculate_real_velocity_smooth(self, pos_history):
        """
        计算真实鱼速度 (Global Frame)
        同之前的逻辑：差分 + cumsum平滑
        """
        pos_array = np.array(pos_history)

        # if len(pos_array) < 2: return np.zeros(2, dtype=np.float32)

        x, y = pos_array[:, 0], pos_array[:, 1]
        
        vx, vy = np.zeros_like(x), np.zeros_like(y)
        vx[1:] = (x[1:] - x[:-1]) / self.dt
        vy[1:] = (y[1:] - y[:-1]) / self.dt
        vx[0], vy[0] = vx[1], vy[1]

        window = self.history_window
        kernel = np.ones(window, dtype=np.float32) / window

        def smooth(arr):
            if len(arr) < window: return arr
            conv = np.convolve(arr, kernel, mode='valid')
            prefix = np.cumsum(arr[:window-1]) / np.arange(1, window)
            return np.concatenate([prefix, conv])

        # 3. 执行平滑
        # 我们只需要最新的速度，但为了平滑正确，需要传入历史片段
        vx_s = smooth(vx)
        vy_s = smooth(vy)

        # vx_s = vx
        # vy_s = vy

        # 返回当前时刻 (最新) 的速度
        return np.array([vx_s[-1], vy_s[-1]], dtype=np.float32)
    

    def update_constraints(self, new_rx, new_ry, constraint_type='box'):
        """
        动态更新约束参数 和 约束形状
        """
        self.rx = new_rx
        self.ry = new_ry
        self.constraint_type = constraint_type  # 更新模式

    def _log_data(self, rx, ry, r_vel, vx, vy, action_vel):
        self.data_log.append({
            "step": self.step_count,
            "real_x": float(rx), "real_y": float(ry),
            "real_vx": float(r_vel[0]), "real_vy": float(r_vel[1]),
            "virtual_x": float(vx), "virtual_y": float(vy),
            "action_vx_global": float(action_vel[0]), 
            "action_vy_global": float(action_vel[1])
        })

    def save_csv(self, filename="deployment_log.csv"):
        if not self.data_log: return
        pd.DataFrame(self.data_log).to_csv(filename, index=False)
        print(f"Log saved to {filename}")


        