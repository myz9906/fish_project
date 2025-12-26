import gymnasium as gym
from gymnasium import spaces
import numpy as np

import torch
# from models import DynamicsModel
# from planner_RL import TrajectoryPlanner

# from dynamics_models import LearnedDynamicsModel
import matplotlib.pyplot as plt  # for rendering


# class CustomFishEnv(gym.Env):
#     """
#     你的自定义虚拟鱼引导环境
#     它连接了 DynamicsModel (物理) 和 TrajectoryPlanner (导航)
#     """

#     # _# --- 新增 (用于 Render) ---_
#     metadata = {"render_modes": ["human"], "render_fps": 30}
    
#     def __init__(self, 
#                  dynamics_model: LearnedDynamicsModel, 
#                  planner: TrajectoryPlanner,
#                  dt: float = 0.1,
#                  max_episode_steps: int = 1000,
#                  device: str = 'cpu',
#                  reward_weights: dict = None,
#                  render_mode: str = None, # _# --- 新增 (用于 Render) ---_

#                  reachable_set_rx: float = 0.01, # 矩形约束的 X 半径
#                  reachable_set_ry: float = 0.01  # 矩形约束的 Y 半径
#                  ):
#         """
#         Args:
#             dynamics_model: 已初始化的动力学模型 (你的 f_fw)
#             planner: 已初始化的轨迹规划器
#             dt: 时间步长 (必须与 planner 和 dynamics 匹配)
#             max_episode_steps: 每轮最大步数
#             device: 'cpu' or 'cuda'
#             reward_weights: 奖励项的权重
#         """
#         super(CustomFishEnv, self).__init__()

#         self.device = torch.device(device)
#         self.dtype = torch.float32

#         # --- 1. 核心模块 ---
#         self.dynamics_model = dynamics_model
#         self.planner = planner
#         self.dt = dt
#         self.max_episode_steps = max_episode_steps

#         # self.max_speed_virtual_fish = 0.5
#         self.max_speed_virtual_fish = 1.0

#         self.rx = reachable_set_rx   # 给virtual fish 增加运动约束
#         self.ry = reachable_set_ry



class CustomFishEnv(gym.Env):
    """
    你的自定义虚拟鱼引导环境
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    # =================================================================
    # [关键修改]：将关键配置提升为类属性 (Class Attributes)
    # 这样部署脚本可以直接通过 CustomFishEnv.DT 访问，无需实例化
    # =================================================================
    DT = 0.01                  # 时间步长 (100Hz)
    MAX_SPEED = 1.0            # 虚拟鱼最大速度
    REACHABLE_RX = 0.01        # 可达集约束 X
    REACHABLE_RY = 0.01        # 可达集约束 Y
    MAX_STEPS = 500            # 单局最大步数
    VEL_SMOOTH_WINDOW = 5      # 速度平滑窗口 (这是你之前定义的)
    # =================================================================

    def __init__(self, 
                 dynamics_model, 
                 planner, 
                 # 注意：这里默认值改用 self.DT 等，保持一致性
                 dt: float = DT,
                 max_episode_steps: int = MAX_STEPS,
                 device: str = 'cpu',
                 reward_weights: dict = None,
                 render_mode: str = None,
                 
                 # 这里允许覆盖，但默认使用类属性
                 reachable_set_rx: float = REACHABLE_RX,
                 reachable_set_ry: float = REACHABLE_RY
                 ):
        
        super(CustomFishEnv, self).__init__()
        
        self.device = torch.device(device)
        self.dtype = torch.float32
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        
        # 使用传入的参数或类属性
        self.max_speed_virtual_fish = self.MAX_SPEED 
        self.rx = reachable_set_rx
        self.ry = reachable_set_ry

        self.dynamics_model = dynamics_model
        self.planner = planner



        # --- 2. 奖励权重 ---
        if reward_weights is None:
            self.reward_weights = {'tracking': 1.0, 'smoothing': 0.1}
        else:
            self.reward_weights = reward_weights

        # --- 3. 定义动作空间 (Action Space) --- 这两个空间的目标并不是要规定上下限，更重要的是接入gym环境
        # 虚拟鱼动作 (vx, vy)
        # 假设速度范围是 -1 到 +1 (你需要根据你的动力学调整)
        self.action_space = spaces.Box(
            low=-self.max_speed_virtual_fish, 
            high=self.max_speed_virtual_fish, 
            shape=(2,), 
            dtype=np.float32
        )

        # --- 4. 定义观测空间 (Observation Space) ---
        # 你的要求: [s_v (4), s_r - s_v (4), rel_target_pos (2)]
        obs_dim = 4 + 4 + 2 
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )

        # --- 5. 初始化内部状态 ---
        self.s_v = None         # 虚拟鱼状态 (5,) tensor
        self.s_r = None         # 真实鱼状态 (5,) tensor

        self.current_step = 0
        
        # 用于奖励计算的缓存
        self.prev_dist_to_target = None
        self.prev_action = None # 上一步动作 (2,) tensor

        # --- 6. 渲染相关的状态 --- # _# --- 新增 (用于 Render) ---_
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.virtual_fish_history = []
        self.real_fish_history = []
        if self.render_mode == "human":
            plt.ion() # 开启交互模式


    def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            
            # 1. 初始化状态 (Global Frame)
            self._initialize_state_vectors()
            
            # 2. 处理 Options (例如指定角度)
            init_angle = np.pi

            if options:
                init_angle = options.get('init_angle', 0.0)
                if 'init_r_pos' in options: self.s_r[:2] = torch.tensor(options['init_r_pos'], device=self.device)
                if 'init_v_pos' in options: self.s_v[:2] = torch.tensor(options['init_v_pos'], device=self.device)
            
            # 3. [极简调用] 全权委托 Planner 初始化
            # Planner 会在内部算好旋转矩阵，准备好服务
            initial_target =self.planner.reset(initial_pos=self.s_r[:2], initial_angle=init_angle)

            # 5. 重置奖励计算器
            initial_pos = self.s_v[:2]
            target_pos = initial_target[:2]
            self.prev_dist_to_target = torch.norm(initial_pos - target_pos)
            # 2. 重置上一步动作 (用于平滑奖励)
            self.prev_action = torch.zeros(2, device=self.device, dtype=self.dtype)
            
            # ... (重置缓存、历史记录等代码不变) ...

            self.virtual_fish_history.clear()
            self.real_fish_history.clear()
            # 添加初始点
            self.virtual_fish_history.append(self._torch_to_np(self.s_v[:2]))
            self.real_fish_history.append(self._torch_to_np(self.s_r[:2]))

            # _# --- 新增 (用于 Render) ---_
            # 4. 在 reset 后渲染第一帧
            if self.render_mode == "human":
                self.render()
            
            return self._get_observation(), {}

    def step(self, action_np: np.ndarray):
        """
        执行一步。SB3 会调用这个。
        """
        
        # SB3 给的是 numpy, 我们的动力学用 torch
        action_tensor = self._np_to_torch(action_np)

        action_tensor *=  self.max_speed_virtual_fish

        # --- 为了在 SB3 中高效运行，所有 torch 计算都应在 no_grad() 下 ---
        with torch.no_grad():
            
            # 1. 模块化：获取“行动前”的奖励信息
            # (例如：计算当前到目标的距离，用于下一步的奖励计算)
            reward_info = self._get_pre_step_reward_info()

            # 1. [极简调用] 动作转换 (Local -> Global)
            # 这一行替代了原来那一堆 sin/cos
            action_global = self.planner.transform_to_global(action_tensor)
            
            # 2. 物理推进 (使用 Global Action)
            new_s_v, new_s_r = self._run_dynamics(action_global)
            new_s_v = self._apply_reachable_set_cutoff(new_s_v, new_s_r)
            
            self.s_v = new_s_v
            self.s_r = new_s_r
            self.current_step += 1
            self.planner.step()

            # 5. 缓存渲染历史
            self.virtual_fish_history.append(self._torch_to_np(self.s_v[:2]))
            self.real_fish_history.append(self._torch_to_np(self.s_r[:2]))

            # 5. 模块化：获取“行动后”的观测
            observation_np = self._get_observation()

            # 6. 模块化：计算奖励
            reward = self._get_reward(reward_info, action_tensor)

            # 7. 模块化：检查是否结束
            terminated, truncated = self._check_done()

            info = {} # 可以添加调试信息
            
            # SB3 需要标准的 python float
            reward_float = float(reward)

            return observation_np, reward_float, terminated, truncated, info

    # ==================================================================
    # 模块化辅助函数
    # ==================================================================

    def _initialize_state_vectors(self):
        """简单初始化状态向量"""
        self.s_v = torch.zeros(4, device=self.device, dtype=self.dtype) + torch.tensor([0.02, 0.02, 0, 0])
        self.s_r = torch.zeros(4, device=self.device, dtype=self.dtype) + torch.tensor([0.01, 0.01, 0, 0])




    def _initialize_state(self):
        """
        模块化：重置所有内部状态变量
        """
        # 1. 重置鱼的初始状态 (例如在原点附近)
        # s = [x, y, vx, vy]
        self.s_v = torch.zeros(4, device=self.device, dtype=self.dtype) + torch.tensor([0.02, 0.02, 0, 0])
        self.s_r = torch.zeros(4, device=self.device, dtype=self.dtype) + torch.tensor([0.01, 0.01, 0, 0])
        # 你也可以在这里添加小的随机扰动
        

        
        # 3. 重置步数
        self.current_step = 0
        
        # 4. 重置规划器，并使其与 s_v 对齐
        initial_target = self.planner.reset(self.s_r)
        
        # 5. 重置奖励计算器
        initial_pos = self.s_v[:2]
        target_pos = initial_target[:2]
        self.prev_dist_to_target = torch.norm(initial_pos - target_pos)
        # 2. 重置上一步动作 (用于平滑奖励)
        self.prev_action = torch.zeros(2, device=self.device, dtype=self.dtype)

    def _get_observation(self) -> np.ndarray:
        """
        获取观测：完全依赖 Planner 进行坐标系归一化
        """
        # 1. 获取 Global 状态
        sv_pos, sv_vel = self.s_v[:2], self.s_v[2:]
        sr_pos, sr_vel = self.s_r[:2], self.s_r[2:]

        # 2. [极简调用] 状态转换 (Global -> Local)
        # Planner 内部已经存了 origin 和 rotation，直接调
        sv_pos_local, sv_vel_local = self.planner.transform_to_local(sv_pos, sv_vel)
        sr_pos_local, sr_vel_local = self.planner.transform_to_local(sr_pos, sr_vel)

        # 3. 组装 Observation (Local Frame)
        s_v_obs = torch.cat([sv_pos_local, sv_vel_local])
        rel_state_obs = torch.cat([sr_pos_local - sv_pos_local, sr_vel_local - sv_vel_local])
        
        # 4. [极简调用] 相对目标 (Local Frame)
        rel_target_local = self.planner.get_next_target_relative_local(self.s_v)
        
        obs_tensor = torch.cat([s_v_obs, rel_state_obs, rel_target_local], dim=0)
        return self._torch_to_np(obs_tensor)

    # def _run_dynamics(self, action_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     模块化：调用动力学模型
    #     """
    #     # 你的 dynamics_model.step() 负责返回 *新* 的状态
    #     new_s_v, new_s_r = self.dynamics_model.step(
    #         self.s_v, self.s_r, action_tensor, self.dt
    #     )
    #     return new_s_v, new_s_r


    def _run_dynamics(self, action_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        [关键修正]：支持旋转不变性的动力学推进
        
        问题：如果网络只在水平轨迹上训练，直接输入竖直轨迹的 Global 差值会导致预测偏差。
        解决：先将 Global 状态转回 Local，让网络在“熟悉”的局部坐标系下预测，再把结果转回 Global。
        """
        
        # --- 1. 将当前 Global 状态转为 Local 状态 ---
        # 注意：这里我们主要关心相对量，所以用 transform_to_local 即可
        # 但 dynamics 需要的是绝对速度 vector，位置可以用相对的
        
        # 获取当前 Global 状态
        s_v_global = self.s_v
        s_r_global = self.s_r
        
        # 利用 Planner 的接口转到 Local
        # Local Pos: 用于计算相对位置差
        # Local Vel: 用于计算相对速度差
        s_v_pos_local, s_v_vel_local = self.planner.transform_to_local(s_v_global[:2], s_v_global[2:])
        s_r_pos_local, s_r_vel_local = self.planner.transform_to_local(s_r_global[:2], s_r_global[2:])
        
        # 组装成 Local 状态向量 (用于喂给网络)
        s_v_local = torch.cat([s_v_pos_local, s_v_vel_local])
        s_r_local = torch.cat([s_r_pos_local, s_r_vel_local])
        
        # 同时也得把 action (global) 转成 local，因为网络是预测 local 下的反应
        # 这里实际上就是反向旋转一下速度矢量
        action_local = torch.matmul(self.planner.rotation_matrix.T, action_global)

        # --- 2. 在 Local Frame 下运行动力学模型 ---
        # 此时网络看到的输入是旋转归一化后的，就像在训练集里一样
        # 返回的 new_s_v_local, new_s_r_local 也是 Local 的
        new_s_v_local, new_s_r_local = self.dynamics_model.step(
            s_v_local, s_r_local, action_local, self.dt
        )
        
        # --- 3. 将预测结果转回 Global Frame ---
        # new_s_v_local 包含 (pos_local, vel_local)
        
        # 解包
        pred_sv_pos_local = new_s_v_local[:2]
        pred_sv_vel_local = new_s_v_local[2:]
        pred_sr_pos_local = new_s_r_local[:2]
        pred_sr_vel_local = new_s_r_local[2:]
        
        # 变换回 Global
        # Pos: Local -> Global (先正向旋转 R*P，再加 Origin)
        # 注意：dynamics model 算出的 pos 是相对于 transform 时的原点增量吗？
        # 不，它是绝对坐标。所以： Global = R * Local + Origin
        
        # 这里的 transform_to_global 只处理了向量旋转，没处理位置平移
        # 我们手动处理一下位置:
        pred_sv_pos_global = torch.matmul(self.planner.rotation_matrix, pred_sv_pos_local) + self.planner.traj_origin
        pred_sr_pos_global = torch.matmul(self.planner.rotation_matrix, pred_sr_pos_local) + self.planner.traj_origin
        
        # 速度直接旋转
        pred_sv_vel_global = self.planner.transform_to_global(pred_sv_vel_local)
        pred_sr_vel_global = self.planner.transform_to_global(pred_sr_vel_local)
        
        # 组装回 Global Tensor
        new_s_v_global = torch.cat([pred_sv_pos_global, pred_sv_vel_global])
        new_s_r_global = torch.cat([pred_sr_pos_global, pred_sr_vel_global])
        
        return new_s_v_global, new_s_r_global
    
    def _apply_reachable_set_cutoff(self, s_v: torch.Tensor, s_r: torch.Tensor) -> torch.Tensor:
        """
        模块化：对虚拟鱼应用可达集约束。
        将 s_v (虚拟鱼) 的位置限制在 s_r (真实鱼) 周围的 [rx, ry] 矩形内。
        这通过 clamp (裁剪) 相对位置来实现。
        """
        # 复制一份 s_v，我们只修改它的位置
        s_v_corrected = s_v.clone() 
        
        # 1. 计算相对位置 (v-fish pos - r-fish pos)
        rel_pos = s_v_corrected[:2] - s_r[:2]
        
        # 2. 使用 torch.clamp 来裁剪相对距离
        # torch.clamp(input, min, max)
        clamped_rel_x = torch.clamp(rel_pos[0], -self.rx, self.rx)
        clamped_rel_y = torch.clamp(rel_pos[1], -self.ry, self.ry)
        
        # 3. 从真实鱼的位置重构虚拟鱼的“被裁剪后”的绝对位置
        s_v_corrected[0] = s_r[0] + clamped_rel_x
        s_v_corrected[1] = s_r[1] + clamped_rel_y
        
        # (可选) 如果你还想在“撞墙”时把速度清零，
        # 你可以在这里添加逻辑来修改 s_v_corrected[2:4]
        # 但目前我们只约束位置
        
        return s_v_corrected


    # 测试旋转时候能否跟踪上的情况

    # def _apply_reachable_set_cutoff(self, s_v, s_r):
    #     s_v_corrected = s_v.clone() 
    #     rel_pos = s_v_corrected[:2] - s_r[:2]
        
    #     # 改用圆形约束验证
    #     dist = torch.norm(rel_pos)
    #     max_radius = 2 * self.rx # 假设 rx = ry
        
    #     if dist > max_radius:
    #         # 缩放回圆内
    #         scale = max_radius / (dist + 1e-8)
    #         rel_pos = rel_pos * scale
            
    #     s_v_corrected[0] = s_r[0] + rel_pos[0]
    #     s_v_corrected[1] = s_r[1] + rel_pos[1]
        
    #     return s_v_corrected

    def _get_pre_step_reward_info(self) -> dict:
        """
        模块化：在 step 发生前，缓存计算奖励所需的信息
        """
        # 我们缓存“上一步”到“当前目标”的距离
        return {
            'prev_dist': self.prev_dist_to_target,
            'prev_action': self.prev_action,
        }

    def _get_reward(self, reward_info: dict, action_tensor: torch.Tensor) -> torch.Tensor:
        """
        模块化：计算组合奖励
        """
        # --- 1. 轨迹跟踪奖励 (你描述的第一点) ---
        # “当前时刻离参考轨迹越近则差值越大”
        # 我们实现为 "Potential-based Reward"
        # R_track = (旧的到目标的距离) - (新的到目标的距离)
        
        # current_pos = self.s_v[:2]       # 虚拟鱼的位置作为奖励

        current_pos = self.s_r[:2]

        current_target_pos = self.planner.get_current_target()[:2]
        
        dist_now = torch.norm(current_pos - current_target_pos)
        dist_prev = reward_info['prev_dist']
        previous_action = reward_info['prev_action']


        
        # 奖励 = 距离的减少量
        tracking_reward = dist_prev - dist_now     # 势能奖励 nature 论文中的方案

        # tracking_reward = torch.exp(-1.0 * dist_now)  # Gemini 提供的方案


        

        
        # --- 2. 动作平滑奖励 (你描述的第二点) ---
        # 我推荐用简单的“二范数差值”，它可微且高效
        # R_smooth = - (a_t - a_{t-1})^2
        
        action_diff = action_tensor - previous_action
        smoothing_penalty = torch.sum(action_diff**2)

        # 更新缓存，供下一步使用
        self.prev_dist_to_target = dist_now
        self.prev_action = action_tensor

        # --- 3. 组合 ---
        total_reward = (self.reward_weights['tracking'] * tracking_reward - 
                        self.reward_weights['smoothing'] * smoothing_penalty)
        
        return total_reward

    def _check_done(self) -> tuple[bool, bool]:
        """
        模块化：检查是否终止
        """
        # 1. Terminated (任务成功)
        # (暂时不设成功条件)
        terminated = False
        
        # 2. Truncated (超时)
        truncated = self.current_step >= self.max_episode_steps
        
        return terminated, truncated

    # --- Torch/Numpy 转换工具 ---
    
    def _torch_to_np(self, tensor: torch.Tensor) -> np.ndarray:
        """
        (CPU) Tensor -> Numpy
        """
        return tensor.cpu().numpy().astype(np.float32)

    def _np_to_torch(self, arr: np.ndarray) -> torch.Tensor:
        """
        Numpy -> (Device) Tensor
        """
        return torch.from_numpy(arr).to(device=self.device, dtype=self.dtype)
    
        
    # def render(self):
    #     # _# --- 新增 (用于 Render) ---_
    #     if self.render_mode != "human":
    #         return

    #     # --- 1. 初始化绘图窗口 ---
    #     if self.fig is None:
    #         self.fig, self.ax = plt.subplots(figsize=(8, 8))
    #         self.fig.show()

    #     # --- 2. 清空上一帧 ---
    #     self.ax.clear()

    #     # --- 3. 绘制目标路径 (Target Path) ---
    #     # (我们绘制 *完整* 的对齐后轨迹)
    #     full_target_traj_np = self.planner.base_trajectory.cpu().numpy()
    #     offset = self.planner.pos_offset.cpu().numpy()
        
    #     aligned_target_x = full_target_traj_np[:, 0] + offset[0]
    #     aligned_target_y = full_target_traj_np[:, 1] + offset[1]
        
    #     self.ax.plot(
    #         aligned_target_x, 
    #         aligned_target_y,
    #         'k--',  # 颜色: 黑色 (k), 线型: 虚线 (--)
    #         label='Target Path',
    #         linewidth=2
    #     )

    #     # --- 4. 绘制虚拟鱼路径 (Virtual Fish Path) ---
    #     # (我们绘制 *历史* 轨迹)
    #     if self.virtual_fish_history:
    #         vfish_path_np = np.array(self.virtual_fish_history)
    #         self.ax.plot(
    #             vfish_path_np[:, 0], 
    #             vfish_path_np[:, 1],
    #             'b-o',  # 颜色: 蓝色 (b), 线型: 实线 (-), 标记: 圆圈 (o)
    #             label='Virtual Fish',
    #             markersize=4
    #         )

    

    #     # --- 5. 绘制真实鱼路径 (Real Fish Path) ---
    #     # (我们绘制 *历史* 轨迹)
    #     if self.real_fish_history:
    #         rfish_path_np = np.array(self.real_fish_history)
    #         self.ax.plot(
    #             rfish_path_np[:, 0], 
    #             rfish_path_np[:, 1],
    #             'C1-^', # 颜色: 橙色 (C1), 线型: 实线 (-), 标记: 三角 (^)
    #             label='Real Fish',
    #             markersize=4
    #         )

    #     # --- 6. 设置绘图属性 ---
    #     self.ax.set_title(f"Fish Guidance: Step {self.current_step}")
    #     self.ax.set_xlabel("X Position")
    #     self.ax.set_ylabel("Y Position")
    #     self.ax.legend()
    #     self.ax.grid(True)
    #     self.ax.axis('equal') # 保持 x, y 轴等比例，M 字母不会变形
        
    #     # --- 7. 刷新画布 ---
    #     self.fig.canvas.draw()
    #     plt.pause(0.001) # 必须加 pause 才能刷新

    def render(self):
        # _# --- 新增 (用于 Render) ---_
        if self.render_mode != "human":
            return

        # --- 1. 初始化绘图窗口 ---
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.fig.show()

        # --- 2. 清空上一帧 ---
        self.ax.clear()

        # --- 3. 绘制目标路径 (Target Path) ---
        # [关键修改] 直接使用 planner.active_trajectory
        # 因为在 planner.reset() 里，active_trajectory 已经被计算好（旋转+平移）了
        # 它就是真正的世界坐标系下的轨迹
        full_target_traj_np = self.planner.active_trajectory.cpu().numpy()
        
        self.ax.plot(
            full_target_traj_np[:, 0], # 直接画 X 
            full_target_traj_np[:, 1], # 直接画 Y
            'k--',  # 颜色: 黑色 (k), 线型: 虚线 (--)
            label='Target Path',
            linewidth=2
        )

        # --- 4. 绘制虚拟鱼路径 (Virtual Fish Path) ---
        # (我们绘制 *历史* 轨迹)
        if self.virtual_fish_history:
            vfish_path_np = np.array(self.virtual_fish_history)
            self.ax.plot(
                vfish_path_np[:, 0], 
                vfish_path_np[:, 1],
                'b-o',  # 颜色: 蓝色 (b), 线型: 实线 (-), 标记: 圆圈 (o)
                label='Virtual Fish',
                markersize=4
            )

        # --- 5. 绘制真实鱼路径 (Real Fish Path) ---
        # (我们绘制 *历史* 轨迹)
        if self.real_fish_history:
            rfish_path_np = np.array(self.real_fish_history)
            self.ax.plot(
                rfish_path_np[:, 0], 
                rfish_path_np[:, 1],
                'C1-^', # 颜色: 橙色 (C1), 线型: 实线 (-), 标记: 三角 (^)
                label='Real Fish',
                markersize=4
            )

        # --- 6. 设置绘图属性 ---
        self.ax.set_title(f"Fish Guidance: Step {self.current_step}")
        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")
        self.ax.legend()
        self.ax.grid(True)
        self.ax.axis('equal') # 保持 x, y 轴等比例，M 字母不会变形
        
        # --- 7. 刷新画布 ---
        self.fig.canvas.draw()
        plt.pause(0.001) # 必须加 pause 才能刷新


    def close(self):
        # _# --- 新增 (用于 Render) ---_
        if self.fig is not None:
            plt.close(self.fig)
            plt.ioff() # 关闭交互模式
            self.fig = None
            self.ax = None