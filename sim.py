import torch
import math

def step_dynamics(s_v, s_r, a_v, model, dt):
    """
    s_v, s_r: (5,) 张量
    a_v: (2,) 张量
    model: 通过 load_dynamics_model() 加载好了的模型
    dt: float
    """

    # 1) 更新虚拟鱼
    sv = s_v.clone()
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

    # 2) 归一化 + 前向， 得到网络学到的真实鱼的输出
    inp = torch.cat([sv, s_r], dim=0)
    if hasattr(model, 'stats') and 'in_mean' in model.stats:
        inp = (inp - model.stats['in_mean']) / model.stats['in_std']

    with torch.no_grad():
        pred_n = model(inp.unsqueeze(0))[0]

    # 3) 反归一化， 得到真实鱼在物理空间上的输出
    if hasattr(model, 'stats') and 'out_mean' in model.stats:
        v_r = pred_n * model.stats['out_std'] + model.stats['out_mean']
    else:
        v_r = pred_n

    # 4) 更新真实鱼
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

def step_dynamics_without_theta(s_v, s_r, a_v, model, dt):
    """
    s_v, s_r: (5,) 张量
    a_v: (2,) 张量
    model: 通过 load_dynamics_model() 加载好了的模型
    dt: float
    """

    # 1) 更新虚拟鱼
    sv = s_v.clone()
    vx, vy = a_v[0], a_v[1]
    sv[2:4] = a_v

    sv[0:2] = sv[0:2] + a_v * dt


    # 预测真实鱼的轨迹

    # 2) 归一化 + 前向， 得到网络学到的真实鱼的输出
    inp = torch.cat([s_r[0:2]-sv[0:2], s_r[2:4]-sv[2:4]], dim=0)
    if hasattr(model, 'stats') and 'in_mean' in model.stats:
        inp = (inp - model.stats['in_mean']) / model.stats['in_std']

    with torch.no_grad():
        pred_n = model(inp.unsqueeze(0))[0]

    # 3) 反归一化， 得到真实鱼在物理空间上的输出
    if hasattr(model, 'stats') and 'out_mean' in model.stats:
        v_r = pred_n * model.stats['out_std'] + model.stats['out_mean']
    else:
        v_r = pred_n

    # 4) 更新真实鱼
    sr = s_r.clone()
    sr[2:4] = v_r
    sr[0:2] = sr[0:2] + v_r * dt


    return sv, sr