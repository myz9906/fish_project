import os
import glob
import argparse
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib.pyplot as plt

from models import DynamicsModel

def parse_args():
    parser = argparse.ArgumentParser("Train dynamics model (virtual→real fish mapping)")
    parser.add_argument(
        '--csv_dir',
        type=str,
        default='./fish_data/Fish-system-identification',
        help='相对路径：fish_data/Fish-system-identification/*.csv'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='DynamicsModel_without_theta.pth',
        help='训练后模型保存路径'
    )
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--normalize', action='store_true',
                        help='是否对输入和标签进行归一化处理')  # action='store_true' 是一种特殊的设置方式，用在布尔型（flag）选项上，有为真，没有为假
    return parser.parse_args()

class FishDynamicsDataset(Dataset):
    def __init__(self, csv_dir, normalize=False):
        """
        Modifications:
        1. 计算初始速度时用前两帧差分，不再 pad 第0帧为 0。
        2. 构建输入时只遍历到 N-2, 保证所有输入特征与速度/朝向长度一致（全为 N-1）。
        3. 路径默认来自 --csv_dir 指定的 ./fish_data/Fish-system-identification 文件夹。
        4. 向量化差分、平滑与朝向计算，减少 Python 循环。
        5. 新增 normalize 参数，可对输入/标签做均值-方差归一化。
        """
        self.samples = []
        all_files = glob.glob(os.path.join(csv_dir, '*.csv'))
        if not all_files:
            raise FileNotFoundError(f"No CSV files in {csv_dir}")
        for fn in tqdm(all_files, desc="Loading CSV"):
            df = pd.read_csv(fn)
            for trail_id, subdf in df.groupby('trail'):
                subdf = subdf.reset_index(drop=True)
                if len(subdf) < 2:
                    continue
                xv = subdf['Virtualfish x'].values.astype(np.float32)
                yv = subdf['Virtualfish y'].values.astype(np.float32)
                xr = subdf['Realfish x'].values.astype(np.float32)
                yr = subdf['Realfish y'].values.astype(np.float32)
                N = len(xv)
                dt = 1.0 / 100.0

                # 1) 速度差分
                vvx = np.empty_like(xv)  # vx 表示 virtual fish 的 x 方向
                vvy = np.empty_like(yv)
                vrx = np.empty_like(xr)   # rx 代表 real fish 的x 方向
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

                # 构造输入数据，相对位置和相对速度：

                p_x_relative = np.empty_like(xv)
                p_y_relative = np.empty_like(yv)
                p_x_relative =  xv - xr
                p_y_relative=  yv - yr

                v_x_relative = np.empty_like(xv)
                v_y_relative = np.empty_like(yv)
                v_x_relative =  vvx_s - vrx_s
                v_y_relative =  vvy_s - vry_s




                # 4) 构建样本
                for t in range(N-1):
                    inp = np.array([
                        p_x_relative[t], p_y_relative[t], v_x_relative[t], v_y_relative[t]
                    ], dtype=np.float32)
                    tgt = np.array([vrx_s[t+1], vry_s[t+1]], dtype=np.float32)
                    self.samples.append((inp, tgt))

        if not self.samples:
            raise RuntimeError("No samples generated.")

        # 5) 归一化
        self.normalize = normalize
        if normalize:
            data_in = np.stack([s for s, _ in self.samples], axis=0)
            data_out = np.stack([t for _, t in self.samples], axis=0)
            self.in_mean = data_in.mean(axis=0)
            self.in_std  = data_in.std(axis=0) + 1e-6  # 避免除以0
            self.out_mean = data_out.mean(axis=0)
            self.out_std  = data_out.std(axis=0) + 1e-6
            for idx, (inp, tgt) in enumerate(self.samples):
                inp_n = (inp - self.in_mean) / self.in_std
                tgt_n = (tgt - self.out_mean) / self.out_std
                self.samples[idx] = (inp_n, tgt_n)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, tgt = self.samples[idx]
        return torch.from_numpy(inp), torch.from_numpy(tgt)

def train_dynamics(args):
    start_time = time.time()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    dataset = FishDynamicsDataset(args.csv_dir, normalize=args.normalize)
    print(f"Total samples: {len(dataset)} (normalize={args.normalize})")

    # 2) 自动推断网络输入维度
    network_input_dim = dataset[0][0].shape[0]
    print(f"Automatically inferred input dim: {network_input_dim}")

    # 3) 划分训练/测试集（80% 训练，20% 测试）
    test_ratio = 0.2
    test_size  = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f"Train samples: {train_size}, Test samples: {test_size}")

    # 4) DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    

    model = DynamicsModel(input_dim=network_input_dim, hidden_dim=64).to(device)      # 这个代码里面要手动控制神经网络的输入维数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()


    # 6) 训练循环
    epoch_losses = []
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            out      = model(inp)
            loss     = criterion(out, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inp.size(0)
        avg_loss = total_loss / train_size
        epoch_losses.append(avg_loss)
        print(f"[Epoch {epoch:04d}/{args.epochs}]  Train MSE: {avg_loss:.6f}")

    # 7) 保存模型
    save_dict = {'model_state': model.state_dict()}
    if args.normalize:
        save_dict.update({
            'in_mean':  dataset.in_mean.astype(np.float32),
            'in_std':   dataset.in_std.astype(np.float32),
            'out_mean': dataset.out_mean.astype(np.float32),
            'out_std':  dataset.out_std.astype(np.float32),
        })
    torch.save(save_dict, args.save_path)
    print(f"Saved dynamics model to {args.save_path}")

    # 8) 测试评估（仅在完全训练后）
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inp, tgt in test_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            out      = model(inp)
            test_loss += criterion(out, tgt).item() * inp.size(0)
    avg_test_loss = test_loss / test_size
    print(f"Test   MSE: {avg_test_loss:.6f}")

    # 9) 绘制训练曲线
    plt.figure()
    plt.plot(np.arange(1, args.epochs+1), epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE')
    plt.title('DynamicsModel Training Loss')
    plt.tight_layout()
    plt.show()

    elapsed = time.time() - start_time
    print(f"Training + evaluation completed in {elapsed:.1f} seconds.")

if __name__ == '__main__':
    args = parse_args()
    train_dynamics(args)
