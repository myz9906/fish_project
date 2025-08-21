## 第一步：

cd 到 fish_project 的文件夹

## 第二步：

### 创建环境
conda env create -f environment.yml

### 激活环境
conda activate fish_RL



# 文件说明

**interaction_train without_theta.py** 是用来训练真实鱼的动力学的代码

**models.py** 里面是模拟真鱼动力学的神经网络模型，目前用的是三层的MLP网络

**test_trained_model.py**: 可视化每个trail的数据，以及展示神经网络对动力学预测的效果，里面调用的神经网络模型就是DynamicsModel_without_theta.pth

**test_trained_model_Letter_M.py**：测试函数，用来查看当虚拟鱼走"M"字样时，真实鱼的运动路线是怎样的。如果说主函数是通过真实鱼反求虚拟鱼运动，那么这个测试函数就是正过来的情况：已知虚拟鱼的运动，看神经网络得到的真实鱼该怎么走。目前测试下来的结果说明至少真实鱼也能走一个M字样出来，至少说明神经网络训练能work

**model_utils.py**: 加载训练好的network模型

**sim.py**: 真鱼和虚拟鱼的动力学更新