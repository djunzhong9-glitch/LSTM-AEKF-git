# 完整复现参数

本文 Major Revision 实验由下列命令完整生成：

```powershell
python -u run_major_revision.py --mode paper --mc-runs 30 --workers 8 --resume-models --resume-results
```

机器可读配置、五个训练实例、权重 SHA256、GM 选参记录和所有逐轨迹结果位于 `artifacts/major_revision_v3/`。

## 数据与网络

| 项目 | 设置 |
|---|---|
| 轨迹 | 2500 步，`dt=0.1 s`，总时长 249.9 s |
| 训练/验证 | 320 条完整轨迹，固定 256/64 划分 |
| 训练场景 | gaussian、piecewise、train、dropout、nonlinear 循环 |
| LSTM | 单层 NumPy LSTM，输入/隐藏/输出维度 `15/128/9` |
| 序列 | 2500 步完整 BPTT，无截断 |
| 隐状态 | 每条序列开始时 `h_0=c_0=0` |
| 输出 | acceleration 3、log10 Q scale 3、log10 R scale 3 |
| 激活 | `a_max*tanh(.)`；噪声尺度为 `2*tanh(.)` |
| `a_max` | `[1.5e-3,1.5e-3,1.0e-3] m/s^2` |

输入顺序为：标准化距离/仰角/方位角创新 3、`log10(normalized_innovation^2+1e-8)` 3、`log10(max(diag(P_pred)/p_ref,1e-12))` 6、量测掩码 3。`p_ref=[5^2,5^2,5^2,0.05^2,0.05^2,0.05^2]`。创新标准差分母下限为 `1e-15`，不使用额外 z-score。

## 训练

| 项目 | 设置 |
|---|---|
| 训练 seed | `20260719,20260791,20260863,20260935,20261007` |
| Optimizer | Adam，`beta1=0.9`，`beta2=0.999`，`epsilon=1e-8` |
| Learning rate | `2e-3` |
| Epoch / batch | `60 / 16` 条完整轨迹 |
| 梯度裁剪 | 全局 L2 范数 1.0 |
| Loss | `L_a + 0.7 L_QR + 0.02 L_smooth` |
| 模型选择 | 各实例恢复最低验证总损失对应权重 |

恢复最佳权重后，以 64 条固定验证轨迹上的加速度残差二阶矩估计 `Sigma_a`，特征值下限为 `1e-12`。测试时将其冻结，并在使用加速度头的预测中加入 `G Sigma_a G^T`。

## Q/R 与自适应参数

| 参数 | 设置 |
|---|---|
| 名义 Q PSD | `[2.0e-7,2.0e-7,1.2e-7] m^2/s^3` |
| Q log10 边界 | `[-1.2,0.8]` |
| 名义 R 方差 | `[4.0e-2,1.75459634e-7,1.75459634e-7]`，顺序为距离、仰角、方位角 |
| R log10 边界 | `[-2,2]` |
| NIS 反馈 | EMA `0.92`，增益 `0.08`，比值下限 `1e-6` |
| R 快速通道 | head EMA `0.86`；CM window `20`；CM weight `0.70`；update alpha `0.30`；最大 log10 步长 `0.12` |
| Q 慢速通道 | head EMA `0.94`；CM window `35`；相关阈值 `0.18`；update alpha `0.10`；CM weight `0.12`；校正邻域 `±0.18`；最大 log10 步长 `0.015` |

融合顺序与论文伪代码一致。R 依次执行 head EMA、NIS、CM、线性融合、速率限制和边界；Q 依次执行 head EMA、相关触发的有限 CM、线性融合、速率限制、边界与 Van Loan 离散化。

## 随机性与统计

- 五个训练实例共享数据和划分，只改变初始化与 batch 顺序。
- ID/OOD 使用 `22260719:22260748`；单因素扫描使用 `23260719:23260748`。
- 主结果使用 25 s burn-in 后区间，并同时保存 full/early/middle/late。
- 学习方法 CI 以 5 个“训练 seed 内 30 测试 seed 均值”为推断单位。
- 传统基线 CI 以 30 个测试 seed 为单位。
- 20 个确认性消融检验采用 Holm 校正。

## 正式冻结记录

- GM-9D：`tau=15 s`，`q_g=3.2e-8 m^2/s^5`；原始最优相同，1% 近优集合含 2 个双轴内部点，测试集未参与选择。
- 训练 seed `20260719`：验证损失 `0.07652518`，SHA256 `75db3a1111d09b989fd0330d586141a19c9b863bab25d5f11f0ecdf6e83ab296`。
- 训练 seed `20260791`：验证损失 `0.07495531`，SHA256 `afad7f1c1e3bb8c2cb6ea48db0c60bb35296e5922f8ae836e3dac41003db042e`。
- 训练 seed `20260863`：验证损失 `0.07420323`，SHA256 `c1ca170cb62f39972e5a5ca7d2e387d0bc36362d91385a4e3e6c1af84cdf533e`。
- 训练 seed `20260935`：验证损失 `0.07582076`，SHA256 `be513574435a3e96b681a7b52a6bf74f14a35fe63fbb31c0b2cee466cf8749a0`。
- 训练 seed `20261007`：最佳 epoch `60`，验证损失 `0.07772410`，SHA256 `dde6d0d9565be057f516c20caf1c784b6f9ebc7b5339e02a7cf03f7b09d7271c`。

前四个最佳权重在长时任务中断前已写盘，恢复时重新核验固定划分、验证损失、加速度残差协方差与 SHA256；其逐 epoch 曲线未持久化，也未追溯性补造。第五个实例保留完整曲线。`artifacts/major_revision_v3/audit_report.json` 中全部完整性检查为通过。
