# Adaptive GM-AEKF 与 PC-DH-LSTM-AEKF Python 复现

本项目实现面向航天器相对导航的自适应 Gauss-Markov 状态增广 EKF，并保留物理约束双头 LSTM-AEKF、GM-9D EKF 与 GM-9D + LSTM residual 作为学习辅助对照。代码覆盖 CW 名义模型、非线性相对二体真值模型、距离/仰角/方位角量测、Van Loan 过程噪声离散化、Q/R 分离自适应、9 维 Gauss-Markov 加速度状态增广 EKF、时变 `tau_k/q_drive,k` 自适应，以及同 seed 配对 Monte Carlo 统计。

## 安装与运行

```powershell
cd D:\LSTM-AEKF
python -m pip install -r requirements.txt
python run_major_revision.py --mode quick --mc-runs 30 --workers 8
```

论文规模正式命令：

```powershell
python -u run_major_revision.py --mode paper --mc-runs 30 --workers 8
```

`quick` 模式用于端到端核验，采用 500 步、20 条训练轨迹、48 个隐藏单元和 12 个 epoch。`paper` 模式采用 2500 步、320 条训练轨迹、128 个隐藏单元和 60 个 epoch。两种模式都强制至少 5 个训练 seed 和 30 个测试 seed；正式管线还自动执行扩展 GM 网格、分布内结构化非平稳测试、Adaptive GM-AEKF 消融、单因素边界扫描、Holm 校正、权重哈希与复现清单。

## 数据与测试场景

训练集覆盖随机正弦单阶跃、阴影切换、多谐波脉冲、状态相关阻力、旧版结构化复合扰动和新版可学习残差压力场景。正式主测试使用 `nonlinear_structured` 场景，即状态门控、定时脉冲拖尾、短周期微脉冲和阴影/阻力基底的分布内非平稳组合。该场景故意超出单一固定时间常数 GM-9D 的假设，但仍由种子和因果状态确定，目的是检验时变 GM 参数能否比固定 GM-9D 更好地兼顾平稳段平滑性和突变段响应速度。

## 实验矩阵

主要方法包括：

- Fixed EKF；
- Innovation-CM AEKF；
- Sage-Husa/Fading AEKF；
- Coupled-NIS Q/R AEKF；
- GM-9D EKF；
- GM-9D EKF + R-CM；
- Adaptive GM-AEKF；
- Known-scale EKF；
- PC-DH-LSTM-AEKF；
- GM-9D EKF + LSTM residual。

GM-9D 的时间常数和驱动噪声只在 64 条 LSTM 验证轨迹上从扩展网格选择，随后对主测试和边界扫描冻结。Adaptive GM-AEKF 在相同 9 维状态模型上，根据创新一阶相关、NIS 偏离和加速度状态差分在线调节 `tau_k` 与 `q_drive,k`，并保留 R-CM 以降低 Q/R 混淆。Adaptive 消融包括固定 `tau`、固定 `q_drive`、去掉创新相关项、去掉加速度差分项、去掉 R-CM 和关闭轴向驱动差异。LSTM 消融保留为诊断对照，不再作为唯一核心主张。

跨种子报告位置 RMSE、速度 RMSE、NIS/DOF、NEES/6、NIS 95% 覆盖率和 NEES 95% 覆盖率的均值、样本标准差及双侧 95% Student-t 置信区间。学习方法以 5 个“训练 seed 内 30 测试 seed 均值”为推断单位，确认性消融的 20 个主检验采用 Holm 校正。详细定义见 [EXPERIMENTS.md](D:/LSTM-AEKF/EXPERIMENTS.md)，完整参数见 [REPRODUCIBILITY.md](D:/LSTM-AEKF/REPRODUCIBILITY.md)。

## 正式结果边界

当前正式路线聚焦分布内结构化非平稳场景。`Adaptive GM-AEKF` 是与固定 `GM-9D EKF` 直接比较的主方法；`PC-DH-LSTM-AEKF` 和 `GM-9D EKF + LSTM residual` 用于展示学习补偿在不同滤波结构中的边际作用。跨扰动族 OOD 泛化不作为主结果，也不作为论文贡献主张。完整统计与谨慎解释见 [MONTE_CARLO_RESULTS.md](D:/LSTM-AEKF/MONTE_CARLO_RESULTS.md)。

## 输出

- `artifacts/major_revision_v9/training_ensemble.json` 与 `models/*.npz`：五次训练记录、权重和 SHA256；
- `artifacts/major_revision_v9/residual_training_ensemble.json`：匹配 R-CM shadow 的 GM-aware 总加速度监督、早停与验证集残差注入增益；
- `artifacts/major_revision_v9/gm_validation_tuning.json`：42 点网格、平台诊断与冻结参数；
- `artifacts/major_revision_v9/adaptive_gm_validation_tuning.json`：Adaptive GM-AEKF 在线参数候选、验证集评分与冻结参数；
- `artifacts/major_revision_v9/crossed_id.{json,csv}`：分布内交叉矩阵、分阶段汇总、LSTM 消融和 Adaptive GM 消融；
- `artifacts/major_revision_v9/single_factor_scan.{json,csv}`：四类边界扫描；
- `artifacts/major_revision_v9/reproducibility_manifest.json`：版本、完整配置、命令和哈希；
- `artifacts/major_revision_v9/audit_report.json`：训练 seed、模型哈希、GM 选参隔离、Holm 检验族与 CSV 完整性审计；
- `fig/11_single_factor_boundaries.*`、`fig/12_training_seed_summary.*`、`fig/13_holm_ablation_training_seeds.*`、`fig/14_consistency_coverage.*`、`fig/15_adaptive_gm_ablation.*`：主修订图。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖扰动逐轨迹随机化、RMS 控制单因素变体、连续白噪声样本协方差与 Van Loan $Q_d$ 一致性、加速度头误差协方差传播、LSTM 缓存等价性、GM 离散过程噪声半正定性、固定 GM 与 Adaptive GM 有限性、Adaptive 关闭自适应后严格退化为固定 GM，以及验证集调参不使用测试集。
