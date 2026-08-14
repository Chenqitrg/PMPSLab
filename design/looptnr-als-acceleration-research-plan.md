# Loop-TNR ALS 收敛加速与并行化研究计划

## 状态与范围

本文提出一项针对 Loop-TNR 内层交替最小二乘（ALS）扫描的研究计划，重点解决
fixed point 附近收敛缓慢的问题，并评估能够安全引入的并行计算。本文不改变当前
`split` 算法，也不承诺任何候选算法最终会进入生产实现。

研究分为两个相互独立的问题：

1. 减少达到给定精度所需的 sweep 数量；
2. 降低 fixed-point 区域内单个 sweep 的计算成本。

这两个问题必须分别建立基线并分别验收，避免把“每轮更便宜”和“所需轮数更少”
混为一谈。

## 背景与初步观察

目前的经验是，一个高精度 Loop-TNR 优化通常需要约 40 个 sweep：

- 前 5 个 sweep 可使误差下降约 $10^3$；
- 后 35 个 sweep 总共只能再使误差下降约 $10$。

如果粗略假设每一阶段按固定比例收敛，那么平均收敛因子约为

$$
q_{\mathrm{early}} \approx 10^{-3/5} \approx 0.251,
\qquad
q_{\mathrm{tail}} \approx 10^{-1/35} \approx 0.936.
$$

因此，早期阶段每个 sweep 都能取得显著进展，而后期阶段每个 sweep 平均只改善
约 6.4%。这表明主要成本可能来自 fixed point 附近的慢收敛模，而不仅仅是单个
site update 的计算复杂度。

原始 Loop-TNR 算法参见
[Loop Optimization for Tensor Network Renormalization](https://arxiv.org/abs/1512.04938)。

## 研究目标

### 主要目标

1. 确定后期慢收敛是物理优化方向、gauge 漂移、病态局部方程，还是数值误差造成的。
2. 研究 safeguarded fixed-point acceleration，减少后期 sweep 数量。
3. 研究 Pairwise Perturbation，降低后期 environment 构造成本。
4. 测量不改变数学更新顺序的精确并行上限。
5. 评估 fixed-point acceleration 与 perturbative environment 是否能够组合。

### 非目标

本阶段不计划：

- 直接将所有 site 改为无保护的 Jacobi 同步更新；
- 假设小 tensor 更新自动意味着并行迭代稳定；
- 为了并行而改变最终优化目标；
- 在没有复杂度推导前显式构造巨大的 environment Jacobian；
- 将平台调度、分布式 job 管理或多用户资源隔离加入数值算法。

## 核心研究问题与假设

### RQ1：后期 slow tail 是否对应真实慢模？

需要区分以下几种可能性：

- 真实的物理优化慢方向；
- MPS gauge 漂移；
- tensor normalization 漂移；
- 病态 local normal equation；
- truncation 或 local solver tolerance 形成的数值平台区。

**假设 H1：** 在统一 gauge 后，完整 sweep map 的线性化仍然存在接近 1 的
dominant eigenvalue，因此 slow tail 是实际优化慢模，而不只是表示冗余。

### RQ2：fixed-point acceleration 能否减少后期 sweep 数？

候选方法包括：

- Anderson acceleration；
- nonlinear GMRES；
- ALS-preconditioned L-BFGS；
- Riemannian 或 preconditioned conjugate gradient。

**假设 H2：** 使用真实 cost 检查和 serial fallback 后，可以减少后期 sweep 数，
同时保持最终 cost 和物理量不劣于 serial ALS。

### RQ3：微扰展开能否降低单轮成本？

在参考状态 $\bar B$ 附近，将构造 site $i$ 子问题所需的 environment 写成

$$
E_i(B)
\approx
E_i(\bar B)
+
\sum_{j \ne i} R_{ij}[\Delta B_j]
+
O(\|\Delta B\|^2),
\qquad
\Delta B_j = B_j - \bar B_j.
$$

**假设 H3：** 在后期 tensor 变化足够小时，应用紧凑表示的 response operator
$R_{ij}$ 比重新进行完整 environment contraction 更便宜。

这是必须通过复杂度推导验证的假设。如果每个 $R_{ij}[\Delta B_j]$ 仍然需要一次
与完整 environment 同阶的 $O(D^6)$ contraction，那么该方案没有价值。

### RQ4：两条路线能否组合？

**假设 H4：** fixed-point acceleration 减少 sweep 数，Pairwise Perturbation
降低剩余 sweep 的成本，两种收益可以部分叠加。

## 阶段一：建立可重复基线

### 基准问题

至少选择三类代表性输入：

1. 非临界 Ising 模型；
2. 临界 Ising 模型；
3. 一个实际使用中收敛较慢的模型。

每类问题应测试多个 bond dimension，例如

$$
D \in \{8, 16, 32, \ldots\}.
$$

所有对比实验应固定：

- 初始 tensor；
- 随机种子；
- sweep 顺序；
- local solver tolerance；
- truncation 设置；
- Julia、PMPSLab 与主要依赖版本；
- BLAS 和 Julia thread 数量。

### 每个 sweep 的记录项

记录以下数值与性能数据：

- 全局 cost；
- 每个 site 的 local cost；
- 完整 sweep 时间；
- `environment(cacheBB, pb)` 时间；
- `environment(cacheAB, pa)` 时间；
- $W$ contraction 时间；
- `rdiv` 时间；
- directional cache update 时间；
- peak memory；
- raw tensor update norm；
- gauge-aligned tensor update norm；
- local optimality residual。

主要性能指标应使用 time-to-tolerance：

$$
T(\epsilon)
=
\text{达到指定相对误差阈值 } \epsilon \text{ 所需的 wall time}.
$$

同时单独报告 sweep 数和单 sweep 时间，从而识别收益来自算法收敛还是工程加速。

### 阶段产物

- 一个固定输入和随机种子的 benchmark 集；
- 每阶段 cost、residual 和 wall-time 曲线；
- 逐操作 profile；
- 前 40 个 sweep 的完整状态序列；
- 不同 $D$ 下的时间和内存 scaling。

## 阶段二：分析 fixed-point slow tail

### Gauge alignment

在比较连续状态 $B^k$ 和 $B^{k+1}$ 前，必须将它们对齐到一致 gauge。至少同时
记录

$$
\delta_{\mathrm{raw}}
=
\frac{\|B^{k+1}-B^k\|}{\|B^k\|}
$$

和

$$
\delta_{\mathrm{aligned}}
=
\min_{G \in \mathrm{Gauge}}
\frac{\|B^{k+1}-G \cdot B^k\|}{\|B^k\|}.
$$

如果 raw change 很大而 aligned change 很小，后期变化主要来自 gauge，而不是
真实优化方向。MPS tangent space 与 gauge 冗余可参考
[Post-Matrix Product State Methods: To Tangent Space and Beyond](https://arxiv.org/abs/1305.1894)。

### 估计局部收敛谱

把一个完整 serial sweep 看作 fixed-point map：

$$
B^{k+1}=G(B^k),
\qquad
r_k=G(B^k)-B^k.
$$

使用后期保存的状态与 residual：

1. 估计经验收敛因子；
2. 对 residual history 做 SVD；
3. 判断是否由少数 dominant slow modes 控制；
4. 必要时通过 finite-difference Jacobian-vector product 估计
   $DG(B^\star)v$ 的主特征值。

### 继续条件

只有满足以下条件，才进入 fixed-point acceleration 实验：

- gauge alignment 后仍有稳定的 slow mode；
- residual 明显高于数值噪声；
- local solver tolerance 没有提前截断收敛；
- 全局 cost 可以稳定、可靠地比较。

## 阶段三：Fixed-point acceleration

### 首选原型：Safeguarded Anderson acceleration

Anderson acceleration 不要求显式 Hessian，也不要求先实现完整 gradient。它只需要
若干历史状态和 residual，因此适合作为第一项实验。

研究历史深度

$$
m \in \{1,2,4,8\}.
$$

每次 acceleration 的流程为：

1. 完成一个正常 serial sweep，得到 $G(B^k)$；
2. 将历史 tensor 对齐到一致 gauge；
3. 解一个小型 least-squares，构造 accelerated candidate；
4. canonicalize 并 normalize candidate；
5. 计算真实全局 cost；
6. cost 下降则接受；
7. 否则减小 mixing coefficient；
8. 仍失败则使用普通 ALS 输出，并按策略清空历史。

基础参考为
[Anderson Acceleration for Fixed-Point Iterations](https://epubs.siam.org/doi/pdf/10.1137/10078356X)。
该工作讨论了 Anderson acceleration 与 GMRES、quasi-Newton 的关系。

### 对照方法：Nonlinear GMRES

N-GMRES 的基本结构为：

```text
普通 ALS tentative step
        ↓
组合历史状态产生 accelerated step
        ↓
line search
```

该方法已经被直接用于 tensor decomposition，尤其适合需要高精度 stationary point
的困难问题。参见
[A Nonlinear GMRES Optimization Algorithm for Canonical Tensor Decomposition](https://arxiv.org/abs/1105.5331)。

### 进阶方法：ALS-preconditioned L-BFGS

如果 Anderson 或 N-GMRES 显示出稳定收益，再研究将完整 ALS sweep 当作 nonlinear
preconditioner 的 L-BFGS。这一路线可能更强，但需要额外定义：

- gauge-consistent gradient；
- tangent-space inner product；
- retraction 或 canonicalization；
- line search；
- limited-memory history 的 checkpoint 表示。

参考
[Nonlinearly Preconditioned L-BFGS as an Acceleration Mechanism for ALS](https://arxiv.org/abs/1803.08849)。

### 成功标准

基础目标：

- slow-tail time-to-tolerance 至少降低 2 倍；
- 总 wall time 至少降低 1.5 倍；
- 最终 cost 不差于 baseline；
- 所有 accelerated candidate 都通过真实 cost 检查；
- 在完整 benchmark 集中不产生 NaN、space mismatch 或 normalization 失控。

挑战目标：

- 将典型的后 35 个 sweep 减少到 10 至 15 个；
- 在多个模型和 bond dimension 上保持稳定收益。

## 阶段四：Pairwise Perturbation 可行性研究

### 先推导，再实现

选择一个参考状态 $\bar B$。对于 BB transfer matrix，写成

$$
T_j^{BB}=T(B_j,\overline{B_j}).
$$

在 $B_j=\bar B_j+\Delta B_j$ 下展开：

$$
\begin{aligned}
T_j^{BB}
=\;&T(\bar B_j,\overline{\bar B_j})
+T(\Delta B_j,\overline{\bar B_j})\\
&+T(\bar B_j,\Delta\overline B_j)
+T(\Delta B_j,\Delta\overline B_j).
\end{aligned}
$$

随后推导周期 environment

$$
E_i=\prod_{j\ne i}T_j
$$

中的：

- reference environment；
- 单 site 一阶 response；
- 同 site 的二阶 BB correction；
- 不同 site 微扰之间的高阶交叉项；
- AB environment 对一对相邻 B tensor 的 response。

### 复杂度判定

对以下操作比较 FLOP、最大中间 tensor 和峰值内存：

1. 当前精确 environment：$E_i=R_iL_i$；
2. 一阶 perturbative correction；
3. reference response operator 的构造或刷新。

必须回答：

- correction 是否能降低 $D$ 的复杂度指数；
- response operator 能否保持为 lazy tensor-network contraction；
- 是否需要 $O(N^2)$ 个大对象；
- correction 是否能按来源 site 并行；
- reduction 与通信是否抵消计算收益。

### 停止条件

若出现以下任意情况，不继续实现完整 PP-ALS：

- 单个 correction 与完整 environment 同阶且常数接近；
- 每个目标 site 需要 $O(N)$ 个同样昂贵的 correction；
- response operator 内存超过 baseline peak memory 的约 2 至 3 倍；
- gauge 或 condition number 使误差无法可靠控制；
- reference operator 的刷新成本无法由 slow tail 摊销。

### 原型策略

若复杂度分析通过：

1. 前若干 sweep 使用 exact ALS；
2. gauge-aligned relative update 小于阈值时初始化 PP；
3. 使用一阶或局部二阶 correction；
4. 定期抽取一个 site 计算 exact environment；
5. 监测

   $$
   \eta_i
   =
   \frac{\|E_i^{PP}-E_i^{\mathrm{exact}}\|}
        {\|E_i^{\mathrm{exact}}\|};
   $$

6. 超过误差阈值时立即 refresh；
7. 结束前执行若干 exact polishing sweeps。

Pairwise Perturbation 的主要参考为
[Accelerating Alternating Least Squares for Tensor Decomposition by Pairwise Perturbation](https://arxiv.org/abs/1811.10573)。
该工作对近收敛区域的近似误差进行了分析，并报告最高约 3.1 倍的完整求解加速。

其分布式并行版本将 Pairwise Perturbation 与 dimension tree 结合，并在 1024 个
processors 上进行了测试：
[Efficient Parallel CP Decomposition with Pairwise Perturbation and Multi-sweep Dimension Tree](https://arxiv.org/abs/2010.12056)。

## 阶段五：不改变更新顺序的精确并行基线

在引入近似算法前，测量现有算法的精确并行上限。候选并行层次包括：

- BB 与 AB environment 的并行计算；
- contraction 内部多线程；
- symmetry sector 并行；
- independent RHS 并行；
- `rdiv` 或 factorization 使用 threaded BLAS；
- GPU tensor contraction；
- 独立 loop、plaquette 或 RG block 之间的并行。

至少测试

$$
n_{\mathrm{threads}} \in \{1,2,4,8\}.
$$

记录：

- wall-time speedup；
- parallel efficiency；
- peak memory；
- 不同线程数下的数值差异；
- nested threading 是否产生 oversubscription。

一般 CP-ALS 的大规模并行主要针对 MTTKRP/environment contraction，而不是同步更新
不同 factor。参考
[Parallel Nonnegative CP Decomposition of Dense Tensors](https://arxiv.org/abs/1806.07985)。

## 阶段六：组合算法

最终比较以下四个版本：

| 版本 | 减少 sweep 数 | 降低单 sweep 成本 |
|---|---:|---:|
| Serial exact ALS | 否 | 否 |
| Anderson 或 N-GMRES ALS | 是 | 否 |
| Pairwise-Perturbation ALS | 否或有限 | 是 |
| Accelerated PP-ALS | 是 | 是 |

候选最终流程为：

```text
Exact serial warm-up
        ↓
检测进入 slow-tail 区域
        ↓
Safeguarded Anderson / N-GMRES
        ↓
若 PP 可行，使用 approximate environments
        ↓
Exact polishing sweeps
```

Exact polishing 必须运行到与 baseline 相同的 stopping criterion，以降低 approximate
environment 对最终精度和收敛点的影响。

## 评估矩阵

每个候选算法都应在相同 benchmark 上报告：

| 指标 | 目的 |
|---|---|
| 达到每个误差阈值的 sweep 数 | 测量收敛加速 |
| 达到每个误差阈值的 wall time | 测量最终收益 |
| 单 sweep 时间 | 区分算法收益和 kernel 收益 |
| 最终全局 cost | 验证优化质量 |
| 最终物理 measurement | 验证物理一致性 |
| peak memory | 防止用不可接受的内存换时间 |
| candidate rejection 数 | 判断 accelerator 稳定性 |
| PP refresh 数与环境误差 | 判断微扰近似有效区间 |
| 不同线程数下的结果差异 | 判断数值可重复性 |

所有性能结论都应同时报告首次编译后的稳定运行结果，避免将 Julia compilation time
计入数值算法对比。

## Checkpoint 与暂停要求

如果候选算法进入实现，其附加状态也必须支持 checkpoint。

Anderson 或 N-GMRES 至少需要保存：

- 最近 $m$ 个 gauge-aligned states；
- residual history；
- mixing 参数；
- rejection 或 restart 状态。

Pairwise Perturbation 至少需要保存或重建：

- reference tensors；
- response operators 或重建它们所需的信息；
- cumulative perturbations；
- refresh counter。

如果 response operators 过大，可以只 checkpoint reference tensors，并在恢复时重建
operators。这样会增加恢复时间，但降低 checkpoint 大小。

如果要求暂停后与未暂停运行在机器精度范围内遵循相同轨迹，就必须保存 acceleration
history。只保存当前 tensor 虽然可以继续求解，但恢复后的 accelerator state 不同，后续轨迹
也可能不同。

## 决策顺序

建议按以下顺序推进，每一步完成后单独 review：

1. 建立 40-sweep baseline 与 profiler；
2. 实现连续状态的 gauge alignment 和 residual 记录；
3. 对后期 residual history 做 SVD，确认 slow-mode 结构；
4. 实现 depth-2 safeguarded Anderson 原型；
5. 扩展历史深度并与 N-GMRES 对比；
6. 仅在确认 environment 是主要成本后推导 Pairwise Perturbation；
7. 仅在复杂度推导显示收益后实现 PP 原型；
8. 最后研究 exact kernel parallelism 和组合算法。

第一项最小实验应是：保存现有 40 个 sweep 的完整状态，在统一 gauge 后对 residual
history 做 SVD，然后实现带真实 cost 回退的 depth-2 Anderson acceleration。这一步改动
相对较小，却能直接判断后期 35 个 sweep 是否主要由少数 slow modes 导致。

## 相关文章

### Loop-TNR 与 tensor ring

1. Yang, Gu, and Wen,
   [Loop Optimization for Tensor Network Renormalization](https://arxiv.org/abs/1512.04938)。
   原始 Loop-TNR 算法。
2. Zhao et al.,
   [Tensor Ring Decomposition](https://arxiv.org/abs/1606.05535)。
   Tensor ring 表示以及 ALS、block-wise ALS 等基础算法。
3. Chen, Li, and Lu,
   [Tensor Ring Decomposition: Optimization Landscape and One-loop Convergence of ALS](https://arxiv.org/abs/1905.07101)。
   讨论 tensor ring ALS 的非凸 landscape 和特定条件下的 one-loop convergence。
4. Yu and Li,
   [Practical Alternating Least Squares for Tensor Ring Decomposition](https://arxiv.org/abs/2210.11362)。
   讨论 normal equation 构造、中间数据膨胀与 QR 稳定化。

### Fixed-point 与 ALS acceleration

5. Walker and Ni,
   [Anderson Acceleration for Fixed-Point Iterations](https://epubs.siam.org/doi/pdf/10.1137/10078356X)。
   Anderson acceleration 的标准现代参考。
6. De Sterck,
   [A Nonlinear GMRES Optimization Algorithm for Canonical Tensor Decomposition](https://arxiv.org/abs/1105.5331)。
   使用 ALS tentative step、历史组合与 line search 加速 tensor decomposition。
7. De Sterck and Howse,
   [Nonlinearly Preconditioned L-BFGS as an Acceleration Mechanism for ALS](https://arxiv.org/abs/1803.08849)。
   将 ALS 作为 L-BFGS 的 nonlinear preconditioner。
8. Evans et al.,
   [A Proof That Anderson Acceleration Improves the Convergence Rate in Linearly Converging Fixed-Point Methods](https://arxiv.org/abs/1810.08455)。
   分析 Anderson 对线性收敛 fixed-point iteration 的一阶加速效果。

### Perturbative 与并行 ALS

9. Ma and Solomonik,
   [Accelerating ALS for Tensor Decomposition by Pairwise Perturbation](https://arxiv.org/abs/1811.10573)。
   利用近收敛区域的小更新近似构造 ALS 子问题。
10. Ma and Solomonik,
    [Efficient Parallel CP Decomposition with Pairwise Perturbation and Multi-sweep Dimension Tree](https://arxiv.org/abs/2010.12056)。
    将 Pairwise Perturbation、dimension tree 与分布式并行结合。
11. Ballard, Hayashi, and Kannan,
    [Parallel Nonnegative CP Decomposition of Dense Tensors](https://arxiv.org/abs/1806.07985)。
    以 MTTKRP、dimension tree 和 processor-grid 数据分布实现大规模 parallel ALS。

### Gauge 与几何优化

12. Haegeman, Osborne, and Verstraete,
    [Post-Matrix Product State Methods: To Tangent Space and Beyond](https://arxiv.org/abs/1305.1894)。
    MPS tangent space 与 gauge 冗余的基础参考。
13. Gao, Peng, and Yuan,
    [Riemannian Preconditioned Algorithms for Tensor Completion via Tensor Ring Decomposition](https://arxiv.org/abs/2302.14456)。
    使用 block-Hessian 近似构造 tensor-ring Riemannian preconditioner。
