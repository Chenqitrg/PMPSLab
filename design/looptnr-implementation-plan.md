# LoopTNR 分阶段实现计划

## 状态与用途

**状态：仅设计，尚未实施。**

本文用于把当前已经讨论并确认的 LoopTNR 设计交接给后续 GPT session。后续
session 应按本文的阶段顺序逐步实现，每次只完成一个可独立验收的小阶段，不应在
一次修改中同时完成整个算法。

本文的直接目标是建立以下完整链路：

```text
TensorCell
→ U-shaped PeriodicMPS target
→ resumable PeriodicSplit child run
→ user-defined LoopTNR coarse graining
→ CheckerboardGates
→ in-place normalization
→ fixed-point observation
→ next RG step
```

第一版必须复用现有 execution infrastructure，使 LoopTNR 可以：

- 在任意 split site 完成后暂停；
- 在任意完整 RG 完成后暂停；
- snapshot、checkpoint 和 restore；
- 将每个 RG 的 observation 通过 observer event 即时发送给前端绘图；
- 在恢复后保留已经得到且无法重建的 observation history。

本文不实现任何源码，不确定具体 coarse-graining contraction 公式，也不实现
conformal transfer matrix。

## 已确定的设计决策

以下决策已经由项目维护者确认，后续实现不应重新发明另一套方案。

1. LoopTNR 输入支持 `UniformGates`、`CheckerboardGates` 和
   `PlaquetteGates`。
2. 三种输入都先通过现有 `PeriodicMPS(cell)` 转为四个二物理腿 target
   tensor。
3. TensorCell 到 PMPS 的顺序是 U-shape：
   `NW → SW → SE → NE → NW`。
4. target 通过现有 `PeriodicSplit` 优化为八个单物理腿 PMPS tensor。
5. coarse graining 不是 `pair_contraction`；标准路径将八个 optimized tensor
   收缩为两个四腿 tensor，输出 `CheckerboardGates`。
6. 具体 coarse-graining contraction 由项目维护者亲自实现。其他 session 不得
   猜测箭头、dual、旋转或 black/white 分组。
7. 第一版不单独实现 entanglement filtering，也不在 split 前调用 target
   `gaugefix!`。
8. 默认 split ansatz 使用 `PeriodicGaugeAnsatz`：
   - SVD 初始化只使用 `truncerror(rtol=1e-13)`；
   - `ProjectorCanonical` 使用
     `truncrank(maxrank) & truncerror(rtol=1e-13)`；
   - 普通 SVD 不限制 rank，low-rank 限制发生在 periodic projector gauge 中。
9. `PeriodicGaugeAnsatz` 只是默认值。核心 `LoopTNR` 构造器允许调用者传入使用
   其他 ansatz、solver、environment policy 或 stopping criterion 的完整
   `PeriodicSplit`。
10. 默认 sweep 是 `CircularSweep(8; maximal_sweep=40)`。
11. `maximal_rg` 必须显式提供，表示最多执行多少次 coarse-graining RG
    transformation。
12. `maximal_rg=0` 仍然执行 RG 0 normalization 和 observation，因此最终
    observation history 长度为一。
13. 未来可以加入 outer RG stopping criterion，但第一版只使用显式
    `maximal_rg` hard cap，不预留无行为的 stopping 字段。
14. RG 0 也必须计算 observation。
15. 每轮先原地 normalize 当前候选 TensorCell，再从 normalized tensor 计算
    fixed-point observable。
16. normalization 使用所有独立存储 gate tensor 的 infinity norm 几何平均，
    不使用 2×2 torus contraction，也不使用 Frobenius norm。
17. conformal spectrum 暂时只保留为 future fixed-point observable interface，
    第一版不实现。
18. `LoopTNRLog` 只存控制和收敛数据，例如时间、cost 和 child `SplitLog`；
    不存 TensorCell，也不存 outer fixed-point observation。
19. 现有 `SiteLog.observable` 必须保留。它属于更内层的 split/site observable，
    未来可记录 loop entanglement entropy 等诊断量，不等同于 outer RG
    observation。
20. outer `ObservationHistory` 是 durable scientific result。历史 RG tensor
    不再存在时 observation 无法重建，因此 snapshot 和 result 都必须保存它。
21. `LoopTNRResult` 使用命名字段 struct，不返回长 tuple。
22. blocking convenience API 名为 `renormalize`。
23. 前端回调继续使用 execution observer，不向数值配置、run 或 snapshot 中加入
    callback。
24. observer 异常沿用现有 execution 语义：在最近一次已经提交的安全边界停止并
    将 session 标记为 failed，不静默吞掉异常。

## 当前可复用基础设施

实现前应先阅读以下文件，不要复制它们已经解决的问题。

| 现有能力 | 位置 | LoopTNR 用法 |
| --- | --- | --- |
| TensorCell containers 与空间一致性 | `src/tensorcell/container.jl` | 输入、当前 RG cell、coarse-grain 输出 |
| U-shaped TensorCell → PMPS | `src/mps/tensorcell.jl` | 每轮构造四站点 split target |
| Periodic split configuration | `src/split/interface.jl` | `LoopTNR.split` 字段 |
| Gauge ansatz | `src/split/ansatz.jl` | 默认 low-rank 初始化 |
| Resumable split child | `src/split/run.jl` | 每个 outer `step!` 委托一个 site |
| Split snapshot identity | `src/split/identity.jl` | active child 的恢复验证 |
| Run/snapshot/event protocol | `src/execution/interface.jl` | outer run 必须实现同一接口 |
| Observer adapter | `src/execution/observer.jl` | site/RG event 实时通知 |
| Local pause/cancel | `src/execution/local_session.jl` | 不新增 LoopTNR 专用 task/lock |
| Atomic checkpoint codec | `src/execution/checkpoint.jl` | 注册 `LoopTNRSnapshot` payload |
| Parent-child composition proof | `test/composition_test.jl` | outer-child 状态机参考 |
| Split logs | `src/logs.jl` | `RGLog` 可包含已完成 child `SplitLog` |

`test/composition_test.jl` 已经证明父 run 可以在 child 内部和 child 交界处暂停、
snapshot 和恢复。LoopTNR 应把这个测试模式落到真实 `SplitRun` 上，而不是再建一套
执行框架。

## 数值流程

### RG 0

`initialize_run` 只复制输入并建立 cursor，不执行 normalization 或 observation。
第一次 outer `step!` 完成 RG 0：

```text
copy internal input into a local candidate
→ normalize candidate in place
→ evaluate fixed-point observable on normalized candidate
→ construct RG-0 observation and control log
→ atomically commit candidate, history and log
→ emit LoopTNRStepCompleted(rg=0)
```

必须用 local candidate。若 fixed-point observation 抛错，不能让 `run.cell` 已经
被 normalize 而 normalization result 尚未进入 history；否则重试时原始 scale 已
丢失。

### RG 1 到 `maximal_rg`

每轮流程为：

```text
current normalized TensorCell
→ PeriodicMPS(cell)
→ initialize child SplitRun
→ delegate one child site per outer step!
→ child becomes complete
→ coarse_grain(optimized child result)
→ assertconsistency(candidate CheckerboardGates)
→ normalize candidate in place
→ evaluate fixed-point observable
→ atomically commit cell, RGLog and observation
→ emit LoopTNRStepCompleted(rg=k)
```

最后一个 child site 和 coarse graining 必须是两个不同的 outer safe step：

1. final child site 完成后，outer `step!` 返回
   `LoopTNRSplitStepCompleted`，phase 进入 coarse grain；
2. 下一次 outer `step!` 才执行 coarse grain、normalization、observation 和 RG
   commit，并返回 `LoopTNRStepCompleted`。

这样用户既能在最后一个 site 后暂停，也能在完整 RG commit 后暂停。

## 默认算法配置

### 核心配置类型

目标形状：

```julia
struct LoopTNR{
        S<:PeriodicSplit,
        O<:AbstractObservable,
    } <: AbstractAlgorithmConfig
    split::S
    maximal_rg::Int
    observable::O
end
```

inner constructor 至少验证：

- `maximal_rg >= 0`；
- `sweep_domain(split.sweep) == Base.OneTo(8)`。

不要限制 `split.ansatz` 必须是 `PeriodicGaugeAnsatz`。自定义 split 是明确支持的
研究接口。

### 完整自定义构造器

```julia
LoopTNR(
    split::PeriodicSplit;
    maximal_rg::Integer,
    observable::AbstractObservable=TNRObservable(),
)
```

### 默认便利构造器

```julia
LoopTNR(
    maxrank::Integer;
    maximal_rg::Integer,
    observable::AbstractObservable=TNRObservable(),
)
```

它应生成等价于以下配置的 `PeriodicSplit`：

```julia
svd_truncation = truncerror(rtol=1e-13)
projector_truncation =
    truncrank(maxrank) & truncerror(rtol=1e-13)

ansatz = PeriodicGaugeAnsatz(
    SVDAnsatz(svd_truncation),
    ProjectorCanonical(; trscheme=projector_truncation),
)

split = PeriodicSplit(
    ansatz,
    CircularSweep(8; maximal_sweep=40),
    CholeskySolve();
    stopping=CostConvergence(
        atol=0,
        rtol=1e-10,
        min_sweeps=5,
        consecutive_sweeps=2,
    ),
)
```

`maxrank` 必须为正整数。需要不同 solver、environment、sweep 或 stopping 的调用者
应先构造 `PeriodicSplit`，再调用核心构造器；便利构造器不要不断增加透传 keyword。

## Observable 体系

### 分层原则

项目需要区分两类 observable：

1. **split/site observable**：与某个 site update 对齐，保存在现有
   `SiteLog.observable` 中；
2. **outer fixed-point observation**：从每个 normalized RG TensorCell 计算，
   保存在独立 `ObservationHistory` 中。

不要删除或复用 `SiteLog.observable`。不要把 outer observation 塞进
`LoopTNRLog`。

### 通用接口

建议在独立 generic observable 层定义：

```julia
abstract type AbstractObservable <: AbstractAlgorithmConfig end
abstract type AbstractObservation end

function observe end
```

fixed-point 配置可以进一步使用：

```julia
abstract type AbstractFixedPointObservable <: AbstractObservable end
```

第一版允许 `TNRObservable.fixed_point === nothing`，并定义 no-op 行为返回
`nothing`。不要为了未来 conformal spectrum 现在就写空数组或虚假的物理结果。

### TNR observable 配置

```julia
abstract type AbstractNormalization <: AbstractAlgorithmConfig end

struct InfinityNormNormalization <: AbstractNormalization end

struct TNRObservable{N,F} <: AbstractObservable
    normalization::N
    fixed_point::F
end

TNRObservable() =
    TNRObservable(InfinityNormNormalization(), nothing)
```

字段名必须是 `fixed_point`，不能写死为 `conformal`。未来可能加入：

- conformal spectrum；
- corner spectrum；
- fixed-point distance；
- ground-state degeneracy；
- 其他不依赖 conformal interpretation 的量。

### Observation 输出

建议输出类型：

```julia
struct NormalizationObservation{T<:Real} <: AbstractObservation
    scale::T
    log_scale::T
end

struct TNRObservation{N,F} <: AbstractObservation
    rg::Int
    normalization::N
    fixed_point::F
end

mutable struct ObservationHistory
    values::Vector{TNRObservation}
end
```

`TNRObservable` 是不可变算法参数；`TNRObservation` 是某一 RG 的不可重建科学
结果。两者不能合并为同一个 mutable 对象。

history invariant：

```text
values[1].rg == 0
values[k + 1].rg == k
length(values) == completed_rg + 1
```

observation 一旦 append 就视为只读。前端 observer 不得修改 event 中共享的
observation 内容。

### Future conformal implementation

conformal transfer matrix 暂不实现。未来应参考维护者原先编写、后来整理过的
TNRKit `src/utility/cft.jl`，以新的 fixed-point observable subtype 接入：

```julia
ConformalObservable <: AbstractFixedPointObservable
ConformalObservation <: AbstractObservation
```

加入它时不应修改 `LoopTNRRun` 状态机、`LoopTNRLog` 或 snapshot 的基本结构。

## TensorCell normalization

### 数学定义

令当前 TensorCell 中按对象 identity 去重后的独立存储 gate tensor 为
`G₁, …, Gₘ`。定义：

```math
a_j = \lVert G_j \rVert_\infty,
\qquad
\log s = \frac{1}{m}\sum_{j=1}^{m}\log a_j,
\qquad
s = \exp(\log s).
```

然后对每个独立 tensor 原地执行：

```julia
rmul!(gate, inv(scale))
```

并返回 `NormalizationObservation(scale, log_scale)`。

TensorKit 的 `norm(gate, Inf)` 是 tensor coefficient 最大绝对值，这正是这里所需
定义，不是矩阵 induced operator norm。

### 独立存储 tensor

逻辑上需要支持：

- `UniformGates`：一个 gate；
- `CheckerboardGates`：black 和 white；
- `PlaquetteGates`：NW、NE、SW、SE。

但 concrete fields 可能意外引用同一个 mutable `TensorMap`。normalization 必须按
`===` 去重，确保同一对象只参与几何平均一次，也只被缩放一次。不能简单遍历周期
matrix，因为 Uniform 和 Checkerboard 的 periodic indexing 会重复返回 tensor。

### 错误行为

在修改任何 tensor 之前先计算并验证全部 norms：

- 任一 norm 为零：抛 `ArgumentError`；
- 任一 norm 非有限：抛 `DomainError`；
- 计算出的 `log_scale` 或 `scale` 非有限：抛 `DomainError`。

验证完成后才原地缩放，避免部分 tensor 已修改而后续 tensor 验证失败。

### 验收性质

- 返回的 `scale` 与直接几何平均一致；
- 极大/极小 norm 混合时不因先乘积而 overflow/underflow；
- normalize 后独立 gates 的 infinity norm 几何平均约等于一；
- gate object identity 保持不变；
- alias gate 不会被缩放两次；
- 输入为零或非有限时 cell 不发生部分修改。

## Coarse-graining 数值接缝

### 公共接口

先声明 generic function：

```julia
function coarse_grain end
```

LoopTNR 所需方法接受八站点单物理腿 `PeriodicMPS`，返回
`CheckerboardGates`。outer run 在调用前后负责结构检查：

- 输入长度必须为八；
- 每个 optimized tensor 必须是单物理腿 MPS tensor；
- 输出必须是 `CheckerboardGates`；
- 输出必须通过 `assertconsistency`。

### 当前 U-shape 物理腿顺序

现有 `PeriodicMPS(cell)` 与 split ordering 共同给出八个物理腿：

```text
1 = North of NW
2 = West  of NW
3 = West  of SW
4 = South of SW
5 = South of SE
6 = East  of SE
7 = East  of NE
8 = North of NE
```

该顺序只用于帮助维护者实现 contraction，不授权其他 session 根据 TNRKit 旧代码
猜测 transpose。

### 明确禁止

- 不得用 `pair_contraction(optimized, 1)` 或
  `pair_contraction(optimized, 2)` 代替 coarse graining；
- 不得把四个 pair contraction tensor 强行解释为 `PlaquetteGates`；
- 不得在没有维护者确认的情况下照搬 TNRKit `ΨB_to_TATB`，因为本项目 PMPS
  arrow 和 U-shape convention 已重新选择。

### 维护者负责的阶段

具体 contraction 是本计划唯一明确保留给项目维护者的数值实现。其他 session 可以
先完成接口、normalization、RG 0、日志和状态类型，但在该 contraction 合入前不要
声称 multi-RG LoopTNR 已完成。

## Log、history 与 result

### Outer control log

建议：

```julia
struct RGLog{S}
    rg::Int
    split_log::S
    coarse_grain_time::Float64
    normalization_time::Float64
    fixed_point_time::Float64
end

mutable struct LoopTNRLog{L} <: ConvergenceLog
    rg_logs::Vector{L}
end
```

RG 0 使用：

```text
split_log = nothing
coarse_grain_time = 0
```

后续 RG 的 `split_log` 保存完整 child `SplitLog`。final split cost 可从
`last(split_log.site_logs).error` 读取，不必在 `RGLog` 中复制同一个数值。

如果自定义 split 有零个 site update，final cost 不存在，显示层必须允许
`nothing`。

### Split site observable

现有 `SiteLog.observable` 保留。当前 `SplitLog` alias 把 observable 固定为
`Nothing`，未来真正加入 entanglement entropy 时需要单独拓宽 `SplitLog` 和
`SplitRun` 的 log type。

如果未来 site observable 含 mutable vector，必须同时修复
`_copy_split_log` 的浅复制问题。当前 `copy(log.site_logs)` 只在 site log payload
不可变或为 `nothing` 时足够。

### Result

```julia
struct LoopTNRResult{C,H,L}
    cell::C
    observations::H
    log::L
end
```

result invariant：

- `cell` 是最终 normalized TensorCell；
- `observations` 包含 RG 0 到最终 RG 的全部 scientific result；
- `log` 包含同样 RG 范围内的 control records；
- result 不保存每轮历史 tensor。

不保存 tensor history 是刻意设计。若未来需要逐轮 tensor archive，应建立独立
artifact/storage policy，而不是让 log 或 observation history 无限膨胀。

## Outer run 状态机

### Phase

建议显式 enum：

```julia
@enum LoopTNRPhase::UInt8 begin
    TNR_INITIALIZE
    TNR_PREPARE_SPLIT
    TNR_OPTIMIZE
    TNR_COARSE_GRAIN
    TNR_FINISHED
end
```

### Run

概念形状：

```julia
mutable struct LoopTNRRun{A<:LoopTNR} <: AbstractAlgorithmRun
    algorithm::A
    cell::AbstractTensorCell
    completed_rg::Int
    phase::LoopTNRPhase
    child::Union{Nothing,SplitRun}
    log::LoopTNRLog
    observations::ObservationHistory
end
```

`cell` 和 `child` 有意在 outer orchestration boundary 使用 type-erased field：
coarse graining 和 truncation 会改变具体 virtual spaces 与 concrete TensorMap type，
不能假设所有 RG 都具有同一个 concrete cell type。

初始化状态：

```text
cell = copy(input)
completed_rg = -1
phase = TNR_INITIALIZE
child = nothing
log/history empty
```

### Phase invariants

| Phase | `completed_rg` | Child | 含义 |
| --- | ---: | --- | --- |
| `TNR_INITIALIZE` | `-1` | `nothing` | RG 0 尚未提交 |
| `TNR_PREPARE_SPLIT` | `0:maximal_rg-1` | `nothing` | 下一轮 child 尚未建立 |
| `TNR_OPTIMIZE` | `0:maximal_rg-1` | active, unfinished | 下一步委托一个 site |
| `TNR_COARSE_GRAIN` | `0:maximal_rg-1` | complete | optimized PMPS 等待 RG commit |
| `TNR_FINISHED` | `maximal_rg` | `nothing` | result 可读取 |

任意 snapshot 或 restore 都必须验证这些组合，而不是只相信 enum。

### `step!` transition

#### `TNR_INITIALIZE`

完成 RG 0 transaction，返回 `LoopTNRStepCompleted`。若
`maximal_rg == 0`，进入 `TNR_FINISHED`；否则进入
`TNR_PREPARE_SPLIT`。

#### `TNR_PREPARE_SPLIT`

从 `PeriodicMPS(run.cell)` 建立 child `SplitRun`。

- child 未完成：立即委托第一个 child site，返回 nested site event；
- child 因自定义 zero-length sweep 在初始化时已完成：进入
  `TNR_COARSE_GRAIN`，同一次 outer step 可以继续执行 coarse-grain
  transaction 并返回完整 RG event。

#### `TNR_OPTIMIZE`

调用一次 `step!(child)`。

- child 仍未完成：保持 phase；
- child 刚完成：phase 改为 `TNR_COARSE_GRAIN`，但本次 outer step 不执行
  coarse graining。

两种情况都返回 `LoopTNRSplitStepCompleted`。

#### `TNR_COARSE_GRAIN`

从 complete child 读取 optimized PMPS，在 local variables 中完成：

```text
coarse_grain
→ consistency validation
→ normalization
→ fixed-point observation
→ RGLog construction
```

全部 fallible numerical work 成功后才 commit：

```text
run.cell = candidate
append log
append observation
run.completed_rg += 1
run.child = nothing
phase = FINISHED or PREPARE_SPLIT
```

返回 `LoopTNRStepCompleted`。

#### `TNR_FINISHED`

拒绝 `step!`，抛 `ArgumentError`。

### Transaction rule

coarse graining、normalization、fixed-point observation 或 log construction 任一
失败时：

- 当前 committed `run.cell` 不变；
- `completed_rg` 不变；
- log/history 长度不变；
- complete child 保留；
- phase 保持 `TNR_COARSE_GRAIN`，允许检查或重试。

RG 0 同样遵守这一规则。

## Events 与前端绘图

### Nested site event

```julia
struct LoopTNRSplitStepCompleted{E<:AlgorithmEvent} <: AlgorithmEvent
    rg::Int
    event::E
end
```

第一版 `event` 通常是 `SplitSiteCompleted`。它使前端可以显示：

```text
LoopTNR RG 4/20
  Split sweep 2/40
    Site 7/8 completed
```

### RG event

```julia
struct LoopTNRStepCompleted{L,O} <: AlgorithmEvent
    rg::Int
    log::L
    observation::O
end
```

RG 0 和后续 RG 使用同一个 event type。前端只需过滤
`LoopTNRStepCompleted` 就能实时追加 normalization 或 future fixed-point 曲线。

event 不保存 TensorCell。observer 对 event payload 只有只读权限。

### Observer failure

observer 在 event 返回之后才运行，因此 event 对应的数值状态已经提交。observer
抛错时：

- blocking `renormalize` 传播异常；
- `LocalRunSession` 进入 `LOCAL_FAILED`；
- `run` 仍停在最后一个 committed safe boundary；
- 不自动继续，也不静默忽略绘图失败。

当前 durable session helpers 不接受 `LOCAL_FAILED`。若需要从 failed session 自动
checkpoint，应作为 execution adapter 的独立后续改动，不混入 LoopTNR 数值实现。

## Blocking API

```julia
renormalize(
    cell::AbstractTensorCell,
    algorithm::LoopTNR;
    observer=_DEFAULT_OBSERVER,
) -> LoopTNRResult
```

实现模式应与 `split` 一致：

```text
initialize_run
→ resolve observer
→ observer running
→ repeated step! + observer step
→ observer completed
→ run_result
```

不得修改调用者输入。`initialize_run` 至少复制输入一次，之后 run 只操作自己的
TensorCell。

## Snapshot、restore 与 checkpoint

### Snapshot payload

概念形状：

```julia
struct LoopTNRSnapshot{C,P,L,H,I} <: AbstractAlgorithmSnapshot
    cell::C
    completed_rg::Int
    phase::LoopTNRPhase
    child::P
    log::L
    observations::H
    identity::I
end
```

必须保存：

- 当前 committed TensorCell；
- phase 与 RG cursor；
- active 或 complete child 的 `SplitSnapshot`；
- complete outer logs；
- complete `ObservationHistory`；
- exact immutable `LoopTNR` algorithm identity。

不得保存：

- `Task`；
- locks；
- observer/callback；
- split transfer caches；
- checkpoint path；
- 历史 TensorCell。

### Copy semantics

snapshot 必须与 live run 隔离：

- TensorCell 中每个 gate 深度 copy；
- outer log vectors copy；
- observation history 使用 `deepcopy`，因为 future fixed-point observation 可能
  含 mutable arrays；
- child 使用现有 `snapshot(child)`；
- algorithm identity 使用 `deepcopy(algorithm)`，沿用
  `SplitProblemIdentity` 的 value-semantics 做法。

### Restore API

推荐：

```julia
restore_run(
    saved::LoopTNRSnapshot,
    algorithm::LoopTNR,
) -> LoopTNRRun
```

与 `SplitSnapshot` 不同，LoopTNR snapshot 必须保存当前 evolved TensorCell，因此
恢复不需要最初输入 cell。active split target 可从 saved current cell 重新执行
`PeriodicMPS(cell)` 得到，再调用：

```julia
restore_run(saved.child, target, algorithm.split)
```

restore 必须验证：

- saved algorithm 与 supplied algorithm 值相等；
- phase invariants；
- `0 <= completed_rg <= maximal_rg`，初始化 phase 除外；
- log/history 的 RG 序列连续且都从 0 开始；
- log/history 长度等于 `completed_rg + 1`；
- child presence 和 completion status 与 phase 一致；
- active child target spaces 与 saved cell 转换得到的 target 一致；
- finished snapshot 没有 child；
- saved `completed_rg` 不超过 supplied `maximal_rg`。

### Checkpoint codec

在 snapshot/restore 通过后再注册：

```julia
_checkpoint_algorithm(::LoopTNRSnapshot) = :looptnr
```

不要提前提高 checkpoint schema version。当前项目策略只承诺同 package/dependency
环境恢复；加入一种新的 payload type 不等同于改变已有 split payload schema。

checkpoint round-trip 必须验证 observation history，没有 observation 的 checkpoint
不能被视为有效 LoopTNR checkpoint。

## 建议文件布局

保持 generic observable 与 LoopTNR-specific code 分层：

```text
src/
  observable/
    interface.jl          # AbstractObservable, AbstractObservation, observe
    history.jl            # ObservationHistory
  looptnr/
    interface.jl          # LoopTNR config and constructors
    normalization.jl      # InfinityNormNormalization
    observable.jl         # TNRObservable and TNRObservation
    coarsegrain.jl        # function declaration + maintainer implementation
    logs.jl               # RGLog and LoopTNRLog
    run.jl                # LoopTNRRun, phases, step!, result
    identity.jl           # snapshot identity validation
    progress.jl           # default observer, added late
```

测试：

```text
test/looptnr/
  runtests.jl
  normalization_test.jl
  observable_test.jl
  interface_test.jl
  coarsegrain_test.jl
  run_test.jl
  snapshot_test.jl
  progress_test.jl
  checkpoint_test.jl
```

不要一开始创建所有空文件。每个阶段只创建当时已有行为的最少文件。

## 分阶段实施顺序

### Phase 0：确认基线

**目标：** 确保开始实现时工作区状态和现有测试已知。

步骤：

1. 阅读本文和 `design/resumable-execution.md`。
2. 检查 `git status`，不要覆盖维护者尚未提交的 TensorCell show/layout 修改。
3. 运行 focused TensorCell、split、composition 与 checkpoint tests。
4. 记录与 LoopTNR 无关的已有失败，不顺手修复。

验收：没有源码变化。

### Phase 1：通用 observable 基础类型

**目标：** 建立不依赖 LoopTNR 的最小接口。

实现：

- `AbstractObservable`；
- `AbstractObservation`；
- generic `observe` declaration；
- 最小 `ObservationHistory` container；
- history append/access/copy 或 deepcopy 行为。

测试：

- history 保持顺序；
- deep copy 后 mutable payload 不 alias；
- 不引入 callback 或 tensor ownership。

本阶段不实现 normalization、LoopTNR config 或 conformal observable。

建议独立 commit。

### Phase 2：TensorCell infinity normalization

**目标：** 完成可独立验证的 normalization primitive。

实现：

- `AbstractNormalization`；
- `InfinityNormNormalization`；
- `NormalizationObservation`；
- 三种 TensorCell 的独立 gate enumeration；
- 事务式 norm validation 和 in-place scaling。

测试矩阵：

- Uniform、Checkerboard、Plaquette；
- nonuniform gate norms；
- aliased concrete fields；
- zero tensor；
- non-finite tensor；
- extreme scales；
- object identity preserved；
- returned `log_scale` 正确。

本阶段不实现 outer run。

### Phase 3：TNR observable、config、log 与 RG 0

**目标：** 在没有 coarse-grain contraction 的情况下交付一个完整可运行的
`maximal_rg=0` LoopTNR。

实现：

- `TNRObservable` / `TNRObservation`；
- `LoopTNR` core constructor 和 default `maxrank` constructor；
- `RGLog` / `LoopTNRLog`；
- `LoopTNRResult`；
- phases、`LoopTNRRun`；
- RG 0 `step!` transaction；
- `run_result`；
- `renormalize` blocking API；
- RG 0 event。

测试：

- all three input TensorCells；
- input object/tensors unchanged；
- `maximal_rg=0` 在一次 `step!` 后完成；
- history/log 都只含 RG 0；
- default constructor 精确包含两级 truncation 和 circular 40 sweeps；
- custom non-gauge `PeriodicSplit` 被允许；
- wrong sweep domain 和负 `maximal_rg` 被拒绝；
- fixed-point failure 不提交 normalized candidate；
- observer 收到 RG 0 event。

完成此阶段后，项目已经有 observation 与 outer result 语义，但尚未有 multi-RG。

### Phase 4：维护者实现 coarse graining

**负责人：项目维护者。**

实现：

- 八站点 optimized PMPS → `CheckerboardGates` contraction；
- 完整 transpose/dual convention；
- black/white orientation。

验收至少包含：

- length/rank validation；
- `assertconsistency`；
- scalar/space type preservation；
- 一个人工小张量或与维护者认可 reference implementation 的数值对照；
- 清晰记录 U-shape site ordering。

其他 session 不应代写这个 contraction。

### Phase 5：嵌套 SplitRun 与 multi-RG

**目标：** 将真实 `SplitRun` 作为 child 接入 outer state machine。

实现：

- PREPARE/OPTIMIZE/COARSE_GRAIN transitions；
- 每个 outer `step!` 最多执行一个 child site；
- nested site event；
- final child site 与 RG commit 分成两个 safe steps；
- coarse-grain transaction；
- append log/history；
- `isfinished` 和 multi-RG `run_result`。

测试：

- event 顺序为 RG0、8-site updates、RG1，考虑多 sweep；
- 中间 site pause/resume；
- final site 后 pause，再 coarse grain；
- RG event 后 pause，再开始下一 child；
- `maximal_rg=1` 与 `2`；
- custom zero-length split sweep；
- coarse-grain/normalization/fixed-point injected failure 保持 commit invariants；
- final result history 长度为 `maximal_rg + 1`。

### Phase 6：Snapshot 与 restore

**目标：** 每个 phase 都可 durable restore。

至少测试以下边界：

1. RG 0 前；
2. RG 0 后、child 创建前；
3. child 中间 site；
4. child 最后 site 完成、coarse grain 前；
5. RG commit 后；
6. finished run。

每个边界都比较 uninterrupted 与 restored：

- final TensorCell；
- complete RG logs；
- complete observations；
- child split costs；
- event continuation，不重复已经完成的 event。

另外测试 snapshot 与 live run 隔离，尤其是 mutable observation arrays。

### Phase 7：Checkpoint codec

**目标：** JLD2 round-trip 和本地 durable pause。

实现：

- `:looptnr` checkpoint registration；
- read/write round-trip；
- restored root run；
- `pause_and_checkpoint!` 在 child site 和 RG boundary 的集成测试。

验收：

- observation history 完整；
- active child restore 后不重复 site；
- header/payload algorithm mismatch 被拒绝；
- write failure 不损坏 in-memory run。

### Phase 8：默认 progress 与 README

**目标：** 最后再加入用户界面，不让显示逻辑干扰数值状态机。

实现：

- `LoopTNRProgressObserver`；
- nested site summary；
- 每个 RG 一条永久 summary line；
- RG event 中 observation 的前端示例；
- README 从 blocking 到 controllable run 的最小示例。

默认 observer 不应直接依赖具体 conformal result fields；只显示通用 RG、cost、time
和 normalization scale。custom frontend observer 从 event 读取 future fixed-point
payload 绘图。

## 测试总矩阵

最终实现至少覆盖：

| 维度 | Cases |
| --- | --- |
| Input cell | Uniform / Checkerboard / Plaquette |
| Scalar | Float64 / ComplexF64 |
| Space | ComplexSpace / 至少一个 graded space |
| RG count | 0 / 1 / 2 |
| Split ansatz | default PeriodicGauge / one custom ansatz |
| Split completion | normal convergence / hard cap / zero-step custom sweep |
| Pause point | middle site / final site / RG boundary |
| Restore point | every outer phase |
| Failure | child update / coarse grain / normalization / fixed point / observer |
| Observation | normalization-only / test fixed-point observable with mutable payload |

完整 suite 前先运行最小 focused tests。不要为了 LoopTNR 修复无关测试或 formatter
问题。

## 非目标与延后事项

第一版明确不做：

- conformal transfer matrix；
- conformal spectrum、central charge 或 modular parameter 的数值实现；
- 单独 entanglement filtering；
- target PMPS 的预先 `gaugefix!`；
- outer RG early stopping；
- history TensorCell archive；
- job ID、scheduler、remote storage 或 frontend transport；
- acceleration research plan 中的 Anderson、pairwise perturbation 或 parallel ALS；
- 修改现有 `SiteLog.observable` 语义。

future outer stopping 应在完整 RG observation commit 后检查
`ObservationHistory`，同时保留 `maximal_rg` hard cap。不要让它在 child sweep 中间停止
outer RG。

## 后续 session 操作规范

接手本计划的 session 应遵循：

1. 先报告准备实施的 phase，不默认实施全部计划。
2. 每次只修改该 phase 必需的文件。
3. 在开始前检查 scoped `AGENTS.md` 和现有未提交工作区。
4. 不覆盖维护者的 TensorCell show/layout 或 coarse-grain convention 修改。
5. 不猜测 `coarse_grain` contraction。
6. 每个 phase 先跑 focused tests，再决定是否跑 full suite。
7. 不自行 commit，除非用户明确要求。
8. 完成后在本文对应 phase 下更新状态或新增简短 implementation note，避免后续
   session 重复工作。

## 最终完成标准

只有以下条件全部满足，才能声称第一版 LoopTNR 完成：

- `renormalize` 能从任意三种 TensorCell 完成至少两次 RG；
- default gauge ansatz 使用已确定的两级 truncation；
- coarse graining 由维护者确认 convention；
- 每个 site 和每个 RG 都是可暂停安全边界；
- RG 0 到最终 RG 的 observation 全部进入 durable history；
- log 不保存 outer observation 或 tensor history；
- snapshot/restore 覆盖每个 phase；
- JLD2 checkpoint round-trip 保留 active child 和 observation history；
- observer 能在每轮 RG 后即时收到绘图数据；
- observer failure 不造成半提交 RG；
- focused 与 full regression tests 通过，或只剩明确记录的无关环境问题。
