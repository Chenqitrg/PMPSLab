# PeriodicMPS 结构与特例审查报告

## 审查状态

- 审查日期：2026-08-09
- 审查基线：`main` 分支，`8f0110b`，并包含审查时工作区内尚未提交的修改
- 审查范围：`src/`、`test/`、`README.md`、`SPEC.md` 与相关 `design/` 文档
- 审查方式：静态阅读、完整测试套件、针对不变量的最小反例
- 本报告只记录审查结论，不包含源码修复

### 后续处理状态（2026-08-16）

本报告的反例和分析保留为历史依据；其中的“当前代码”均指审查基线，而不是现有
工作区。后续处理结果如下：

| 编号 | 当前状态 | 后续实现 |
|---|---|---|
| R1 | 已修复 | 初始化和恢复都会用 `sweep_domain` 验证实际 approximation 长度。 |
| R2 | 已修复 | `PeriodicMPS` 的构造、`setindex!` 和 `copyto!` 都在写入前验证环闭合，失败写入保持原子性。 |
| R3 | 已修复 | `PeriodicCache` 记录左右有效边界；乱序、重复或跳跃更新不能再静默返回陈旧 environment，而会抛出 `StaleCacheError` 或要求显式 rebuild。 |
| R4 | 已修复 | 空乘积由 `FormalIdentity` 表示，不再构造带伪截断元数据的满秩 identity transfer。 |
| R5 | 未修复 | Krylov 未收敛仍只记录 warning 并返回最后 iterate；仍需显式 solver outcome 和原子提交策略。 |
| R6 | 已修复 | `SPEC.md`、README 和实现统一使用 execution observer；旧 `callback` 关键字已明确移除。 |

此外，`SplitEnvironmentPolicy` 已把 dense/truncated transfer policy 接入完整
`ALSSplit` 运行路径，LoopTNR v1 的嵌套执行、snapshot、checkpoint 与 observer
也已经完成。下文“建议实施顺序”是审查当时的建议，不再是当前路线图。

本报告采用以下判断标准：

1. 尽量让非法状态无法构造，而不是依靠后续检查发现。
2. 一个事实只保留一个权威来源，避免多个字段或对象必须人工同步。
3. 避免依赖调用顺序、方向或隐藏前置条件才能保持正确的接口。
4. 区分必要的边界处理与由错误数据结构产生的特例。
5. 失败应进入类型和控制流，不能只通过日志或 warning 表示。

## 总体结论

项目的张量表示、类型分派和数值回归测试基础较好。对不同物理腿数量、非均匀虚空间、实数/复数和分级空间的支持，大多通过 Julia 分派和泛型实现，而不是在主流程中堆叠运行时类型判断。这部分符合“普通路径覆盖更多情况”的目标。

当前主要结构风险集中在执行层和缓存层：若干必须保持一致的量被保存为互相独立的可变状态；部分公开操作只有在调用者遵守隐含协议时才正确；某些失败信号没有进入算法结果。现有测试全部通过仍不能阻止这些状态出现，因为测试主要覆盖合法调用序列。

按优先级，本报告建议首先解决 sweep 与问题尺寸脱钩、`PeriodicMPS` 可被写成不一致状态、cache 依赖隐藏更新顺序这三项。

## 发现摘要

| 编号 | 严重度 | 结论 | 性质 |
|---|---|---|---|
| R1 | 高 | sweep 长度与实际 MPS 长度可以不一致，算法可能只更新部分 site 仍报告完成 | 已复现的正确性问题 |
| R2 | 中高 | `PeriodicMPS` 只在构造时检查环闭合，公开写入可立即破坏不变量 | 已复现的非法状态 |
| R3 | 中高 | cache 正确性依赖单调、连续的隐藏更新顺序，合法参数也能产生陈旧结果 | 已复现的正确性问题 |
| R4 | 中 | truncated cache 的单位元绕过截断预算，数据与策略元数据语义不一致 | 结构与规模风险 |
| R5 | 中 | Krylov 不收敛只产生 warning，调用链仍把该步作为成功更新提交 | 失败状态丢失 |
| R6 | 低 | `SPEC.md` 描述的 callback API 与实现和测试相反 | 规范漂移 |

## 详细发现

### R1：sweep 尺寸与问题尺寸是两个独立事实

相关代码：

- `src/split/interface.jl`：`ALSSplit` 独立持有一个已经包含 `N` 的 sweep。
- `src/split/run.jl`：`initialize_run` 从 ansatz 构造 approximation 后，直接迭代 `algorithm.sweep`，没有验证 sweep 覆盖 `length(approximation)`。
- `src/split/run.jl`：`isfinished` 只判断 sweep iterator 是否耗尽。

最小反例使用包含 2 个双物理腿 tensor 的目标，因此 approximation 有 4 个 B site；算法却传入 `CircularSweep(1; maximal_sweep=1)`。实际结果为：

```text
target_B_sites=4 updated_steps=1 result_sites=4
reported_run_finished=true
```

这不是单纯的参数校验遗漏，而是数据模型中同时保存了两个“site 数量”：一个来自问题本身，一个来自 sweep 配置。二者可以分离，状态机仍把 sweep 耗尽解释为算法完成。

建议方向：

- 让 sweep 配置只表达顺序与轮数，例如 `CircularOrder(maximal_sweep=...)`；
- 在 `initialize_run` 中由 `length(context.mpsB)` 生成具体 iterator；
- 如果需要支持任意自定义 sweep，则定义显式的 sweep-domain 接口，并在开始运行前验证 domain，而不是运行到一半才遇到 `BoundsError`；
- 完成条件同时表达“iterator 耗尽”和“计划覆盖了合法 domain”。

### R2：`PeriodicMPS` 的环闭合不变量只在入口成立

相关代码：

- `src/mps.jl`：外层构造器调用 `assertconsistency`。
- `src/mps.jl`：`setindex!` 直接委托给内部 `PeriodicVector`。
- `src/mps.jl`：`copyto!` 同样不会重新验证环闭合。

已验证可以向 bond dimension 为 2 的三站点 MPS 写入一个 bond dimension 为 3 的 tensor。赋值操作成功，只有之后执行 `assertconsistency` 或 `dense` 时才在远离写入点的位置抛出 `SpaceMismatch`：

```text
assignment_succeeded=true
consistency_error=SpaceMismatch
downstream_error=SpaceMismatch
```

该结构使 `PeriodicMPS` 的类型名和构造器承诺弱于对象的实际生命周期：它不是“始终闭合的 periodic MPS”，而是“构造瞬间曾经闭合、之后可能任意失效的 tensor vector”。

建议方向：

- 默认 `setindex!` 在写入前检查左右相邻 bond；
- 算法内部若已由类型或推导保证空间不变，可使用明确命名的私有快速路径；
- `copyto!` 在长度或空间布局可能变化时先整体验证，避免部分写入后失败；
- 增加“失败写入不改变原对象”的测试。

### R3：`PeriodicCache` 依赖未编码的更新序列

相关代码：

- `src/circularupdate/cache.jl`：cache 刻意不保存活动 site 或源 product。
- `src/circularupdate/cache.jl`：`advance!` 只验证 site 范围和方向，不验证本次更新是否接续上一次更新。
- `src/circularupdate/cache.jl`：prefix/suffix 直接基于相邻缓存槽原地更新。

以四站点 transfer product 为例，依次更新 site 3、site 2，并对两次调用都传入 `dir = +1`。两个调用分别满足现有参数检查，但第二次更新使 `left[2]` 改变后，不会使已经依赖旧 `left[2]` 的 `left[3]` 失效。之后查询 site 4 environment 会静默返回陈旧值：

```text
stale_cache=true
```

当前 `SplitRun` 生成的正常 sweep 顺序符合 cache 协议，因此这不是现有主路径必现错误；风险在于协议由调用者记忆维持，而 `advance!` 和相关类型又作为公共 API 导出。未来增加新的 sweep 或单独复用 cache 时，很容易产生数值正确、类型正确但内容陈旧的结果。

建议方向：

- 让一个对象原子地拥有 factor、cache 和 sweep cursor；
- 或让 cache 记录有效前缀/后缀边界，使乱序更新触发重建或明确报错；
- 若坚持当前高性能协议，将 `advance!` 限定为内部接口，并提供安全的公开更新入口；
- 增加乱序、重复和跳跃更新测试，明确哪些序列应被拒绝。

### R4：截断表示中的精确单位元是由表示方式制造的特例

相关代码：

- `src/transfermatrix/periodic.jl`：truncated `edge_identity` 先创建 dense identity，再以无截断策略执行完整 SVD，最后重新附上原截断策略。
- `src/transfermatrix/truncated.jl`：`TruncatedTransferMatrix` 的文档将 `trunc` 描述为其因子化与后续乘法使用的截断策略。

在虚空间维数为 3、策略为 `truncrank(1)` 时，单位元的中间空间仍保留 9 维：

```text
requested_rank=1
identity_retained_dimension=9
```

保持单位元精确在数学上是合理的；问题在于普通 `TruncatedTransferMatrix` 被同时用于表示近似 transfer 和代数单位元。于是代码需要构造一个“因子没有按 `trunc` 截断、但携带该 `trunc` 元数据”的特殊对象。对于大 bond，这也可能破坏用户选择 truncated 模式时预期的内存上界。

建议方向：

- 用独立的 lazy identity 类型表示空乘积；
- 为 transfer product 定义统一的单位元乘法，使 identity 不需要执行 SVD；
- 或让 cache 的边界槽表达为空，而不是强迫空乘积具有与普通 factor 完全相同的具体类型。

这里不建议简单地截断单位元，因为那会改变空环境的数学含义。应修正表示，而不是用另一个数值特例覆盖现有特例。

### R5：Krylov 收敛状态没有进入结果类型

相关代码：

- `src/solve.jl`：Krylov `ldiv`、矩阵形式 `rdiv` 和 action 形式 `rdiv` 都丢弃 `info`，仅在 `info.converged == 0` 时记录 warning。
- `src/split/context.jl`：`solvenew` 无法区分收敛解和未收敛的最后 iterate。
- `src/split/run.jl`：只要 `update!` 返回，该 site 就会写入日志并推进 cursor。
- `src/logs.jl`：`SweepLog.converged` 没有承载每次局部求解的状态。

warning 是观测信息，不是控制流。批处理、后台 session 或关闭日志的环境仍会把未收敛解提交到 MPS 和 cache，最后得到一个形式上正常完成的 run。

建议方向：

- 让 Krylov wrapper 返回包含 `value`、`converged`、`residual` 和迭代信息的结果；
- 由算法策略决定未收敛时是失败、重试、降级还是接受近似值；
- 只有决定接受后才原子地写入 tensor、cache、log 和 cursor；
- 将局部求解状态记录进 `SiteLog` 或专门的 solve event，而不是只有全局布尔值。

### R6：规范与实现对 callback 的承诺相反

`SPEC.md` 描述了：

```julia
split(mps, alg; callback = nothing)
```

当前 `split` 实现没有该关键字参数，测试还明确要求该调用抛出 `MethodError`。README 当前采用 observer/session 模型，与实现更接近。

建议选择一个权威契约：若 callback 已被 execution event 取代，应删除或标记 SPEC 中的旧设计；若它仍是需求，则测试不应把缺失 API 固化为正确行为。

## 澄清：`TruncatedTransferMatrix.mul!` 只检查 S 空间不是问题

本项在初始审查中曾被错误地列为高优先级问题，现已撤销。

在 `PeriodicCache` 的定义域内，一个缓存槽对应固定区间：

- `u` 的外空间由区间左端固定；
- `vᴴ` 的外空间由区间右端固定；
- site 更新不改变 virtual space；
- 同一 product 的 factor 使用同一个 truncation policy，且 `combine_truncation(t, t) == t`。

因此原地乘法中只有 SVD 产生的中间空间可能变化，`_can_reuse(destination, s)` 只检查 `s` 是合理的热路径设计。此前混合不同截断策略构造的反例违反了这一 cache 前置条件，不能证明正常缓存路径有错误。

如果 `mul!` 被定位为通用公共 API，可以补充或收紧其前置条件；如果它是 cache 内部操作，则无需为已经由 cache 不变量固定的两侧空间重复付出检查成本。

## 做得好的部分

以下设计符合本报告采用的标准，应当保留：

1. `HalfInt` 和 `HalfPeriodicVector` 把 site 与 bond 索引的语义差异放进类型，而不是散布 `i ± 1` 约定。
2. `PeriodicMPS` 的外层构造器拒绝空 period，并集中检查初始环闭合。
3. dense 和 truncated transfer 通过分派实现；新增 split environment consumer 也通过 transfer 类型分派选择 dense 或 factored contraction，没有在主循环中使用 `isa` 分支。
4. canonicalization helper 对物理腿数量参数化，避免为一腿、两腿复制算法。
5. snapshot 排除了可重建 cache，并在 restore 时验证 cursor 与日志，减少持久化状态数量。
6. 单站点、周期边界、zigzag 转向、分级空间及 checkpoint 都有明确回归测试。
7. `PeriodicCache` 的 `N == 1` 分支虽然是边界特例，但集中、短小且测试充分；它不是当前主要结构风险。应优先消除跨对象不变量，而不是机械删除所有条件分支。

## 建议实施顺序

### 第一阶段：阻止静默错误

1. 让 sweep domain 从 `SplitContext` 的 MPS 长度派生，或在初始化时强制验证。
2. 使 `PeriodicMPS` 的默认写入保持左右 bond 一致，并保证失败写入不产生部分修改。
3. 明确 cache 更新序列的所有权：安全封装或把不安全接口内部化。

### 第二阶段：使失败成为状态

1. 为局部求解定义带收敛信息的结果类型。
2. 规定 site update 的提交边界，避免 tensor 已写入而 cache/log/cursor 尚未提交。
3. 为后台 session 提供可查询、可传播的失败原因。

### 第三阶段：消除表示诱发的特例

1. 为 transfer product 引入不需要 SVD 的 identity 表示。
2. 统一规范文档、README 与测试中的公共 API。
3. 根据实际公开边界，收紧 raw constructors、`setindex!`、`advance!` 和三参数 `mul!` 的可见性与契约。

## 验证说明

审查完成后，包含上述未提交 environment 修改的当前工作区通过了完整
`Pkg.test()`。另执行了四个针对不变量的最小实验：

1. sweep 只覆盖一个 site 时，四站点 approximation 仍被报告为完成；
2. 写入不兼容 tensor 后，`PeriodicMPS` 直到后续 contraction 才失败；
3. cache 以非单调顺序更新后，environment 与完整重算不一致；
4. `truncrank(1)` 的 truncated identity 实际保留 9 维中间空间。

这些结果说明测试套件对合法数值路径覆盖较好，但还需要更多“不变量破坏测试”。推荐把反例转成回归测试时，先确定期望架构：有些反例应在构造时拒绝，有些应触发安全重建，而不是只断言当前的晚期异常。
