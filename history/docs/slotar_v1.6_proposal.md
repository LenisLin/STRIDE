# Proposal V1.6（全文整合稿）

## Selection-aware longitudinal change atlas for multi-ROI IMC
### Community prototypes + (Density/Shape/Scale) + Unbalanced OT + Uncertainty + Remapping + Cohort-level inference

---

## 版本说明（V1.6 相对 V1.5 的关键更新）

**(1) Density/Scale 聚合改为 area-weighted（总和比值）**：使 `S_{p,t,g}` 具备严格物理语义（单位面积总细胞密度）与可审计性。  
**(2) Creation/Destruction 指标从 L1 改为正部 $(\cdot)_+$**：避免 UOT（KL 松弛）下边际超配污染语义，并输出归一化比率用于跨患者比较。  
**(3) Retention 阈值 $\tau$ 改为“组内基线校准”默认策略**：避免固定常数任意性与 cost 分位数的几何依赖；提供低成本敏感性版本。  
**(4) Active set 语义拆分**：将“原型裁剪（统计/生物语义）”与“数值 floor（仅避免 log(0)）”分离，并输出 `mass_pruned_ratio` 审计字段与触发规则。  
**(5) Batch drift 风险标记修复维度错位 + 同尺度**：仅在 marker 子空间计算 cosine，并对 drift 向量施加与原型一致的 robust scaling。  
**(6) 单 ROI block bootstrap 保留“冻结特征”，但明确 estimand；提供可选 moving-block 作为备选**。  
**(7) 补齐队列级推断模块**：两部模型（组存在性 + 条件连续指标）+ 混合效应模型 + 多重检验策略。

---

## 0) 摘要（1 段落）

在 multi-ROI、pre/post 选择机制不对称且不可控的 IMC 纵向数据中，我们用“局部社区实例 → 全局原型字典（prototype dictionary）”建立统一语义坐标系；在患者-时间点-分层组（区室或 ROI-state）上，将原型密度聚合成经验测度，并通过 **Unbalanced Optimal Transport (UOT)** 将 pre→post 变化分解为 **retention / remodeling / creation / destruction**。我们同时输出 **scale（单位面积总密度）变化**与 **shape（归一化结构形状）变化**以解耦“总量/覆盖”与“结构形状”。不确定性采用 ROI bootstrap；当某组仅 1 个 ROI 时，采用 **自适应网格 block bootstrap（带有效面积与 composition-stratified 约束）**，并冻结 kNN/社区特征以避免拓扑撕裂导致的伪边缘效应。所有关键事件可回映射到 ROI 原图以便病理复核。UOT 目标函数在广义 KL + 熵正则下有显式下界，数值求解默认启用 **log-domain + clamp + ε-scaling** 退火以保证在稀疏高维分布上可收敛。最后在队列层面，基于患者级指标输出构建两部模型与混合效应推断，显式处理结构性零与批次效应。

---

## 1) 问题定义（严格边界）

### 1.1 数据现实约束

- 层级结构：cell → ROI → timepoint → patient
- pre/post 组织状态集合不对称（如 post 出现 RTB，pre 不存在；pCR post 可能没有 CT）
- post ROI 来自更大母体组织抽样，选择机制不对称且不可控
- ROI 面积近似一致（≈1 mm²），但仍应优先使用 tissue mask 的有效面积；可计算细胞密度（cells/mm²）

### 1.2 我们要解决的“蓝海输出”

对每位患者、每个分层组 \(g\)（区室或 ROI-state），输出：

1. **可归因变化分解**：retention / remodeling / creation / destruction（density-level 与 shape-level 各一套）
2. **可置信**：bootstrap 区间、事件复现概率、敏感性版本、批次漂移风险标记、单 ROI 降级声明
3. **可回映射**：事件 → 原型 → ROI → 空间区域 → 细胞集合（供病理复核）

> 关键解释边界（必须写入工具文档与手稿）  
> creation/destruction 表示在给定表示、代价、超参、分层条件与 ROI 覆盖下的 **unmatched mass**，不等价于真实生物学生成/消亡。我们通过 shape/scale 解耦、组内参数校准、UQ 与 drift 风险标记提供证据等级。

---

## 2) 输入与输出（明确到字段级）

### 2.1 输入

对患者 \(p\)，时间 \(t\in\{0,1\}\)，ROI \(r\)，细胞 \(c\)：

- 坐标 \(\mathbf{s}_{p,t,r,c}=(x,y)\)
- marker 表达 \(\mathbf{m}_{p,t,r,c}\in\mathbb{R}^M\)（\(M\approx 30\)）
- ROI 有效面积 \(Area_{p,t,r}\)（优先 tissue mask；否则 nominal）
- 可选：ROI 标签 \(\ell_{p,t,r}\in\mathcal{L}\)（CT/PT/RTB/AB 或自定义）
- 可选：批次信息 batch_id；临床协变量 \(\mathbf{v}_p\)；结局 \(y_p\in\{\mathrm{pCR},\mathrm{NR}\}\)

### 2.2 输出（核心对象）

对每个 \((p,g)\)：

- **Scale**：\(\mathrm{scale\_ratio}_{p,g}=S_{p,1,g}/S_{p,0,g}\)，并输出审计字段 \(A_{p,t,g}\)、ROI 数、细胞数
- **Density-level UOT**：\((R^{dens},M^{dens},B^{dens},D^{dens})\) 的指标（详见第 7 节）+ 事件表 + UQ
- **Shape-level UOT**：\((R^{shape},M^{shape},B^{shape},D^{shape})\) 指标 + 事件表 + UQ
- **Batch drift 风险标记**：每条 remodeling edge 的 drift-alignment 分数与标记；并输出包含/排除 drift-aligned edges 的敏感性版本
- **Remapping package**：事件→ROI→细胞坐标/邻域集合→marker/组成摘要→可信度标签
- **审计字段**：  
  `lambda_dens[g]`, `lambda_shape[g]`, `tau_g[g]`, `s_C`, `eta_floor`,  
  `proto_prune_rule`, `n_min_proto`, `mass_pruned_ratio`,  
  `UQ_mode`, `area_mode`, `G_used`, `n_blocks_valid`, `eps_schedule_id`

---

## 3) 表示层：社区实例与原型字典（统一语义坐标系）

### 3.1 community instance \(u_c\)（在完整 ROI 上计算一次）

默认用 kNN（减弱边缘效应），对每个细胞 \(c\)：

\[
\mathbf{p}_c=\frac{1}{k}\sum_{c'\in \mathcal{N}_k(c)}\mathbf{e}_{c'},\quad
\bar{\mathbf{m}}_c=\frac{1}{k}\sum_{c'\in \mathcal{N}_k(c)}\mathbf{m}_{c'},\quad
\delta_c=\frac{k}{\pi r_k(c)^2}
\]

其中 \(\mathbf{e}_{c}\) 为 coarse identity（例如免疫/肿瘤/基质或粗细胞类 one-hot/soft）。定义：

\[
u_c=[\mathbf{p}_c,\bar{\mathbf{m}}_c,\mathbf{m}_c,\delta_c]
\]

对四个块做 block-wise robust scaling 得 \(\tilde u_c\)。V1.6 强制保存 scaling 参数（median/MAD 或等价 robust scaler），用于 drift 子空间对齐（第 9 节）。

### 3.2 全局原型字典（分层均匀下采样 + k-means）

从每个 \((p,t)\) 抽同数量 \(N_{bal}\) 的 \(\tilde u_c\) 构成 \(\mathcal{U}_{bal}\)，在其上 k-means 学得原型中心 \(\mathbf{z}_1,\dots,\mathbf{z}_K\)。对全量细胞硬分配：

\[
a(c)=\arg\min_{k}\|\tilde u_c-\mathbf{z}_k\|_2
\]

并缓存每个 ROI 的 prototype counts（后续不再重复计算）。

---

## 4) 分层组 \(g\)：区室优先，缺失退化为 ROI-state

- 若有 \(\ell_{p,t,r}\)：\(g=\ell\)
- 否则对 ROI 的原型密度/比例向量聚类得到 ROI-state：\(g=\hat\ell\)
- 或用户指定不分层：\(g=\mathrm{all}\)

> 组内校准（\(\lambda,\tau\)）与组内 UOT 都按 \(g\) 执行，避免空间异质性污染参数语义。

---

## 5) 三件套聚合：Density / Shape / Scale（V1.6：area-weighted 强制）

### 5.1 ROI 原型计数与密度

\[
n_{p,t,r,k}=\#\{c\in ROI_{p,t,r}:a(c)=k\},\quad
\rho_{p,t,r,k}=\frac{n_{p,t,r,k}}{Area_{p,t,r}}
\]

### 5.2 组级聚合（总和比值，单位 cells/mm²）

对每个 \((p,t,g)\)：

\[
N_{p,t,g,k}=\sum_{r:\ g(r)=g} n_{p,t,r,k},\qquad
A_{p,t,g}=\sum_{r:\ g(r)=g} Area_{p,t,r}
\]

\[
a_{p,t,g,k}=\frac{N_{p,t,g,k}}{A_{p,t,g}}\quad (A_{p,t,g}>0)
\]

### 5.3 Scale 与 Shape

\[
S_{p,t,g}=\sum_k a_{p,t,g,k}
=\frac{\sum_k N_{p,t,g,k}}{A_{p,t,g}}
\]

\[
\bar{\mathbf{a}}_{p,t,g}=\frac{\mathbf{a}_{p,t,g}}{S_{p,t,g}}\quad (S_{p,t,g}>0)
\]

### 5.4 审计字段（强制输出）

对每个 \((p,t,g)\) 输出：

- `A_{p,t,g}`（总有效面积）、`n_roi_{p,t,g}`、`n_cells_{p,t,g}`
- `area_mode ∈ {mask, nominal}`

---

## 6) UOT：目标函数、cost scaling、(\lambda) 校准、active set、数值求解

### 6.1 全局静态 cost scaling（强制）

原型学习后一次性计算：

\[
s_C=\mathrm{median}\{\|\mathbf{z}_i-\mathbf{z}_j\|_2^2:\ i\neq j\}
\]

任意 UOT 子问题都用：

\[
C_{ij}=\frac{\|\mathbf{z}_i-\mathbf{z}_j\|_2^2}{s_C}\ge 0
\]

这保证 \(\lambda\) 的量纲语义在校准与推断之间不漂移。

### 6.2 Active set：语义裁剪 vs 数值 floor（V1.6 必须分离）

**(A) 原型支持集裁剪（统计/生物语义）**

对每个 \((p,g)\)（在 dens 或 shape 输入之前，先基于计数定义 active）：

\[
N_{p,g,k}^{(0+1)} = N_{p,0,g,k}+N_{p,1,g,k}
\]

\[
\mathcal{K}_{p,g}=\{k:\ N_{p,g,k}^{(0+1)}\ge n_{\min}^{proto}\}
\]

其中 \(n_{\min}^{proto}\) 为最小总计数阈值（默认建议 2 或 5；必须写入审计字段）。

并输出：

\[
\texttt{mass\_pruned\_ratio}=
\frac{\sum_{k\notin\mathcal{K}}(a_k+b_k)}{\sum_k(a_k+b_k)+\epsilon}
\]

建议触发规则：若 `mass_pruned_ratio > 0.5%`，自动生成敏感性重跑（例如降低 \(n_{\min}^{proto}\)）或输出降级声明。

**(B) 数值 floor（纯数值语义）**

为避免 \(\log(0)\)，仅对输入向量做：

\[
a\leftarrow \max(a,\eta_{floor}),\quad b\leftarrow \max(b,\eta_{floor})
\]

`eta_floor` 只用于数值稳定，不用于裁剪原型。

### 6.3 UOT 目标函数（density-level 或 shape-level 仅输入向量不同）

对 \(a,b\in\mathbb{R}_+^{|\mathcal{K}|}\)，求：

\[
\Pi^*=\arg\min_{\Pi\ge 0}\ 
\langle C,\Pi\rangle
+\lambda\,\mathrm{KL}(\Pi\mathbf{1}\mid a)
+\lambda\,\mathrm{KL}(\Pi^\top\mathbf{1}\mid b)
+\varepsilon \sum_{ij}\Pi_{ij}(\log\Pi_{ij}-1)
\]

广义 KL：

\[
\mathrm{KL}(x\mid a)=\sum_i\left[x_i\log\frac{x_i}{a_i}-x_i+a_i\right]\ge 0
\]

### 6.4 有界性（下界）

熵项中 \(h(x)=x(\log x-1)\) 在 \(x\ge 0\) 的全局最小值为 \(-1\)（在 \(x=1\)），故当 \(|\mathcal{K}|=K'\)：

\[
\sum_{ij}h(\Pi_{ij})\ge -K'^2\quad\Rightarrow\quad
\mathcal{L}(\Pi)\ge -\varepsilon K'^2
\]

因此目标函数有界且在常见条件下解存在。

### 6.5 \(\lambda\) 校准（组内双校准，且与 ε-schedule 强耦合）

在同组 \(g\) 内，用 pre 的 ROI–ROI 配对构造 unmatched ratio 曲线，在候选网格 \(\Lambda\) 上选择使队列中位数匹配容忍度 \(\alpha\) 的 \(\lambda\)。

- 得到 \(\lambda^{dens}_g\)：输入为 density 向量（ROI 或组级）
- 得到 \(\lambda^{shape}_g\)：输入为归一化 shape 向量（同流程、不同输入）

**强制要求**：校准与推断使用完全一致的 ε-scaling 退火序列与收敛准则（记录 `eps_schedule_id`），并保存每组 `median unmatched ratio vs λ` 曲线图作为可审计材料。

### 6.6 数值求解（强制启用 log-domain + clamp + ε-scaling）

**强制**：

- log-domain stabilized Sinkhorn
- clamp（例如 [-50,50]）
- warm-start 对偶变量（跨 ε 层）

**默认启用 ε-scaling**（退火）：

- \(\varepsilon_{init}>\cdots>\varepsilon_{target}\)，如 \(10.0\to 0.1\)，每层乘 0.5
- 每层迭代到 marginal residual 达标或 objective 稳定，再降 ε
- 推荐调用成熟库（POT 或等价稳定实现）；若自写必须严格遵循上述策略

---

## 7) 变化分解指标与事件（V1.6：B/D 用正部，Retention 用校准阈值）

对每个 \((p,g)\)（density-level 与 shape-level 分别计算）：

- transported mass：
\[
T=\sum_{ij}\Pi_{ij}
\]
- marginals：
\[
m^{pre}=\Pi\mathbf{1},\quad m^{post}=\Pi^\top\mathbf{1}
\]
- destruction / creation（正部）：
\[
D^+=\sum_i (a_i-m^{pre}_i)_+,\quad
B^+=\sum_j (b_j-m^{post}_j)_+
\]
- 归一化比率（用于跨患者/队列比较）：
\[
d=\frac{D^+}{\sum_i a_i+\epsilon},\quad
b=\frac{B^+}{\sum_j b_j+\epsilon}
\]
- remodeling 强度：
\[
M=\frac{\sum_{ij} C_{ij}\Pi_{ij}}{T+\epsilon}
\]

### Retention（默认使用组内基线校准阈值 \(\tau_g\)）

\[
R(\tau)=\frac{\sum_{C_{ij}\le \tau}\Pi_{ij}}{T+\epsilon}
\]

**\(\tau\) 策略（V1.6 默认）**：

- **组内基线校准 \(\tau_g\)**：在同组 \(g\) 的 pre ROI–ROI 配对集合上，求 UOT 并收集 \(\Pi\)-加权 cost 分布；取固定分位数 \(q_\tau\)（预先锁定，例如 0.25 或 0.5）作为 \(\tau_g\)。  
- **敏感性（推荐）**：\(\tau\in\{0.5\tau_g,\tau_g,2\tau_g\}\) 的方向一致性（补充材料）。

审计字段：`tau_mode`, `tau_g[g]`, `tau_q`.

### 事件提取（保持 V1.5 思路，但使用新指标）

- retention edges：\(\Pi_{ij}\) 大且 \(C_{ij}\le\tau\)
- remodeling edges：\(C_{ij}\Pi_{ij}\) 大
- create/destroy prototypes：由残差正部 \((a-m^{pre})_+\)、\((b-m^{post})_+\) 的大分量给出  
每个事件附：质量、成本、UQ 复现概率、敏感性标签、drift 风险标记、可回映射索引。

---

## 8) 不确定性（UQ）：ROI bootstrap + 单 ROI 自适应网格 block bootstrap（冻结特征）

### 8.1 多 ROI（默认）

若 \((p,t,g)\) 至少 2 个 ROI：ROI bootstrap（有放回）重采样 ROI，重算 \(\mathbf{a},\bar{\mathbf{a}}\)、UOT、指标与事件；输出置信区间与事件复现概率。

### 8.2 单 ROI（工程规范，冻结特征）

**(i) 特征预计算冻结（必须）**  
在完整 ROI 上一次性计算并冻结：

- kNN 邻域与 \(\tilde u_c\)
- 原型标签 \(a(c)\)
- block 内原型计数向量与有效面积

单 ROI block bootstrap **只重采样带标签的 block 集合**，不重算 kNN/社区特征/原型标签，避免因“拼图式重采样 + 重连邻域”导致的拓扑撕裂与伪边缘效应。

> Estimand 声明（必须写入 Methods）  
> 该 UQ 量化的是：在固定分割与固定局部表征映射下，有限视野 ROI 内空间相关导致的组成与 UOT 指标不确定性。它不覆盖重新分割、重新构图或术后母体组织 ROI 选择机制的不确定性。

**(ii) 自适应 \(G\times G\) 网格与确定性 assignment（必须）**

- 半开区间/`floor` 规则给每个细胞唯一 block id（网格线细胞归属可复现）
- 选择 \(G\) 使有效 blocks 数达到目标（如 16 或 25）且每块细胞数足够；不足则降低 \(G\)

**(iii) 空网格/低细胞块过滤 + pseudo-area（必须）**

- 每个 block \(b\) 预计算：细胞集合、原型计数向量 \(n_{b,k}\)、\(Area_b\)
- 丢弃 \(|Cells(b)|<n_{\min}/3\) 或 \(Area_b<Area_{\min}\) 的 block
- bootstrap 只在有效 block 集合上进行
- 组装 pseudo-ROI：
\[
n^{pseudo}_k=\sum_{b\in sample} n_{b,k},\quad 
Area_{pseudo}=\sum_{b\in sample} Area_b,\quad
\rho^{pseudo}_k=n^{pseudo}_k/Area_{pseudo}
\]
输出 `area_mode` 与 `n_blocks_valid` 用于审计。

### 8.3 可选：moving-block（连通块）模式（不默认启用）

提供连通块抽样（例如随机种子 block + BFS 膨胀到固定块数）作为备选；该模式可进一步降低拓扑断裂风险，但工程复杂度较高，默认不启用，仅作为方法学备选与未来扩展。

审计字段：`UQ_mode ∈ {roi_bootstrap, grid_block_frozen, moving_block_optional}`。

---

## 9) 批次漂移风险标记（V1.6：同子空间 + 同尺度 + 可审计）

### 9.1 漂移向量估计（建议按 batch_id 分层）

选择 anchor 细胞群（如内皮/成纤维；更优为技术对照通道/控制抗体，若存在），估计：

\[
\Delta_{batch}=\mathrm{Scale}_m\big(\bar{\mathbf{m}}_{anchor,post}-\bar{\mathbf{m}}_{anchor,pre}\big)
\]

其中 \(\mathrm{Scale}_m\) 使用与原型 marker 子空间一致的 robust scaling 参数（来自第 3 节保存的 scaler），确保尺度一致。

### 9.2 remodeling edge 的 drift-alignment

令 \(\mathbf{z}^{(m)}_k\) 为原型中心在 marker 子空间（或 \([\bar m,m]\) 子空间）的切片：

\[
S_{i\to j}=\cos(\mathbf{z}^{(m)}_j-\mathbf{z}^{(m)}_i,\Delta_{batch})
\]

若 \(S_{i\to j}>\texttt{drift\_thr}\)（默认 0.85）标记为 drift-aligned，并输出“包含/排除 drift-aligned edges”的敏感性版本。

---

## 10) 回映射（审计与病例展示）

事件 \(\to\) 原型集合 \(\to\) ROI 内细胞集合（按 \(a(c)\)）\(\to\) 细胞坐标/邻域集合 \(\to\) 可视化高亮。输出同时附：

- 原型 marker/identity 摘要
- 事件质量与 UQ 复现概率
- drift 风险标记（及敏感性版本差异）
- 审计字段（\(\lambda,\tau, s_C, n_{\min}^{proto}, mass\_pruned\_ratio, area\_mode\) 等）

---

## 11) 胃癌队列模拟推演（预期输出形态）

### 11.1 队列结构（已知事实）

- \(P\approx 50\)，pre/post 配对；标签 \(y_p\in\{\mathrm{pCR},\mathrm{NR}\}\)
- pre：2–3 ROI（多见 CT/PT）
- post：3–5 ROI（NR 多见 CT；pCR 多见 RTB；可有 AB/PT）
- ROI 面积近似 1mm²；优先 tissue mask 有效面积

### 11.2 运行后典型模式（以 V1.6 指标解释）

**pCR（典型）**

- RTB 组：density-level 的 \(b^{dens}\) 与 \(M^{dens}\) 可能上升；shape-level 若也高，支持“结构形状重塑/新生态位”，而非纯覆盖变化
- CT 组：存在性 \(I_{p,g}\) 可能从 1→0（结构性零），应在两部模型第一部体现，而不是强行在 CT 内做 shape 比较
- 事件回映射：remodeling edges 可落到退缩床区域的空间邻域，供病理复核

**NR（典型）**

- CT 组：retention \(R^{dens}\) 高（低 cost 大质量边多）；\(d^{dens}\) 低
- “抗性生态位候选”：高复现 retention edges，可回映射到 post CT 中稳定残存区域
- 若 remodeling edges 与 \(\Delta_{batch}\) 高对齐 → drift 风险标记，需用敏感性版本复核

---

## 12) 队列级统计推断模块（V1.6 新增）

### 12.1 结构性零：组存在性的两部模型

对每个患者 \(p\)、组 \(g\)，定义 post 组存在性（示例：post 细胞数 > 0）：

\[
I_{p,g}=\mathbf{1}\{n\_cells_{p,1,g}>0\}
\]

**第一部（存在性）**：logistic mixed model

\[
\mathrm{logit}\,P(I_{p,g}=1)=\gamma_0+\gamma_1\mathrm{Response}_p+\gamma_2\mathbf{v}_p+(1|\mathrm{Batch})
\]

**第二部（条件连续指标）**：仅在 \(I_{p,g}=1\) 子集上，对连续指标建模（示例）

- \(\log(S_{p,1,g}/S_{p,0,g})\)
- density-level：\(b^{dens}, d^{dens}, M^{dens}, R^{dens}\)
- shape-level：\(b^{shape}, d^{shape}, M^{shape}, R^{shape}\)（仅当 \(S_{p,t,g}>0\)）

线性混合模型（或稳健回归）：

\[
\mathrm{Metric}_{p,g}=\beta_0+\beta_1\mathrm{Response}_p+\beta_2\mathbf{v}_p+(1|\mathrm{Batch})+\epsilon
\]

> 重要：推断应按组 \(g\) 分层（或加入 \(g\) 与交互项），避免不同组织状态/区室的选择机制差异混入同一回归。

### 12.2 多重检验策略（建议）

- 在每个组 \(g\) 内，对“指标家族”（例如 dens 的 \(\{\log scale, b,d,M,R\}\)，shape 同理）做 FDR
- 或预注册少量核心指标（每层 3–5 个）以降低检验负担

---

## 13) V1.6 伪代码（实现蓝图）

下面三段：**Algorithm 1（端到端）**、**Algorithm 2（组内双 λ + τ 校准）**、**Algorithm 3（UOT 求解：eps-scaling + 正部指标 + τ_g）**。V1.6 固化点包括：area-weighted 聚合、active set 语义拆分、正部 B/D、τ_g 校准、drift 同尺度、UQ 冻结特征。

```python
# Algorithm 1: End-to-end V1.6
def run_pipeline(patients, params):
    # 1) Preprocess markers & coarse identities e(c)
    preprocess_markers(patients)                 # arcsinh/normalization/batch pipeline
    assign_coarse_identities(patients)           # gating or pooled clustering -> e(c)

    # 2) Compute community instances u_c on FULL ROI (once), robust scale -> u_tilde
    #    Save robust scaling parameters for marker subspace (for drift alignment)
    scaler = fit_blockwise_robust_scaler(patients, params)   # returns scaling params
    for roi in all_rois(patients):
        roi.knn_graph = build_knn(roi.coords, k=params.k)
        roi.u_tilde = compute_u_tilde(roi, scaler, params)   # includes delta_c
        roi.area_effective, roi.area_mode = get_effective_area(roi, params)

    # 3) Learn global prototypes with balanced sampling
    U_bal = []
    for (p, t) in all_patient_timepoints(patients):
        U_bal += uniform_sample(patients[p].timepoint[t].all_u_tilde, n=params.N_bal)
    z = kmeans(U_bal, K=params.K, seed=params.seed)          # prototype centers z_k

    # 4) Assign prototype id a(c) for all cells; cache (freeze)
    for roi in all_rois(patients):
        roi.proto_id = assign_prototype(roi.u_tilde, z)
        roi.proto_counts = count_by_prototype(roi.proto_id, K=params.K)  # vector length K

    # 5) Global static cost scale s_C (once)
    s_C = global_median_pairwise_dist2(z)        # median(||z_i-z_j||^2, i!=j)

    # 6) Define group g: compartment label if provided else ROI-state clustering
    define_groups(patients, params)              # g(r)

    # 7) Calibrate lambda for density and shape (group-wise) with fixed eps schedule
    lambda_dens, lambda_shape = calibrate_lambdas_groupwise(patients, z, s_C, params)

    # 8) Calibrate tau_g for retention (group-wise baseline calibration)
    tau_g = calibrate_tau_groupwise(patients, z, s_C, lambda_dens, params)

    # 9) Estimate drift vectors per batch (marker subspace, same scaling)
    drift_by_batch = estimate_batch_drift(patients, scaler, params.anchor_types)

    # 10) Per patient inference (dens + shape)
    outputs = {}
    for p in patients:
        outputs[p] = {}
        for g in groups_for_patient(patients[p]):
            # V1.6: area-weighted aggregation + audits
            a0, audit0 = aggregate_density_vector(patients[p], t=0, g=g, params=params)
            a1, audit1 = aggregate_density_vector(patients[p], t=1, g=g, params=params)
            S0, S1 = a0.sum(), a1.sum()

            # Active set (semantic): based on total counts across pre+post in (p,g)
            active_mask, mass_pruned_ratio = compute_active_mask_by_counts(patients[p], g, params)

            outputs[p][g] = {
                "S0": S0, "S1": S1, "scale_ratio": safe_div(S1, S0),
                "audit_pre": audit0, "audit_post": audit1,
                "mass_pruned_ratio": mass_pruned_ratio,
                "lambda_dens": lambda_dens[g], "lambda_shape": lambda_shape[g],
                "tau_g": tau_g[g], "s_C": s_C,
                "eta_floor": params.eta_floor, "n_min_proto": params.n_min_proto,
                "eps_schedule_id": params.eps_schedule_id
            }

            # Density-level UOT
            Pi_d, metrics_d, events_d = solve_and_decompose_uot(
                a0, a1, z, s_C, lambda_dens[g], tau_g[g],
                active_mask=active_mask, params=params
            )

            # Shape-level UOT
            if S0 > 0 and S1 > 0:
                a0s, a1s = a0 / S0, a1 / S1
                Pi_s, metrics_s, events_s = solve_and_decompose_uot(
                    a0s, a1s, z, s_C, lambda_shape[g], tau_g[g],
                    active_mask=active_mask, params=params
                )
            else:
                Pi_s, metrics_s, events_s = None, None, None

            # Drift alignment flags (per batch)
            batch_id = patients[p].batch_id
            drift_vec = drift_by_batch.get(batch_id, None)
            if drift_vec is not None:
                flag_drift(events_d, z, drift_vec, params)
                if events_s is not None:
                    flag_drift(events_s, z, drift_vec, params)

            outputs[p][g].update({
                "density": {"Pi": Pi_d, "metrics": metrics_d, "events": events_d},
                "shape":   {"Pi": Pi_s, "metrics": metrics_s, "events": events_s},
            })

    # 11) Uncertainty quantification (ROI bootstrap; single-ROI grid block frozen)
    for p in patients:
        for g in outputs[p]:
            uq = bootstrap_uq(patients[p], g, z, s_C, lambda_dens[g], lambda_shape[g],
                             tau_g[g], params)
            outputs[p][g]["UQ"] = uq

    # 12) Remapping package
    for p in patients:
        outputs[p]["remap"] = build_remap_package(patients[p], outputs[p], params)

    # 13) Cohort-level inference (two-part models)
    cohort_stats = run_cohort_inference(outputs, patients, params)
    outputs["_cohort"] = cohort_stats

    return outputs
# Algorithm 2: Group-wise calibration of lambda_dens / lambda_shape and tau_g
def calibrate_lambdas_groupwise(patients, z, s_C, params):
    lambda_dens, lambda_shape = {}, {}
    for g in all_groups(patients):
        pairs = collect_within_time_pairs(patients, t=0, g=g,
                                          max_pairs_per_patient=params.max_pairs)
        if len(pairs) < params.min_pairs_for_calibration:
            continue

        # fixed eps schedule (must equal inference)
        U_dens, U_shape = [], []
        for lam in params.lambda_grid:
            uvals_d, uvals_s = [], []
            for (roiA, roiB) in pairs:
                a = roi_density_vector(roiA, params)  # counts/area
                b = roi_density_vector(roiB, params)

                Pi, met, _ = solve_and_decompose_uot(
                    a, b, z, s_C, lam, tau=None,
                    active_mask=None, params=params, return_events=False
                )
                # unmatched ratio using positive-part masses
                uvals_d.append((met["B_pos"] + met["D_pos"]) / (met["T"] + met["B_pos"] + met["D_pos"] + 1e-12))

                if a.sum() > 0 and b.sum() > 0:
                    as_, bs_ = a / a.sum(), b / b.sum()
                    Pi, met, _ = solve_and_decompose_uot(
                        as_, bs_, z, s_C, lam, tau=None,
                        active_mask=None, params=params, return_events=False
                    )
                    uvals_s.append((met["B_pos"] + met["D_pos"]) / (met["T"] + met["B_pos"] + met["D_pos"] + 1e-12))

            U_dens.append((lam, median(uvals_d)))
            if len(uvals_s) > 0:
                U_shape.append((lam, median(uvals_s)))

        lambda_dens[g]  = argmin_abs(U_dens,  target=params.alpha)
        lambda_shape[g] = argmin_abs(U_shape, target=params.alpha)

        save_unmatched_vs_lambda_curve(g, U_dens, U_shape, params)  # defensive evidence

    fill_missing_with_global_defaults(lambda_dens, lambda_shape, patients, params)
    return lambda_dens, lambda_shape


def calibrate_tau_groupwise(patients, z, s_C, lambda_dens, params):
    # tau_g from Pi-weighted cost distribution under baseline pre ROI-ROI pairs
    tau_g = {}
    C_full = pairwise_dist2(z) / s_C

    for g in all_groups(patients):
        pairs = collect_within_time_pairs(patients, t=0, g=g,
                                          max_pairs_per_patient=params.max_pairs_tau)
        if len(pairs) < params.min_pairs_for_tau:
            continue

        costs, weights = [], []
        lam = lambda_dens[g]
        for (roiA, roiB) in pairs:
            a = roi_density_vector(roiA, params)
            b = roi_density_vector(roiB, params)
            Pi, met, _ = solve_and_decompose_uot(
                a, b, z, s_C, lam, tau=None,
                active_mask=None, params=params, return_events=False
            )
            costs.append(C_full.reshape(-1))
            weights.append(Pi.reshape(-1))

        tau_g[g] = weighted_quantile(np.concatenate(costs), np.concatenate(weights), q=params.tau_q)

    fill_missing_tau_with_global_default(tau_g, patients, params)
    return tau_g
# Algorithm 3: Solve UOT + Decompose (V1.6: eps-scaling + positive-part + tau_g + active mask + eta_floor)
def solve_and_decompose_uot(a, b, z, s_C, lam, tau, active_mask, params, return_events=True):
    # 1) Active set (semantic), optional (if None, use all)
    if active_mask is None:
        active_mask = np.ones_like(a, dtype=bool)

    aA, bA = a[active_mask], b[active_mask]
    zA = z[np.where(active_mask)[0]]

    # 2) Numerical floor only (avoid log(0))
    aA = np.maximum(aA, params.eta_floor)
    bA = np.maximum(bA, params.eta_floor)

    # 3) Cost matrix with GLOBAL static scaling
    C = pairwise_dist2(zA) / s_C

    # 4) UOT solve: log-domain + clamp + eps-scaling (explicit schedule)
    Pi = uot_sinkhorn_log_eps_scaling(
        aA, bA, C, lam,
        eps_schedule=params.eps_schedule,
        max_iter=params.max_iter, tol=params.tol, clamp=params.clamp_range
    )

    # 5) Metrics (V1.6: positive-part B/D)
    T = Pi.sum()
    pre_marg  = Pi.sum(axis=1)
    post_marg = Pi.sum(axis=0)

    D_pos = np.maximum(aA - pre_marg, 0.0).sum()
    B_pos = np.maximum(bA - post_marg, 0.0).sum()

    d_rel = D_pos / (aA.sum() + 1e-12)
    b_rel = B_pos / (bA.sum() + 1e-12)

    M = (C * Pi).sum() / (T + 1e-12)

    # Retention uses tau_g by default; if tau is None, fall back to params.tau_fixed (optional)
    tau_use = tau if (tau is not None) else params.tau_fixed
    R = Pi[C <= tau_use].sum() / (T + 1e-12)

    metrics = {"T": T, "D_pos": D_pos, "B_pos": B_pos,
               "d_rel": d_rel, "b_rel": b_rel,
               "M": M, "R": R, "tau": tau_use}

    events = None
    if return_events:
        events = extract_events(Pi, C, aA, bA, params)

    return Pi, metrics, events
附：默认参数建议（可在实现文档中固化）
n_min_proto: 2（保守）或 5（更强稀疏裁剪）；必须记录并输出 mass_pruned_ratio
eta_floor: 1e-12（仅数值稳定；与密度/比例尺度无关时需谨慎，建议以 float epsilon 为基准）
tau_q: 0.25 或 0.5（预先锁定）；敏感性用 {0.5τ_g, τ_g, 2τ_g}
drift_thr: 0.85（仅风险标记；必须输出包含/排除 drift-aligned edges 两版）
eps_schedule: 明确列表（例如 [10, 5, 2.5, 1.25, 0.625, 0.3125, 0.15625]）并在校准与推断共享
