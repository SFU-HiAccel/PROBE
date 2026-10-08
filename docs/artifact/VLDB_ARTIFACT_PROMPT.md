# PROBE VLDB 开源 artifact 整理 prompt

你是一名数据库系统工程师。请将 PROBE 仓库整理为可公开发布、可由独立研究者安装和复现实验的 VLDB artifact。算法实现、默认参数、实验配置和结果生成流程必须对齐最新版论文。请实际完成代码、配置、文档和验证工作，先建立对照清单，再按依赖顺序实施。

## 工作范围与论文版本

- 代码仓库：当前 PROBE 工作区；目标分支为 `vldb-artifact`。该分支已从 `exp/budget-f1-curves` 的 `69eca25` 创建，本地仍有大量未提交的最新代码与实验文件。先检查 `git status`、worktree 和运行中的实验，逐项识别发布所需文件，不能把 HEAD 当成全部实现，也不能直接 `git add .`。
- 论文工作区：优先读取 `PAPER_ROOT`；本环境为相邻目录 `../PROBE-VLDB`。当前已核对 2026 年 10 月 8 日的论文提交 `815f9fb`，采样、传播、校准与实验协议已逐项复核。仍须以工作区文件内容为准，不能忽略后续已暂存和未暂存修订。
- `docs/artifact/manuscript_snapshot.json` 保存本次核对的版本和 SHA256。开工时核对文件哈希；论文发生更新时，重新读取受影响章节、更新对照表和快照。不要沿用旧结论，也不要自动覆盖论文。
- 必读 `main.tex` 实际启用的 inputs，以及 `sections/02_background.tex`、`03_sampling.tex`、`04_label_propagation.tex`、`05_learned_guided_fallback.tex`、`06_evaluation.tex`、`figures/sas_flow.tex`、`tables/best_points.tex`。跟踪每个图表使用的数据、工作簿和绘图脚本。
- `sections/09_appendix.tex` 当前存在，但其 input 在 `main.tex` 中被注释。单独审查其中参数、敏感性实验和校准说明，标明它是否属于最终提交版本，不能把它无条件当作已启用正文。
- 历史 README、`experiments/PLAN_PROMPT.md`、旧 LASO/SIGMOD 计划和 `docs/design/` 中的早期 prompt 仅作线索。若与最新版论文冲突，以新版论文为实现目标，把冲突及证据记录下来。不要通过修改论文来掩盖代码或结果差异。

本次授权包括开源整理分支和默认分支设置。开工时读取远端实际默认分支、分支 SHA 和当前凭据权限；已经完成的设置无需重复变更。先完成可审查的提交，再正常推送；有管理员权限时设置 GitHub 默认分支并读回验证。无权限时完成所有本地工作，明确报告待执行的远端动作；本地切换分支或设置 `origin/HEAD` 不代表远端默认分支已修改。

## 算法对齐要求

建立 `docs/artifact/algorithm_alignment.md`，逐项列出论文 label/公式/算法行、实现函数、实际生效配置、差异、修复和验证证据。以下内容已与论文 `815f9fb` 的正文核对，后续仍需与实时论文核对。

1. **固定排序与特征。** 默认以 Qwen3-VL-Reranker-2B 对谓词和候选行联合编码，一次按分数降序排序。Join 必须联合编码谓词及左右两行；保持原始 candidate ID 和列的对应关系。代理训练及图构建使用同次 CE 前向的 last-token hidden state，经单位归一化和 256 维 PCA。查验归一化与投影顺序、拟合范围和小样本约定。旧的多轮 SigLIP/LR 混合重排不能作为新版默认流程。
2. **预算。** `B` 是采样加校准的准备预算；`B_L=floor(0.8B)`、`B_A=B-B_L`。路由和 S3 全查询 fallback 在 `B` 之外，不能为了维持旧 hard cap 改变论文行为。稳定 ID 去重、已知标签复用，同时分别记录候选标签数、物理请求数、失败和重试成本。
3. **Uniform pilot 与一次门控。** `n0=floor(0.3B_L)`，无放回均匀抽样；默认 batch 32、`p_star=0.05`、置信尾部 0.05。正文 Eq. `eq:gate` 使用单侧超几何**下界** `pi_low`；仅当 `pi_low > p_star` 时永久进入分层采样，否则进入 multi-interval。不能误用早期 upper-bound LOW 门控。零命中或空 pilot 不证明查询为空。
4. **SAS 两条分支。** 分层分支使用最多 10 个几何 rank strata，由 pilot 估计 Neyman 权重，同时保留非空层最低探索配额；批量整数配额总和不得超过剩余预算。Multi-interval 每批保留 0.25 均匀探索，使用累计小数配额；不足 `n_min=2` 个已知正例时，其余部分从高分端扫描。达到条件后按论文 DP 拟合至多 6 个不相交区间，区间惩罚 1.5，更新 gap/window、收缩因子和种子配额。检查 DP、孤立种子、区间内部抽样与 allocation 的完整对应关系。
5. **S3。** 采样结束仍缺任一类别时，跳过代理、传播与校准路由，保留所有已知标签，对剩余候选调用 LLM。不能把未标注项当负例，不能制造伪正例；fallback 不是丢弃已完成采样后重跑。
6. **LR 与图传播。** 默认 class-balanced L2 LR，`C=1`。构建对称加权 kNN 图，Gaussian bandwidth 为第 k 邻居距离的中位数，使用行归一化 `P=diag(W1)^(-1)W`。两阶段分别为 pin residual 的 bias diffusion 和 pin labels 的 homogeneity propagation；阻尼均 0.8、迭代上限各 50、标签阈值 0.5。正文参数表写 `round(3log2N)`，未启用附录写带 5/50 上下限的版本；先记录冲突并查实际实验配置，不能悄悄任选一个。
7. **校准与交叉拟合。** `A` 从未标注集合 `U=X\L` 的 refined-score strata 抽样：最多 10 个等宽 strata、0.2 均摊份额、剩余按 Neyman 分配，保留实际 inclusion probability 和逆概率权重。`L` 的 OOF 特征必须在各折重新训练代理并重跑传播，不能让 held-out 标签进入图锚点。折数为 `min(5,n_pos,n_neg)`；检查证据不足时的行为。
8. **Platt 与 level anchoring。** 校准形式为非负 slope 乘 refined-score logit，加 intercept；`L` 项另含偏差 offset `xi`。按正文混合加权 A loss 与 L loss，`zeta=0.25`、ridge 项 `0.5xi²`，使用 smoothed targets。仅由 `A` 的 Horvitz–Thompson 估计设置 level anchor，路由不使用 `xi`。零正例分支的 CP mixture 参数为 0.15/0.05；它是正则化启发式，不能包装成分层无放回抽样的严格有限总体证书。核对空 A、全正 A、拟合失败及 cross-fitting 失败分支。
9. **路由。** 保留 refined label；枚举 predicted-positive 低分前缀与 predicted-negative 高分前缀，按论文 counts、two-threshold 和 Dinkelbach 求解 `alpha*(1-estimated_F1)+(1-alpha)*delegation`。默认 `alpha=0.5`，`alpha=0` 不路由。估计目标是与 LLM 标签的一致性；对外报告的 F1 与人工 GT 比较，两种口径必须分清。

重点检查 `fixed_ce_sampling.py`、`threshold_search_multi.py`、`sampling_strata.py`、`ce_ranker.py`、`ce_ranker_backend.py`、`fixed_ce_config.py`、`cluster_proxy.py`、`proxy_model.py`、`weighted_calibrator.py`、`stratified_calibration.py`、`learned_calibration.py`、`budgeted_calibrated_router.py`、`oracle_budget.py`、`candidate_identity.py`、`sem_ops_register.py` 和 `run.py`。部分文件当前尚未跟踪，必须纳入依赖闭包。

已发现 `SamplingConfig()` 默认仍是 legacy/hybrid pilot、fraction 0.1、`n_min=1`，虽然已有 `paper_config()`，但 public operator 和 `run.py` 默认路径未必调用它。新版 artifact 必须在公开入口、CLI、dry-run 和论文配置中实际启用同一套新版协议；存在 paper helper 或通过局部测试不能算完成。历史协议只保留在有明确版本标识的复现入口，不能让它默默影响新版实验。

## 实验与图表对齐要求

建立机器可读的 `experiments/paper_manifest.json` 和人可读的 `docs/artifact/experiment_alignment.md`。覆盖实际启用正文的全部实验、图表和主要数值结论，按 LaTeX label 定位，不能依赖旧图号。每项必须包括查询集、candidate universe、规模、模型 revision、prompt、budget/target、seed/repeats、消融开关、命令、raw run、汇总脚本、绘图脚本、输出、统计与计时口径、当前缺口。

| 论文实验或图表 label | 应复现的协议 |
| --- | --- |
| `fig:sys-f1`、`fig:sys-latency` | 默认 14 查询；PROBE/AIQA/CSV 使用 10% 准备或采样预算，PROBE alpha 0.5；target-based 系统 target 0.9；主质量图为五次运行的均值及均值的 95% CI |
| `fig:sys-tradeoff`、`tab:main-comparison` | Budget sweep 5/10/15/20/25%；target sweep 0.1/0.3/0.5/0.7/0.9；从同一 sweep 自动选择 quality-first 与 cost-first 点；保留正文的 Ecomm Q9 特例 |
| `fig:ablation` 的 sampling panel | SAS/uniform/Neyman-stratified/top-K；共用 class-balanced LR；关闭 refinement、calibration、routing；核对 B 与 B_L 的实际使用 |
| `fig:ablation` 的 refinement panel | 固定 SAS 标签和代理，比较 no-refinement、bias diffusion、完整两阶段；明确其他阶段开关 |
| `fig:alpha-tradeoff` | 同一 sampling/refinement/routing，比较 Platt、GAM、无校准；从当前图数据核对完整 alpha 网格；BARGAIN-A/P/R 使用正文对应的 target 和预算语义 |
| `fig:repr-models` | Qwen3-VL-Reranker-2B、jina-reranker-m0、SigLIP、CLIP-base；按协议同时替换排序与表示，保持其他条件相同 |
| `fig:mllm-generalization` | Qwen3-VL-30B-A3B-Instruct、Qwen3.8-27B、Gemma 4 26B-A4B、GLM-4.6V-Flash；B=10%、alpha=0.5；每个模型使用自身 no-proxy 对照 |
| `fig:time-stack` | 七阶段真实端到端计时；encoder、sampling、LR training/prediction、graph/refinement、calibration、LLM calls 均纳入并解释重叠和加总 |
| `tab:lowsel-results` | Cars Q4、Ecomm Q6、Movie Q5；PROBE B=10%/25%、alpha=0.5；BARGAIN-A、ScaleDoc target 0.9 和 no-proxy；核查 permanent-stratified gate 及样本规模 |
| `fig:encoder-rank`、`fig:sas-multi-process` | Ecomm Q9 的真实排序和采样 trace；完整 25,440 candidates；图示 batch 300 是可读性设置，与默认 batch 32 分开标明 |

首次核查的主查询集及论文 candidate 数如下；实际运行若使用 scale factor、blocking 或子采样，必须同时记录 paper N 和 run N，禁止跨规模直接拼接比较。

| 查询 | 论文 N | 查询 | 论文 N |
| --- | ---: | --- | ---: |
| MMQA Q2a | 10,400 | MMQA Q7 | 230,400 |
| Ecomm Q7 | 525,625 | Ecomm Q8 | 115,768 |
| Ecomm Q9 | 25,440 | Ecomm Q13 | 3,992 |
| Cars Q3 | 10,070 | Cars Q4 | 157,269 |
| Cars Q8 | 30,130 | Animals Q1 | 1,600 |
| LRO 113 | 266 | LRO 121 | 1,426 |
| LRO 202 | 384 | LRO 305 | 1,421 |

主 expert 为 Gemma 4 26B-A4B，共用 vLLM batching 和 prompts。基线协议按最新版正文执行：LOTUS 用 Qwen3-VL-2B token log-probability proxy、precision/recall 0.9、delta 0.1、sampling 10%；BARGAIN accuracy mode 0.9、delta 0.1，复用 PROBE refined proxy；ScaleDoc training/calibration 各 5%、F1 target 0.9；CSV cluster 数 4/2、每层 sampling 10%、一致率 90%；AIQA 使用 UQE active sampling；no-proxy 对所有候选调用同一 expert。不要把历史独立 BARGAIN proxy、learned_deferral 或单 seed case grid 当作新版主实验。

对论文 `815f9fb` 新版正文的主要数值结论建立独立核验条目：默认 PROBE/no-proxy geometric-mean F1 为 0.762/0.797、PROBE geometric-mean delegation 为 12.8%、speedup 为 4.88 倍；默认 ScaleDoc/BARGAIN/LOTUS geometric-mean F1 为 0.530/0.628/0.484，其 median delegation 为 81%/21%/80%。逐项核对这些统计的 query coverage、真实运行来源和聚合口径，不能把默认配置的 median 与完整 sweep 的 median 混用。数值是论文需要证据支持的结论，不能据此填造或手调 raw results。

候选实现与数据线索包括 `experiments/gemma4/`、`fig_gemma4/`、`tradeoff/`、`best_points/`、`sampling_ablation/`、`sas_ablation_ci/`、`stage_ablation/`、`calib_ablation/`、`encoder_ablation_ci/`、`mllm/`、`time_breakdown/`、`fixed_ce/`、`paper_data/`、`paper_figures/`、根目录 `paper_figures/` 和论文 `data/*.xlsx`。先追踪 provenance，再选正式入口；目录名、文件时间和已有 PDF 不能证明协议匹配或实验完成。

未启用附录中的 sensitivity、GAMCAL case 和其他图表应单独列入状态表，保留其真实历史协议和覆盖查询数。只有最终论文启用或明确要求提交的内容才进入正式复现清单；正文对 CP heuristic 等解释仍应保留在 artifact 文档。

## 数据真实性与统计计时

- Raw predictions、真实 oracle responses、resolved config 与运行 provenance 是实验事实依据。论文和工作簿定义目标呈现，但手工编辑后的单元格不能反向充当 raw run。
- 每个发布数值须追溯到真实 run/config/seed/模型和 scorer。记录 binary candidate F1、semantic-set F1、计数质量、ECE 等各自定义，不能只改名称就混用。
- 明确 delegation 分子是否为唯一已标注候选，另列全部物理请求、cache hits、失败与重试；记录 sampling/calibration/routing/fallback 分解和 N 分母。
- 区分单次值、五次均值、标准差和 95% CI。不能用另一 PROBE variant 的方差给当前 variant 画误差条；不齐 seed 的组必须显式报缺失。论文只规定部分实验重复次数时，逐项核对，不自动把历史 seed0 标为 five-run mean。
- 核对每幅图和总体结论使用 arithmetic mean 还是 geometric mean，统一 zero/missing 的规则和 query coverage。只用可验证的原始值重算结论，不能挑点或人工调分来逼近论文数字。
- 延迟包含实际声明的全流程；注明 cold/warm cache、batching、GPU 共享和 idle/serial 状态。已有脚本中 `no_proxy_time * N_paper/N_run * delegation` 和 `100/delegation` 之类仅是估算，不能作为实测端到端 latency/speedup。保留并标注历史估算；缺少有效测量时安排合规补跑。
- 图形应由 raw → validated long-form → aggregate → figure/table 的确定性流水线生成。异常修正须有规则、原始值和理由；不能把 synthetic、GT-oracle replay、手工同步或旧协议结果混为新版真实实验。

## 仓库整理与公开交付

1. 盘点 tracked/untracked 文件、依赖、各 worktree 和 live 实验。保留研究原件及已完成结果；不要 `reset --hard`、`clean -fd`、force-push、删除旧分支或终止现有任务。清理目标应在独立 release worktree 中落实。需要算法或模块迁移时，提供调用对应表并修复 public imports、scripts 和 tests。
2. 正式发布保留算法依赖闭包、统一 experiment entrypoints、configs、scorers、绘图代码、轻量参考结果和必要 tests。备份文件、个人 slides、浏览器产物、缓存、临时队列与过时重复实现移出发布树；必要历史脚本放入有版本说明的 legacy 目录。清理不能删除复现当前论文的唯一入口或数据证据。
3. 重写英文 README，并提供 `docs/artifact/ARTIFACT.md`：系统定位、算法对应关系、安装、资源需求、数据/模型下载、快速样例、逐实验命令、预计时间与调用成本、结果和图表生成、限制及问题排查。核对现有 SIGMOD/LASO 命名，避免把它作为新版默认说明。
4. 数据和模型根目录、oracle endpoint、设备、output、seed 和 model revision 通过 CLI/config/env 设置；发布运行路径不得依赖作者 home、其他私人 checkout 或隐藏缓存。单独锁定 client 与 vLLM server 环境，说明 Python/Java/Spark/PyTorch/CUDA/Transformers/vLLM 的兼容组合；不能直接把过时 README 或当前环境快照冒充论文环境。
5. 每个实验提供一个可复制执行的命令，支持打印 resolved config 的 dry-run、可中断恢复的输出管理和防重复启动。再提供集中 reproduce-all 和 plot-all 入口。`paper`、`smoke`、`historical` 配置身份必须显式；缩小规模 smoke 不能叫 full reproduction。
6. 数据提供公开链接、固定版本/校验和、准备与 GT 生成脚本、访问限制及取得方式。权重和大型结果可外置，但必须给出稳定下载和校验。检查本仓库和 vendored baselines 的 license/NOTICE，保留上游归属；已有 MIT license 的匿名 SIGMOD copyright 不得擅自改成未经确认的权利人。
7. 对候选发布树和可达 Git 历史检查密钥、token、私人数据及个人绝对路径，报告文件名和问题类型，不能回显秘密。发现历史泄漏时先制定处理方案，不能以未经授权的历史重写替代安全发布。`.gitignore` 不会清除已跟踪或历史中的内容。
8. 参考 [PVLDB Reproducibility 官方要求](https://www.vldb.org/pvldb/reproducibility)：提供系统/配置/环境、输入数据或取得/生成方式、产生 raw results 的实验脚本和 raw-to-figure 脚本，使独立研究者能够检验论文主要结论。代码公开可访问和实验可复现分别验收。

## 验证与完成标准

在干净 clone/release worktree 上，从文档第一条安装命令开始验证。所有检查输出都记录版本、命令、时间与实际覆盖范围。

- 先检查算法对照，再运行与改动相关的有效 tests：下界 gate 与严格比较边界、永久分层、无/单类别 fallback、tiny budget 与配额总和、固定排序和 ID 对齐、OOF 无泄漏、Platt anchor/fit failure、路由 counts 和 oracle 成本。CPU substitutes 只证明其覆盖的契约，不证明真实模型质量。
- 所有正式 experiment entrypoints 的 dry-run 必须解析同一 paper config，输出足够判断实际默认行为的信息，并且不启动 Spark 或 oracle 请求。
- 验证实际模型通过 public Spark semantic filter 和 join 的小规模端到端运行，再验证完整论文 grid。真实 GPU/oracle 任务先核查现有 `/proc`/服务和资源占用，使用串行或资源隔离队列，不重复启动、不切换别人正在使用的 server、不干扰现有实验。
- 从合法的参考 raw results 重绘所有正式图表，检查每个数值、seed、query coverage、预算、指标和实测计时 provenance；不同硬件可以数值不同，但结论必须由证据支持。
- 缺数据、模型或测量时先完成独立可做的工作，把缺口精确列为 missing/partial/protocol mismatch。补跑应使用当前论文协议，保留旧结果，不能复用错误缓存或借历史已完成任务证明新版完整性。
- 分阶段形成有说明的 commits。最终推送并设置 `vldb-artifact` 为远端默认，读回 repository.default_branch 和 branch SHA，并从该默认分支重新 clone 验证。权限不足时此项保持未完成。

交付包括：整理后的分支、README/ARTIFACT、论文版本记录、算法和实验对照表、统一实验与绘图入口、环境/数据说明、真实验证报告及明确的剩余缺口。最终汇报应说明具体改动、验证覆盖、图表复现状态、无法复现的原因和远端默认分支实际状态。只有所有要求都有当前状态证据且无必要工作剩余时，才可宣布 artifact 整理完成。
