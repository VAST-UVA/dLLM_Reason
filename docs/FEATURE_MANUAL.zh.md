# dLLM-Reason 功能手册（Feature Manual）

> 版本：v1.5.0  |  最后更新：2026-04-09  |  语言：中文  |  English: [FEATURE_MANUAL.md](FEATURE_MANUAL.md)

## 0. 项目定位

**dLLM-Reason** 是一个基于离散扩散语言模型（discrete diffusion LM, dLLM）的推理研究框架。其核心创新是用 **TokenDAG** 约束 token unmask 的偏序，从而把"推理依赖"显式地注入扩散生成过程。

该框架允许：

- 在任意 dLLM 后端（MDLM / SEDD / D3PM / LLaDA）之上，热切换不同的 **unmasking scheduler**；
- 在推理任务上 **搜索最优 DAG 结构**（6 种搜索方法）；
- 通过 **Episode Pipeline** 把采样轨迹持久化为可复用的"推理经验"；
- 从 Episode 库进行 **SFT / GRPO / DiFFPO / UnmaskRL** 等多种学习；
- 在 10 个 benchmark 上做统一评测；
- 通过 **FastAPI 服务** 和 **Web UI** 进行交互式调试。

---

## 1. 顶层架构

```
                     ┌──────────────────┐
                     │   DiffusionLM    │
                     │ (MDLM/SEDD/D3PM) │
                     └────────┬─────────┘
                              │ logits + confidences
                              ▼
 ┌──────────────┐    ┌────────────────────┐    ┌──────────────┐
 │  TokenDAG    │───>│ UnmaskingScheduler │───>│  Sampled     │
 │  (structure) │    │ (DAGScheduler)     │    │  Sequence    │
 └──────────────┘    └────────────────────┘    └──────────────┘
         ▲                                            │
         │           ┌────────────────────┐           │
         └───────────│  DAGSearcher       │<──────────┘
                     │  (eval feedback)   │   accuracy / reward
                     └────────────────────┘
```

关键 insight：**DAG 约束 ORDER，dLLM 提供 TOKEN PREDICTIONS，两者解耦**。任意 dLLM + 任意 scheduler 可自由组合。

---

## 2. 模型层 `dllm_reason.models`

统一接口 `DiffusionLM`（`models/base.py`）：

```python
class DiffusionLM(nn.Module):
    def forward(self, x_t, t) -> Tensor: ...
    def compute_loss(self, x_0) -> Tensor: ...
    def noise_input(self, x_0, t) -> Tensor: ...
    def sample(self, scheduler, ...) -> Tensor: ...
```

| 模型 | 文件 | 说明 |
|------|------|------|
| **MDLM** | `models/mdlm.py` | Masked diffusion LM；最简单的 baseline |
| **SEDD** | `models/sedd.py` | Score-Entropy Discrete Diffusion |
| **D3PM** | `models/d3pm.py` | Discrete diffusion prior，支持 absorbing / uniform |
| **LLaDA** | `models/llada.py` | HuggingFace `GSAI-ML/LLaDA-8B` 包装器 |

后端 Transformer（`models/backbone/transformer.py`）：
- Pre-LN, RMSNorm, SwiGLU, Rotary Position Embedding
- 支持 timestep embedding（AdaLN / FiLM）

---

## 3. Graph 层 `dllm_reason.graph`

### 3.1 `TokenDAG`（`graph/dag.py`）

- 定义：`(seq_len, seq_len)` bool adjacency tensor，`adj[i, j] = True` 表示 **position i 必须在 j 之前 unmask**
- 核心操作 `ready_positions(unmasked)` 用一次 GPU 矩阵运算确定哪些 position 现在可以 unmask：

```python
ready = (adj.logical_not() | unmasked.unsqueeze(-1)).all(dim=0)
```

- API：
  - `from_edges(edges, seq_len)`
  - `from_levels(levels, seq_len)`
  - `linear_chain(seq_len)`
  - `no_edges(seq_len)`
  - `topological_levels()`
  - `ready_positions(unmasked_mask)`
  - `is_valid()` / `num_edges()`
  - `mutate(p_add, p_remove)`

### 3.2 预定义模板（`graph/templates.py`）

| 名称 | 含义 | 适用场景 |
|------|------|----------|
| `cot` | 输出分段，每段依赖前段，段内并行 | 多步数学推理 |
| `answer_first` | 答案 token 为 root 先 unmask，推理后填 | 验证型推理 |
| `skeleton` | Level 0=结构 token，Level 1=操作数，Level 2=filler | 结构化推理 |
| `bidirectional` | 从两端向中间逐步 unmask | 边界约束任务 |
| `interleaved` | 交错分组，组间顺序、组内并行 | 交错计算/叙述 |
| `linear` | 严格左→右（等价于 AR） | sanity check |
| `random_low` | 随机 DAG，density=0.05 | search 种群 |
| `random_high` | 随机 DAG，density=0.15 | search 种群 |

构造函数：`build_all_templates(seq_len, device)` 一次拿到所有模板；`build_template(name, seq_len, device)` 拿单个。

### 3.3 其他

- `graph/constraints.py`：无环性校验、安全 mutate
- `graph/viz.py`：matplotlib + networkx 可视化

---

## 4. Scheduler 层 `dllm_reason.scheduler`

统一接口：

```python
class UnmaskingScheduler(ABC):
    def select_positions(
        self,
        step: int,
        mask: Tensor,        # 当前 mask
        logits: Tensor,      # (B, L, V)
        confidences: Tensor, # (B, L)
    ) -> Tensor: ...
```

| 名称 | 文件 | 机制 |
|------|------|------|
| `random` | `random_scheduler.py` | 均匀采样 |
| `confidence` | `confidence.py` | top-k max-prob |
| `linear` | `linear_scheduler.py` | 严格左→右 |
| `maskgit_cosine` | `maskgit_cosine.py` | 余弦 schedule |
| `low_confidence_remask` | `low_conf_remask.py` | re-mask 低置信度重新预测 |
| `entropy` | `entropy_scheduler.py` | top-k 负熵 |
| `stochastic_confidence` | `stoch_conf.py` | Gumbel-top-k 采样 |
| `adaptive_dynamic` | `adaptive_dynamic.py` | 置信度 + DAG readiness 加权 |
| `semi_ar` | `semi_ar.py` | 块内并行，块间顺序 |
| `curriculum` | `curriculum.py` | 难度从高到低递进 |
| `dag` | `dag_scheduler.py` | **DAG 引导 ★核心创新** |

**DAGScheduler 每步流程：**

1. 通过 `dag.ready_positions(already_unmasked)` 拿到所有依赖已满足的 position
2. 与仍被 mask 的 position 取交集 → eligible positions
3. 子策略从 eligible 中选 k 个（支持 `all_ready` / `confidence_topk` / `proportional`）
4. 返回选中的 positions

---

## 5. Search 层 `dllm_reason.search`

统一接口：

```python
class DAGSearcher:
    def search(self, model, dataset, fitness_fn, budget) -> TokenDAG: ...
```

| 方法 | 文件 | 适用场景 |
|------|------|----------|
| Greedy | `greedy.py` | 快速 baseline；逐条加/删边 |
| Evolutionary | `evolutionary.py` | population + crossover + mutation |
| RL Policy | `rl_policy.py` | REINFORCE 学 DAG 编辑策略 |
| Differentiable | `differentiable.py` | NOTEARS-style 连续化搜索 |
| End-to-End | `e2e_search.py` | 端到端梯度 |
| NAS Controller | `nas_search.py` | LSTM controller 生成 DAG |

**Fitness 信号**（`search/fitness.py`）：
- `accuracy`, `reward`, `log_likelihood`, `composite`

种群初始化默认用 `build_all_templates(seq_len)` 作为 seeds。

---

## 6. Training 层 `dllm_reason.training`

| 模式 | 文件 | 描述 |
|------|------|------|
| `pretrain` | `pretrain.py` | 标准 MDM 预训练 |
| `finetune` | `finetune.py` | 监督 finetune |
| `dag_aware` | `dag_aware_train.py` | 按 DAG 偏置 mask 分布做训练 |
| `rl` | `rl_train.py` | diffu-GRPO / DiFFPO / UnmaskingPolicyRL |

### 6.1 RL 详解

`rl_train.py` 包含：

- **`DiffuGRPO`**：参考 `d1` 仓库，对 diffusion LM 的 GRPO
- **`DiFFPO`**（Diffusion Fast-Forward Policy Optimization）：
  - PPO-clip advantage
  - `StepBudgetController`：动态控制每个 episode 的扩散步数上限
- **`UnmaskingPolicyNet` + `UnmaskingPolicyRL`**：
  - frozen LM，只训一个小的 policy net 学 scheduler
  - process-level REINFORCE
  - 可把 DAG 结构作为 policy 的输入特征

### 6.2 训练 CLI

```bash
# 基础
dllm-train --model mdlm --dataset gsm8k --mode pretrain

# DAG-aware
dllm-train --model mdlm --dataset gsm8k --dag_aware --dag cot

# RL
dllm-train --model mdlm --dataset gsm8k --mode rl --rl_algo diffu_grpo

# 自定义 run 名字（v1.4.1+）
dllm-train --model mdlm --dataset gsm8k --name my_experiment
```

自动生成的 checkpoint 目录：`checkpoints/<name>_<timestamp>/`

---

## 7. Episode Pipeline & Library `dllm_reason.library`

核心思路：把 dLLM 推理轨迹（prompt, DAG, output, correct, score）作为"经验"持久化，再离线/在线学习。

### 7.1 `DAGEpisode` dataclass

字段：`episode_id, prompt, task_type, ground_truth, strategy_name, dag_seq_len, dag_json, output, correct, score, reward, meta, created_at`

### 7.2 `EpisodeStore`（SQLite WAL 模式）

API：
- `add(episode)` / `add_many(episodes)`
- `query(task_type=..., strategy=..., min_score=..., limit=...)`
- `delete(episode_id)`
- `stats()`
- `close()`

### 7.3 脚本

| 命令 | 功能 |
|------|------|
| `dllm-collect-episodes` | 并发 rollout → 写入 store |
| `dllm-learn-from-episodes` | 从 store 读取 → SFT / GRPO / DiFFPO / UnmaskRL |
| `dllm-inspect-episodes` | CLI 浏览 episode |
| `dllm-manage-library` | 清理、去重、导出 |
| `dllm-add-feedback` | 人类反馈打分入库 |

---

## 8. 评测层 `dllm_reason.eval`

`reasoning_eval.py` 支持 10 个 benchmark：

| 类别 | benchmarks |
|------|-----------|
| 代码 | `mbpp`, `humaneval` |
| 数学 | `gsm8k`, `math`, `aime` |
| 多选 | `arc`, `mmlu`, `gpqa` |
| 多跳 | `hotpotqa` |
| 逻辑 | `prontoqa` |

指标：`exact_match`, `accuracy`, `pass@k`, `rouge`, `reasoning_score`（自定义）

```bash
dllm-evaluate --model mdlm --ckpt <path> --benchmark gsm8k
dllm-eval-dags --model mdlm --benchmark gsm8k --dag_dir <dir>
```

---

## 9. 服务与 UI

### 9.1 FastAPI 服务（`scripts/serve.py`）

- 启动：`dllm-serve --model mdlm --port 8000`
- 端点：
  - `POST /generate` — 生成
  - `POST /switch_strategy` — 热切换 scheduler
  - `GET /strategies` — 列出可用 scheduler
  - `POST /switch_dag` — 热切换 DAG
  - `GET /health`

### 9.2 Web UI（`scripts/webui.py`）

- 启动：`dllm-webui`
- 单文件 HTML Dashboard，支持：
  - 模型/策略切换
  - DAG 可视化
  - Episode 浏览

### 9.3 DAG 分析脚本

- `dllm-visualize-dag` — 导出 PNG/SVG
- `dllm-analyze-dag` — 统计边数、最长路径、ready fan-out

---

## 10. CLI Entry Points（`pyproject.toml [project.scripts]`）

v1.4.2 共 **17 个** entry points：

| 命令 | 脚本 | 功能 |
|------|------|------|
| `dllm-train` | `train.py` | 训练（支持 `--name`） |
| `dllm-evaluate` | `evaluate.py` | 单模型评测 |
| `dllm-eval-dags` | `eval_dags.py` | 批量 DAG 评测 |
| `dllm-search-dag` | `search_dag.py` | 结构搜索 |
| `dllm-visualize-dag` | `visualize_dag.py` | 可视化 |
| `dllm-serve` | `serve.py` | REST 服务 |
| `dllm-webui` | `webui.py` | 浏览器 dashboard |
| `dllm-run-pipeline` | `run_pipeline.py` | 5 阶段端到端管线 |
| `dllm-collect-episodes` | `collect_episodes.py` | Episode 采集 |
| `dllm-learn-from-episodes` | `learn_from_episodes.py` | 离线学习 |
| `dllm-manage-library` | `manage_library.py` | 清理/导出 |
| `dllm-benchmark-schedulers` | `benchmark_schedulers.py` | 多调度对比 |
| `dllm-analyze-dag` | `analyze_dag.py` | 结构统计 |
| `dllm-inspect-episodes` | `inspect_episodes.py` | CLI browse |
| `dllm-generate-templates` | `generate_templates.py` | 生成模板候选 |
| `dllm-add-feedback` | `add_feedback.py` | 人类反馈入库 |
| `dllm-merge-dags` | `merge_dags.py` | DAG 合并（并集/投票） |

---

## 11. 5-Stage Pipeline（`scripts/run_pipeline.py`）

端到端一键管线，支持 `--resume`、`--stop_on_error`。

1. **download** — HuggingFace dataset 缓存
2. **collect** — 用 baseline scheduler 生成初始 Episode 库
3. **search** — 以 Episode 库为 fitness 源搜索 DAG
4. **learn** — 根据 flag 触发 SFT / GRPO / DiFFPO / UnmaskRL
5. **eval** — 对新模型 + 新 DAG 在 benchmark 上评测

示例：

```bash
dllm-run-pipeline \
  --model mdlm \
  --dataset gsm8k \
  --search_method evolutionary \
  --rl_algo diffu_grpo \
  --output_dir runs/exp1
```

---

## 12. 典型工作流

### 12.1 Reproduce baseline

```bash
dllm-train --model mdlm --dataset gsm8k --mode pretrain
dllm-evaluate --model mdlm --ckpt checkpoints/mdlm_gsm8k_pretrain_* --benchmark gsm8k
```

### 12.2 DAG-guided inference

```bash
# 用预定义模板评测
dllm-eval-dags --model mdlm --benchmark gsm8k --templates cot,skeleton,bidirectional

# 搜索最优 DAG
dllm-search-dag --model mdlm --dataset gsm8k --method evolutionary --budget 200

# 用搜索到的 DAG 评测
dllm-evaluate --model mdlm --dag runs/search/best.pt --benchmark gsm8k
```

### 12.3 Episode-based learning

```bash
dllm-collect-episodes --model mdlm --dataset gsm8k --n 10000 --out library.db
dllm-learn-from-episodes --store library.db --mode sft --out ckpt_sft
dllm-learn-from-episodes --store library.db --mode diffu_grpo --out ckpt_rl
```

### 12.4 End-to-end pipeline

```bash
dllm-run-pipeline --model mdlm --dataset gsm8k --output_dir runs/full_exp
```

---

## 13. 配置系统（Hydra）

`configs/` 目录：

```
configs/
├── model/          (mdlm.yaml, sedd.yaml, d3pm.yaml, llada.yaml)
├── graph/          (linear.yaml, cot.yaml, answer_first.yaml, ...)
├── search/         (greedy.yaml, evolutionary.yaml, rl_policy.yaml)
├── task/           (gsm8k.yaml, math.yaml, arc.yaml, ...)
└── experiment/     (组合实验配置)
```

Override 示例：

```bash
dllm-train +experiment=mdlm_gsm8k model.lr=5e-4
```

---

## 14. 模块依赖速查

```
models/base.py            ─┐
models/{mdlm,sedd,d3pm,llada}.py ─┼─→ inference/sampler.py ──→ scripts/evaluate.py
                           │                    │
scheduler/*               ─┘      ┌─────────────┘
graph/dag.py              ──→ scheduler/dag_scheduler.py ──→ inference/dag_sampler.py
graph/templates.py        ──→ search/* ──→ scripts/search_dag.py
library/episode.py        ──→ scripts/collect_episodes.py, learn_from_episodes.py
training/rl_train.py      ──→ scripts/train.py (mode=rl)
eval/reasoning_eval.py    ──→ scripts/evaluate.py, eval_dags.py
```

---

## 15. 版本历史（节选）

- **v1.0** — 初始 MDLM + random/confidence/linear scheduler + DAG core
- **v1.2.3** — SEDD + D3PM baselines，多个 DAG templates
- **v1.3.0** — LLaDA 集成，扩展 benchmarks 到 10 个
- **v1.4.0** — Episode Pipeline + DAG Library + 4 search methods
- **v1.4.1** — Training CLI 增强（`--name` 参数）
- **v1.4.2** — `run_pipeline.py` 5-stage 管线 + 多个新 CLI 脚本
- **v1.4.3** — Bug 修复（`empty()` 重命名），`publish.yml` 仓库守卫
- **v1.5.0** — 研究管线（`run_research_pipeline.py`），消融实验脚本（`run_ablation.py`），批量推理 API，模型热切换

---

## 16. 已知问题

请参见 `docs/BUG_AUDIT_V1.4.2.md`（见随附审计报告），包含 CRITICAL / HIGH / LOW 三级问题列表。本手册对应功能的实现细节可能存在个别 bug，以审计报告和最新 commit 为准。

---

## 17. 致谢与引用

- MDLM: https://github.com/kuleshov-group/mdlm
- SEDD: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
- LLaDA: https://github.com/ML-GSAI/LLaDA
- d1 (diffu-GRPO): https://github.com/dllm-reasoning/d1
- D3PM: https://arxiv.org/abs/2107.03006
- NOTEARS: https://arxiv.org/abs/1803.01422
