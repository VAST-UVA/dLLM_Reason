# References

Papers and codebases this project builds on or is directly inspired by.

---

## Core Discrete Diffusion LMs

| Paper | Where used |
|---|---|
| **MDLM** — Masked Diffusion Language Models. Sahoo et al., 2024. [arXiv:2406.07524](https://arxiv.org/abs/2406.07524) | `models/mdlm.py`, noise schedule, training loss |
| **SEDD** — Score Entropy Discrete Diffusion. Lou et al., 2024. [arXiv:2310.16834](https://arxiv.org/abs/2310.16834) | `models/sedd.py` |
| **D3PM** — Structured Denoising Diffusion Probabilistic Models. Austin et al., 2021. [arXiv:2107.03006](https://arxiv.org/abs/2107.03006) | `models/d3pm.py` |
| **LLaDA** — Large Language Diffusion with mAsking. GSAI-ML, 2024. [GitHub](https://github.com/GSAI-ML/LLaDA) | `models/llada.py`, block-wise sampling loop in `inference/sampler.py` |

---

## Reinforcement Learning for Diffusion LMs

| Paper | Where used |
|---|---|
| **d1** — Scaling Reasoning in Diffusion Large Language Models. 2024. [GitHub](https://github.com/dllm-reasoning/d1) | `training/rl_train.py` — DiffuGRPO base design |
| **DiFFPO** — Training Diffusion LLMs to Reason Fast and Furious via Reinforcement Learning. Zhao, Liang, Tang, Yao, Kallus, 2024. [arXiv:2510.02212](https://arxiv.org/abs/2510.02212) | `training/rl_train.py` — `DiFFPO` class; `scripts/learn_from_episodes.py` `--mode diffppo` |
| **UnmaskPolicy** — Learning Unmasking Policies for Diffusion Language Models. Jazbec, Olausson, Béthune, Ablin, Kirchhof, Monteiro, Turrisi, Ramapuram, Cuturi, 2025. [arXiv:2512.09106](https://arxiv.org/abs/2512.09106) | `training/rl_train.py` — `UnmaskingPolicyNet`, `UnmaskingPolicyRL`; `scripts/learn_from_episodes.py` `--mode unmask_rl` |

**DiFFPO** introduces two innovations adopted here:
1. **Surrogate-policy PPO** — off-policy RL with importance-ratio clipping (`ppo_clip_eps`), replacing plain policy gradient.
2. **Joint sampler–model training** — `StepBudgetController` predicts the optimal denoising step budget per prompt, improving the inference-time compute Pareto frontier.

**UnmaskPolicy** introduces process-level RL for dLLMs:
1. **MDP formulation** — masked diffusion sampling as a Markov Decision Process; state = per-token confidence, action = binary unmask vector.
2. **Frozen LM, trainable policy** — a lightweight single-layer transformer policy (`UnmaskingPolicyNet`) is trained via REINFORCE; LM weights are untouched.

---

## Token Ordering & Unmasking Strategies

| Paper | Where used |
|---|---|
| **MaskGIT** — Masked Generative Image Transformer. Chang et al., 2022. [arXiv:2202.04200](https://arxiv.org/abs/2202.04200) | `scheduler/maskgit_scheduler.py` — cosine unmasking schedule |
| **SemAR** — Semi-Autoregressive generation. | `scheduler/semi_ar_scheduler.py` |

---

## DAG Search & Structure Learning

| Paper | Where used |
|---|---|
| **NOTEARS** — DAGs with NO TEARS. Zheng et al., 2018. [arXiv:1803.01422](https://arxiv.org/abs/1803.01422) | `search/differentiable.py` — differentiable acyclicity constraint |
| **DARTS** — Differentiable Architecture Search. Liu et al., 2019. [arXiv:1806.09055](https://arxiv.org/abs/1806.09055) | `search/nas_search.py` — NAS-style super-net search |

---

## Evaluation & Datasets

| Resource | Where used |
|---|---|
| **GSM8K** — Grade School Math. Cobbe et al., 2021. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) | `data/reasoning_datasets.py`, `scripts/collect_episodes.py` |
| **MATH** — Measuring Mathematical Problem Solving. Hendrycks et al., 2021. [arXiv:2103.03874](https://arxiv.org/abs/2103.03874) | `data/reasoning_datasets.py` |
| **ARC** — Think you have Solved Question Answering? Clark et al., 2018. [arXiv:1803.05457](https://arxiv.org/abs/1803.05457) | `data/reasoning_datasets.py` |
