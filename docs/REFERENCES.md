# References

All baselines, benchmarks, search methods, and RL algorithms used in dLLM-Reason, with paper references.

> Root-level [REFERENCES.md](../REFERENCES.md) is kept in sync and organised by component type.

---

## Unmasking Strategies (Baselines)

| Strategy | Description | Reference |
|----------|-------------|-----------|
| `confidence` | Unmask highest-confidence (argmax prob) positions first; LLaDA default | Nie et al., "Large Language Diffusion Models", 2025. [[arXiv:2502.09992]](https://arxiv.org/abs/2502.09992) |
| `random` | Uniform random unmasking; standard dLLM baseline | Sahoo et al., "Simple and Effective Masked Diffusion Language Models (MDLM)", 2024. [[arXiv:2406.07524]](https://arxiv.org/abs/2406.07524) |
| `entropy` | Unmask lowest-entropy (most certain by full distribution) first | Chang et al., "MaskGIT: Masked Generative Image Transformer", CVPR 2022. [[arXiv:2202.04200]](https://arxiv.org/abs/2202.04200) |
| `semi_ar` | Semi-autoregressive: block-by-block L-to-R, confidence within each block | Savinov et al., "Step-unrolled Denoising Autoencoders for Text Generation (SUNDAE)", ICML 2022. [[arXiv:2112.06749]](https://arxiv.org/abs/2112.06749) |
| `linear` | Left-to-right sequential (autoregressive simulation) | Standard autoregressive baseline; used as reference in all dLLM papers |
| `cot` | Chain-of-Thought DAG: force reasoning segments before answer | Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", NeurIPS 2022. [[arXiv:2201.11903]](https://arxiv.org/abs/2201.11903) |
| `skeleton` | Skeleton-then-Detail: generate structural tokens first, then fill details | Ning et al., "Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation", 2023. [[arXiv:2307.15337]](https://arxiv.org/abs/2307.15337) |
| `bidirectional` | Unmask from both ends toward center | Yang et al., "XLNet: Generalized Autoregressive Pretraining for Language Understanding", NeurIPS 2019. [[arXiv:1906.08237]](https://arxiv.org/abs/1906.08237) |
| `answer_first` | Generate answer region first, then fill reasoning | Zhao et al., "Verify-and-Edit: A Knowledge-Enhanced Chain-of-Thought Framework", 2023. [[arXiv:2305.03268]](https://arxiv.org/abs/2305.03268) |
| `maskgit_cosine` | MaskGIT cosine schedule: unmask more tokens early, fewer later | Chang et al., "MaskGIT: Masked Generative Image Transformer", CVPR 2022. [[arXiv:2202.04200]](https://arxiv.org/abs/2202.04200) |
| `critical_token_first` | Unmask highest-KL-from-uniform (most influential) positions first | Inspired by attention rollout / gradient-based token importance; Li et al., "Making the V in VQA Matter", CVPR 2017. [[arXiv:1612.00837]](https://arxiv.org/abs/1612.00837) |
| `curriculum` | Easy (high confidence + low entropy) tokens first, hard tokens last | Bengio et al., "Curriculum Learning", ICML 2009. [[Paper]](https://dl.acm.org/doi/10.1145/1553374.1553380) |
| `adaptive_dynamic` | Dynamic soft DAG: constructs pairwise influence graph at runtime (**ours**) | Novel contribution of this work |

---

## Benchmarks

| Benchmark | Type | Metric | Reference |
|-----------|------|--------|-----------|
| GSM8K | Math reasoning | Exact match | Cobbe et al., "Training Verifiers to Solve Math Word Problems", 2021. [[arXiv:2110.14168]](https://arxiv.org/abs/2110.14168) |
| MATH | Competition math | Exact match | Hendrycks et al., "Measuring Mathematical Problem Solving with the MATH Dataset", NeurIPS 2021. [[arXiv:2103.03874]](https://arxiv.org/abs/2103.03874) |
| MBPP | Code generation | pass@1 | Austin et al., "Program Synthesis with Large Language Models", 2021. [[arXiv:2108.07732]](https://arxiv.org/abs/2108.07732) |
| HumanEval | Code generation | pass@1 | Chen et al., "Evaluating Large Language Models Trained on Code", 2021. [[arXiv:2107.03374]](https://arxiv.org/abs/2107.03374) |
| ARC-Challenge | Science reasoning | Accuracy | Clark et al., "Think you have Solved Question Answering? Try ARC", 2018. [[arXiv:1803.05457]](https://arxiv.org/abs/1803.05457) |
| MMLU | Knowledge (multi-subject) | Accuracy (5-shot) | Hendrycks et al., "Measuring Massive Multitask Language Understanding", ICLR 2021. [[arXiv:2009.03300]](https://arxiv.org/abs/2009.03300) |
| HotpotQA | Multi-hop QA | EM / F1 | Yang et al., "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering", EMNLP 2018. [[arXiv:1809.09600]](https://arxiv.org/abs/1809.09600) |
| ProntoQA | Logical reasoning | Accuracy | Saparov & He, "Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought", ICLR 2023. [[arXiv:2210.01240]](https://arxiv.org/abs/2210.01240) (ProntoQA is the benchmark introduced in this paper) |
| GPQA | PhD-level science MCQ (diamond subset) | Accuracy | Rein et al., "GPQA: A Graduate-Level Google-Proof Q&A Benchmark", 2023. [[arXiv:2311.12022]](https://arxiv.org/abs/2311.12022) |
| AIME | Competition math (AMC/AIME), integer answers 000–999 | Accuracy | American Mathematics Competitions, MAA. Problems sourced from Art of Problem Solving. [[AoPS]](https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions) |

---

## DAG Search Methods

| Method | Description | Reference |
|--------|-------------|-----------|
| Greedy | Iterative edge add/remove to maximize fitness | Chickering, "Optimal Structure Identification with Greedy Search", JMLR 2002. [[Paper]](https://www.jmlr.org/papers/v3/chickering02b.html) |
| Evolutionary | Population-based: tournament selection, crossover, mutation | Larranaga et al., "Structure Learning of Bayesian Networks by Genetic Algorithms: A Performance Analysis of Control Parameters", IEEE TPAMI 1996. [[IEEE]](https://ieeexplore.ieee.org/document/537345); see also Champion et al., "Inferring Large Graphs Using l1-Penalized Likelihood (GADAG)", Statistics and Computing 2018. [[arXiv:1507.02018]](https://arxiv.org/abs/1507.02018) |
| RL Policy | REINFORCE with Transformer policy to construct DAGs edge-by-edge | Zhu et al., "Causal Discovery with Reinforcement Learning", ICLR 2020. [[arXiv:1906.04477]](https://arxiv.org/abs/1906.04477) |
| Differentiable (NOTEARS) | Continuous relaxation with augmented Lagrangian acyclicity constraint | Zheng et al., "DAGs with NO TEARS: Continuous Optimization for Structure Learning", NeurIPS 2018. [[arXiv:1803.01422]](https://arxiv.org/abs/1803.01422) |
| End-to-End DAG Learning | Joint DAG structure + task loss optimization via differentiable scheduling | Novel contribution; builds on NOTEARS + Gumbel-Sigmoid relaxation |
| NAS-SuperNet (DARTS-like) | Continuous relaxation over span-level superset DAG | Liu et al., "DARTS: Differentiable Architecture Search", ICLR 2019. [[arXiv:1806.09055]](https://arxiv.org/abs/1806.09055) |
| NAS-Controller (ENAS-like) | GRU controller generates DAG configurations, trained with REINFORCE | Pham et al., "Efficient Neural Architecture Search via Parameter Sharing", ICML 2018. [[arXiv:1802.03268]](https://arxiv.org/abs/1802.03268) |

---

## Reinforcement Learning for Diffusion LMs

| Method | Description | Reference |
|--------|-------------|-----------|
| `DiffuGRPO` | Group Relative Policy Optimization adapted for dLLMs; group-relative advantage, no importance weights | d1 — Scaling Reasoning in Diffusion LLMs. [[GitHub]](https://github.com/dllm-reasoning/d1) |
| `DiFFPO` | PPO with importance-ratio clipping + joint sampler training (`StepBudgetController` predicts adaptive step budget per prompt) | Zhao, Liang, Tang, Yao, Kallus. "Training Diffusion LLMs to Reason Fast and Furious via Reinforcement Learning", 2024. [[arXiv:2510.02212]](https://arxiv.org/abs/2510.02212) |

---

## Backbone Models

| Model | Type | Reference |
|-------|------|-----------|
| LLaDA | LLaMA-3 based masked diffusion (8B) | Nie et al., "Large Language Diffusion Models", 2025. [[arXiv:2502.09992]](https://arxiv.org/abs/2502.09992) |
| MDLM | Absorbing-state continuous-time diffusion | Sahoo et al., "Simple and Effective Masked Diffusion Language Models", 2024. [[arXiv:2406.07524]](https://arxiv.org/abs/2406.07524) |
| SEDD | Score-entropy discrete diffusion | Lou et al., "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution", 2024. [[arXiv:2310.16834]](https://arxiv.org/abs/2310.16834) |
| D3PM | Discrete-time structured transitions | Austin et al., "Structured Denoising Diffusion Models in Discrete State-Spaces", NeurIPS 2021. [[arXiv:2107.03006]](https://arxiv.org/abs/2107.03006) |
