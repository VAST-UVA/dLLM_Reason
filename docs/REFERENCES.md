# References

All baselines, benchmarks, and search methods used in dLLM-Reason, with paper references.

---

## Unmasking Strategies (Baselines)

| Strategy | Description | Reference |
|----------|-------------|-----------|
| `confidence` | Unmask highest-confidence (argmax prob) positions first; LLaDA default | Nie et al., "LLaDA: Large Language Diffusion with mAsking", 2025. [[arXiv:2502.09992]](https://arxiv.org/abs/2502.09992) |
| `random` | Uniform random unmasking; standard dLLM baseline | Sahoo et al., "Simple and Effective Masked Diffusion Language Models (MDLM)", 2024. [[arXiv:2406.07524]](https://arxiv.org/abs/2406.07524) |
| `entropy` | Unmask lowest-entropy (most certain by full distribution) first | Chang et al., "MaskGIT: Masked Generative Image Transformer", CVPR 2022. [[arXiv:2202.04200]](https://arxiv.org/abs/2202.04200) |
| `semi_ar` | Semi-autoregressive: block-by-block L-to-R, confidence within each block | Savinov et al., "Step-unrolled Denoising Autoencoders for Text Generation (SUNDAE)", ICML 2022. [[arXiv:2112.06749]](https://arxiv.org/abs/2112.06749) |
| `linear` | Left-to-right sequential (autoregressive simulation) | Standard autoregressive baseline; used as reference in all dLLM papers |
| `cot` | Chain-of-Thought DAG: force reasoning segments before answer | Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", NeurIPS 2022. [[arXiv:2201.11903]](https://arxiv.org/abs/2201.11903) |
| `skeleton` | Skeleton-then-Detail: generate structural tokens first, then fill details | Ning et al., "Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation", 2023. [[arXiv:2307.15337]](https://arxiv.org/abs/2307.15337) |
| `bidirectional` | Unmask from both ends toward center | Yang et al., "XLNet: Generalized Autoregressive Pretraining for Language Understanding", NeurIPS 2019. [[arXiv:1906.08237]](https://arxiv.org/abs/1906.08237) |
| `answer_first` | Generate answer region first, then fill reasoning | Zhao et al., "Verify-and-Edit: A Knowledge-Enhanced Chain-of-Thought Framework", 2023. [[arXiv:2305.03268]](https://arxiv.org/abs/2305.03268) |

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
| ProntoQA | Logical reasoning | Accuracy | Saparov & He, "Language Models Are Greedy Reasoners: A Systematic Formal Analysis of Chain-of-Thought", ICLR 2023. [[arXiv:2210.01240]](https://arxiv.org/abs/2210.01240) |

---

## DAG Search Methods

| Method | Description | Reference |
|--------|-------------|-----------|
| Greedy | Iterative edge add/remove to maximize fitness | Chickering, "Optimal Structure Identification with Greedy Search", JMLR 2002. [[Paper]](https://www.jmlr.org/papers/v3/chickering02b.html) |
| Evolutionary | Population-based: tournament selection, crossover, mutation | Larranga et al., "Structure Learning of Bayesian Networks by Genetic Algorithms", 1996. General framework; see also GADAG. [[arXiv:2101.10769]](https://arxiv.org/abs/2101.10769) |
| RL Policy | REINFORCE with Transformer policy to construct DAGs edge-by-edge | Zhu et al., "Causal Discovery with Reinforcement Learning", NeurIPS 2020. [[arXiv:1906.04477]](https://arxiv.org/abs/1906.04477) |
| Differentiable (NOTEARS) | Continuous relaxation with augmented Lagrangian acyclicity constraint | Zheng et al., "DAGs with NO TEARS: Continuous Optimization for Structure Learning", NeurIPS 2018. [[arXiv:1803.01422]](https://arxiv.org/abs/1803.01422) |

---

## Backbone Models

| Model | Type | Reference |
|-------|------|-----------|
| LLaDA | LLaMA-3 based masked diffusion (8B) | Nie et al., "LLaDA: Large Language Diffusion with mAsking", 2025. [[arXiv:2502.09992]](https://arxiv.org/abs/2502.09992) |
| MDLM | Absorbing-state continuous-time diffusion | Sahoo et al., "Simple and Effective Masked Diffusion Language Models", 2024. [[arXiv:2406.07524]](https://arxiv.org/abs/2406.07524) |
| SEDD | Score-entropy discrete diffusion | Lou et al., "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution", 2024. [[arXiv:2310.16834]](https://arxiv.org/abs/2310.16834) |
| D3PM | Discrete-time structured transitions | Austin et al., "Structured Denoising Diffusion Models in Discrete State-Spaces", NeurIPS 2021. [[arXiv:2107.03006]](https://arxiv.org/abs/2107.03006) |
