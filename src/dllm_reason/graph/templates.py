"""Predefined DAG templates for reasoning tasks.

Each template defines a specific reasoning pattern by constructing a
TokenDAG that captures the dependency structure of that pattern.
These templates are used as:
1. Hand-designed baselines for comparison
2. Starting points for DAG search optimization
3. Interpretable reasoning strategies for analysis
"""

from __future__ import annotations

import torch

from dllm_reason.graph.dag import TokenDAG


def chain_of_thought_dag(
    seq_len: int,
    num_steps: int,
    prompt_len: int = 0,
    device: torch.device | str = "cpu",
) -> TokenDAG:
    """Chain-of-Thought DAG: sequential reasoning steps.

    Partitions the generation positions into `num_steps` segments.
    Each segment depends on ALL previous segments. Within a segment,
    positions are independent (can be unmasked in parallel).

    This mimics "solve step 1, then step 2, then step 3..."

    Example with seq_len=12, prompt_len=0, num_steps=3:
        Level 0: positions [0, 1, 2, 3]     (step 1)
        Level 1: positions [4, 5, 6, 7]     (step 2, depends on step 1)
        Level 2: positions [8, 9, 10, 11]   (step 3, depends on steps 1+2)
    """
    gen_len = seq_len - prompt_len
    positions_per_step = gen_len // num_steps
    remainder = gen_len % num_steps

    levels = []
    pos = prompt_len
    for step in range(num_steps):
        n = positions_per_step + (1 if step < remainder else 0)
        levels.append(list(range(pos, pos + n)))
        pos += n

    # If there's a prompt, make it level 0 and shift everything else
    if prompt_len > 0:
        prompt_level = list(range(prompt_len))
        levels = [prompt_level] + levels

    return TokenDAG.from_levels(levels, seq_len=seq_len, device=device)


def answer_first_dag(
    seq_len: int,
    answer_positions: list[int],
    reasoning_segments: int = 3,
    prompt_len: int = 0,
    device: torch.device | str = "cpu",
) -> TokenDAG:
    """Answer-First DAG: generate the answer, then fill in reasoning.

    The answer token positions are roots (unmasked first).
    Reasoning positions are generated afterward, conditioned on the answer.
    This tests whether knowing the answer helps generate justification.

    Args:
        answer_positions: indices of answer tokens (these are unmasked first)
        reasoning_segments: number of segments to split remaining positions into
    """
    answer_set = set(answer_positions)
    remaining = [i for i in range(prompt_len, seq_len) if i not in answer_set]

    # Answer positions are level 0 (or level 1 if prompt exists)
    levels = []
    if prompt_len > 0:
        levels.append(list(range(prompt_len)))
    levels.append(answer_positions)

    # Split remaining into segments
    seg_size = len(remaining) // reasoning_segments
    for seg in range(reasoning_segments):
        start = seg * seg_size
        end = start + seg_size if seg < reasoning_segments - 1 else len(remaining)
        levels.append(remaining[start:end])

    return TokenDAG.from_levels(levels, seq_len=seq_len, device=device)


def skeleton_then_detail_dag(
    seq_len: int,
    skeleton_positions: list[int],
    detail_positions: list[int] | None = None,
    prompt_len: int = 0,
    device: torch.device | str = "cpu",
) -> TokenDAG:
    """Skeleton-then-Detail DAG: structure first, then fill in.

    Level 0: Skeleton tokens (key structural tokens — operators, connectives,
             answer placeholder, etc.)
    Level 1: Detail tokens (operands, intermediate values)
    Level 2: Filler tokens (everything else — formatting, padding)

    This mimics "plan the structure, then fill in the details."
    """
    skeleton_set = set(skeleton_positions)
    detail_set = set(detail_positions or [])
    filler = [i for i in range(prompt_len, seq_len)
              if i not in skeleton_set and i not in detail_set]

    levels = []
    if prompt_len > 0:
        levels.append(list(range(prompt_len)))
    levels.append(skeleton_positions)
    if detail_positions:
        levels.append(detail_positions)
    if filler:
        levels.append(filler)

    return TokenDAG.from_levels(levels, seq_len=seq_len, device=device)


def bidirectional_dag(
    seq_len: int,
    num_segments: int = 4,
    prompt_len: int = 0,
    device: torch.device | str = "cpu",
) -> TokenDAG:
    """Bidirectional DAG: unmask from both ends toward the middle.

    This creates a structure where the first and last segments are
    unmasked first, then progressively toward the center.
    Useful for tasks where boundary conditions constrain the middle.
    """
    gen_positions = list(range(prompt_len, seq_len))
    n = len(gen_positions)

    # Create pairs from outside in
    levels = []
    if prompt_len > 0:
        levels.append(list(range(prompt_len)))

    left = 0
    right = n - 1
    while left <= right:
        seg_size = max(1, (right - left + 1) // max(1, (num_segments - len(levels))))
        level = []
        # Take from left
        for _ in range(min(seg_size, right - left + 1)):
            level.append(gen_positions[left])
            left += 1
        # Take from right
        for _ in range(min(seg_size, right - left + 1)):
            level.append(gen_positions[right])
            right -= 1
        if level:
            levels.append(level)

    return TokenDAG.from_levels(levels, seq_len=seq_len, device=device)


def random_dag(
    seq_len: int,
    density: float = 0.1,
    prompt_len: int = 0,
    device: torch.device | str = "cpu",
) -> TokenDAG:
    """Random DAG with controlled edge density.

    Generates a random DAG by sampling edges and ensuring acyclicity
    via topological ordering (only allow edges from lower to higher index
    in a random permutation).

    Args:
        density: fraction of possible edges to include (0.0 to 1.0)
    """
    gen_len = seq_len - prompt_len

    # Random permutation defines the topological order
    perm = torch.randperm(gen_len)
    order = torch.zeros(gen_len, dtype=torch.long)
    order[perm] = torch.arange(gen_len)

    adj = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    # Add prompt -> generation edges
    if prompt_len > 0:
        for p in range(prompt_len):
            for g in range(prompt_len, seq_len):
                adj[p, g] = True

    # Sample random edges respecting topological order
    for i in range(gen_len):
        for j in range(gen_len):
            if order[i] < order[j] and torch.rand(1).item() < density:
                adj[prompt_len + i, prompt_len + j] = True

    return TokenDAG(adj)


def interleaved_dag(
    seq_len: int,
    num_groups: int = 2,
    prompt_len: int = 0,
    device: torch.device | str = "cpu",
) -> TokenDAG:
    """Interleaved DAG: alternating groups of positions.

    Positions are split into `num_groups` interleaved groups.
    Group 0 contains positions 0, num_groups, 2*num_groups, ...
    Group 1 contains positions 1, num_groups+1, 2*num_groups+1, ...
    Each group depends on the previous group.

    Useful for tasks with interleaved reasoning patterns
    (e.g., alternating between calculation and narration).
    """
    gen_positions = list(range(prompt_len, seq_len))
    groups = [[] for _ in range(num_groups)]
    for idx, pos in enumerate(gen_positions):
        groups[idx % num_groups].append(pos)

    levels = []
    if prompt_len > 0:
        levels.append(list(range(prompt_len)))
    levels.extend(groups)

    return TokenDAG.from_levels(levels, seq_len=seq_len, device=device)
