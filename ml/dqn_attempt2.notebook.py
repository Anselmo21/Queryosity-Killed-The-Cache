# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: ml_cuda12.2
#     language: python
#     name: python3
# ---

# %%
import random
import sys
from typing import Collection

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

sys.path.insert(0, "..")
from src.simulator.cache_simulator import PageLRUCache
from jaxtyping import Float, Int
from collections import defaultdict
from copy import deepcopy

class DQN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # single Q-value output
        )

    def forward(self, x):
        return self.net(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)


# %%
dfs: list[pd.DataFrame] = []
for i in range(1, 23):
    df = pd.read_csv(f'../page_access/tpch/q{i}.csv')
    df['query'] = i
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

queries: list[list[tuple[str, int]]] = [
    list(zip(g['table'], g['block']))
    for _, g in df.groupby('query')
]

TARGET_COLS = 1000

table2n_blocks: dict[str, int] = {}
for group in queries:
    for table, block in group:
        table2n_blocks[table] = max(table2n_blocks.get(table, 0), block + 1)

tables = table2n_blocks.keys()


# %%
def downsize(
    rows: dict[str, Int[Tensor, 'N']],
    target_cols: int,
    device: torch.device,
) -> Float[Tensor, 'n_rows target_cols']:
    result = []
    for table in tables:
        if table in rows:
            row = rows[table]
        else:
            row = torch.zeros((target_cols,), device=device)
        N = row.shape[0]
        if N < target_cols:
            # Pad with zeros to target length
            padded = torch.zeros(target_cols, device=device)
            padded[:N] = row
            result.append(padded)
        else:
            bin_size = N // target_cols
            # I'm truncating instead of calculating the sum bounds in the
            # SmartQueue paper because they don't make sense to me :P
            truncated = row[:bin_size * target_cols]
            truncated = truncated.float()
            # I'm averaging instead of summing & dividing by floor(|Fi|/|Di|)
            # because I think that's what the SmartQueue paper was trying to
            # achieve anyway :P
            downsized = truncated.view(target_cols, bin_size).mean(dim=1)
            result.append(downsized)
    return torch.stack(result)


def execute(
    query: list[tuple[str, int]],
    cache: PageLRUCache
) -> tuple[float, PageLRUCache]:
    cache = deepcopy(cache)
    hits = 0
    total = 0
    for table, block in query:
        if cache.access((table, block)):
            hits += 1
        total += 1
    hit_rate = hits / total
    return hit_rate, cache


def cache_to_bitmap_vectors(cache: PageLRUCache, device: torch.device) -> dict[str, Int[Tensor, 'N']]:
    rows = dict()
    for table, size in table2n_blocks.items():
        t = torch.zeros(size, dtype=torch.int, device=device)
        for tbl, block in cache._entries.keys():
            if tbl == table:
                t[block] = 1
        rows[table] = t
    return rows

def query_to_bitmap_vectors(query: list[tuple[str, int]], device: torch.device) -> dict[str, Int[Tensor, 'N']]:
    rows = dict()
    for table, size in table2n_blocks.items():
        t = torch.zeros(size, dtype=torch.int, device=device)
        for tbl, block in query:
            if tbl == table:
                t[block] = 1
        rows[table] = t
    return rows

def q_value(network: DQN, cache: PageLRUCache, query: list[tuple[str, int]], device: torch.device) -> Tensor:
    query_vectors = query_to_bitmap_vectors(query, device)
    downsized_query_vector = downsize(query_vectors, TARGET_COLS, device)
    downsized_query_vector = downsized_query_vector.flatten()
    bitmap_vectors = cache_to_bitmap_vectors(cache, device)
    downsized_bitmap_vector = downsize(bitmap_vectors, TARGET_COLS, device).flatten()
    downsized_bitmap_vector = downsized_bitmap_vector.flatten()
    in_vector = torch.cat([downsized_bitmap_vector, downsized_query_vector])
    q = network(in_vector)
    return q


# %%
type Query = list[tuple[str, int]]
type State = tuple[PageLRUCache, list[Query]]
type Action = Query

def q_network(
    queries: list[list[tuple[str, int]]],
    rng: random.Random,
    update_steps: int,
    epsilon: float,
    gamma_schedule: Collection[float] | Tensor,
    tau: float,
    history_size: int,
    mini_batch_size: int,
    cache_capacity_pages: int,
    device: torch.device,
) -> DQN:
    assert history_size >= mini_batch_size, 'History size must be >= mini batch size'
    assert 0 <= tau <= 1

    # =========================================================================
    # Step 1: Initialize network.
    # =========================================================================
    Q = DQN(2 * len(tables) * TARGET_COLS)
    Q.to(device)
    target_network = DQN(16)
    target_network.to(device)
    history: list[tuple[State, Action, float, State, bool]] = []
    steps = 0

    optimizer = torch.optim.Adam(Q.parameters())

    episodes = len(gamma_schedule)
    pbar = tqdm(total=episodes)

    for episode, gamma in enumerate(gamma_schedule):
        # =====================================================================
        # Step 2: Define initial state.
        # =====================================================================
        cache = PageLRUCache(cache_capacity_pages)
        remaining_queries = deepcopy(queries)
        state_counter = 0

        while True:
            state_counter += 1
            pbar.set_description(f'Episode {episode}/{episodes-1}, state: {state_counter}')
            pbar.update(0)
            state: State = (deepcopy(cache), deepcopy(remaining_queries))
            # =================================================================
            # Step 3: Get all possible actions that can be taken in this state
            #         and either:
            #         - choose one randomly, OR
            #         - pick the action with the highest Q-value.
            # =================================================================
            if rng.random() < epsilon:
                idx = random.randrange(len(remaining_queries))
            else:
                with torch.no_grad():
                    idx = max(
                        range(len(remaining_queries)),
                        key=lambda i: q_value(
                            Q,
                            cache,
                            remaining_queries[i],
                            device,
                        ).item()
                    )
            query = remaining_queries[idx]
            next_remaining_queries = deepcopy(remaining_queries)
            next_remaining_queries.pop(idx)

            # =================================================================
            # Step 4: Execute action and observe reward.
            # =================================================================
            reward, next_cache = execute(query, cache)
            done = len(next_remaining_queries) == 0
            next_state: State = (deepcopy(next_cache), deepcopy(next_remaining_queries))
            history.append((state, query, reward, next_state, done))
            if len(history) > history_size:
                history.pop(0)

            # =================================================================
            # Step 5: Sample a batch from the history of actions.
            # =================================================================
            if len(history) < mini_batch_size:
                cache = next_cache
                remaining_queries = next_remaining_queries
                continue
            mini_batch = random.sample(history, k=mini_batch_size)

            # =================================================================
            # Step 6: Compute the temporal difference targets (yi) between the
            #         target network's rewards and the actual reward. Loss is
            #         calculated as MSE between the temporal difference targets
            #         and the predicted values.
            # =================================================================
            losses = torch.zeros((mini_batch_size,), device=device)
            for i, (s1, a, r, s2, d) in enumerate(mini_batch):
                with torch.no_grad():
                    if d:
                        y = r
                    else:
                        c2, rq = s2
                        max_q = max(
                            q_value(
                                target_network,
                                c2,
                                x,
                                device,
                            ).item() for x in rq
                        )
                        y = r + gamma * max_q
                c1, _ = s1
                predicted = q_value(Q, c1, a, device)
                losses[i] = (y - predicted)**2
            loss = losses.sum()
            # =================================================================
            # Step 7: Backpropagation
            # =================================================================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # =================================================================
            # Step 8: Transfer weights from the Q network to the target
            #         network.
            # =================================================================
            steps += 1
            if steps == update_steps:
                steps = 0
                for sp, tp in zip(Q.parameters(), target_network.parameters()):
                    tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

            # =================================================================
            # Step 9: Transition to next state
            # =================================================================
            cache = next_cache
            remaining_queries = next_remaining_queries

            # =================================================================
            # Step 10: If we are in a terminal state, then stop; otherwise, go
            #         back to step 3.
            # =================================================================
            if done:
                break

        pbar.set_description(f'Episode {episode}/{episodes-1}')
        pbar.set_postfix(
            loss=f'{loss.item():.4e}',
        )
        pbar.update(1)

    return Q


# %%
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(f'Using device {device}')

rng = random.Random()

# %%
q_network(
    queries=queries,
    rng=rng,
    update_steps=10,
    epsilon=0.5,
    gamma_schedule=torch.arange(1, 0, -0.1),
    tau=0.1,
    history_size=10000,
    mini_batch_size=100,
    cache_capacity_pages=200,
    device=device,
)

# %%
