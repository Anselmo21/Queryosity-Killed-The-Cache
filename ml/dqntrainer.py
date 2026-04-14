import dataclasses
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


type Query = int
type State = tuple[PageLRUCache, list[Query]]
type Action = Query
"""Query ID"""


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


@dataclasses.dataclass
class TrainingLog:

    # Per-episode aggregates
    episode_mean_loss: list[float] = dataclasses.field(default_factory=list)
    episode_std_loss: list[float]  = dataclasses.field(default_factory=list)
    episode_min_loss: list[float]  = dataclasses.field(default_factory=list)
    episode_max_loss: list[float]  = dataclasses.field(default_factory=list)
    episode_reward: list[float]    = dataclasses.field(default_factory=list)
    episode_steps: list[int]       = dataclasses.field(default_factory=list)
    episode_epsilon: list[float]   = dataclasses.field(default_factory=list)

    # Per-step (global across all episodes)
    step_loss: list[float] = dataclasses.field(default_factory=list)


class DQNTrainer:

    tables: list[str]
    table2n_blocks: dict[str, int]
    query_id_map: dict[int, list[tuple[str, int]]]
    query_id_to_bitmap_vector: dict[int, dict[str, Int[Tensor, 'N']]]
    target_cols: int
    rng: random.Random

    def __init__(
        self,
        table2n_blocks: dict[str, int],
        query_id_map: dict[int, list[tuple[str, int]]],
        target_cols: int,
        rng: random.Random,
    ):
        self.table2n_blocks = table2n_blocks
        self.tables = list(table2n_blocks.keys())
        self.query_id_map = query_id_map
        self.query_id_to_bitmap_vector = dict()
        self.target_cols = target_cols
        self.rng = rng

    def downsize(
        self,
        rows: dict[str, Int[Tensor, 'N']],
        target_cols: int,
        device: torch.device,
    ) -> Float[Tensor, 'n_rows target_cols']:
        result = []
        for table in self.tables:
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
        self,
        query: list[tuple[str, int]],
        cache: PageLRUCache,
    ) -> tuple[float, PageLRUCache]:
        hits = 0
        total = 0
        for table, block in query:
            if cache.access((table, block)):
                hits += 1
            total += 1
        hit_rate = hits / total
        return hit_rate, deepcopy(cache)


    def cache_to_bitmap_vectors(
        self,
        cache: PageLRUCache,
        device: torch.device,
    ) -> dict[str, Int[Tensor, 'N']]:
        # Group blocks by table in one pass
        table2blocks: dict[str, list[int]] = defaultdict(list)
        for table, block in cache._entries.keys():
            table2blocks[table].append(block)

        rows = dict()
        for table, size in self.table2n_blocks.items():
            t = torch.zeros(size, dtype=torch.int, device=device)
            if table in table2blocks:
                t[table2blocks[table]] = 1
            rows[table] = t
        return rows


    def query_to_bitmap_vectors(
        self,
        query_id: int,
        device: torch.device,
    ) -> dict[str, Int[Tensor, 'N']]:
        if query_id not in self.query_id_to_bitmap_vector:
            query = self.query_id_map[query_id]
            rows = dict()
            for table, size in self.table2n_blocks.items():
                t = torch.zeros(size, dtype=torch.int, device=device)
                for tbl, block in query:
                    if tbl == table:
                        t[block] = 1
                rows[table] = t
            self.query_id_to_bitmap_vector[query_id] = rows
        return self.query_id_to_bitmap_vector[query_id]

    def q_value(
        self,
        network: DQN,
        cache: PageLRUCache,
        query_id: int,
        device: torch.device
    ) -> Tensor:
        query_vectors = self.query_to_bitmap_vectors(query_id, device)
        downsized_query_vector = self.downsize(
            query_vectors, self.target_cols, device
        )
        downsized_query_vector = downsized_query_vector.flatten()

        bitmap_vectors = self.cache_to_bitmap_vectors(cache, device)
        downsized_bitmap_vector = self.downsize(
            bitmap_vectors, self.target_cols, device
        )
        downsized_bitmap_vector = downsized_bitmap_vector.flatten()

        in_vector = torch.cat(
            [downsized_bitmap_vector, downsized_query_vector]
        )
        q = network(in_vector)
        return q

    def train(
        self,
        update_steps: int,
        epsilon_schedule: Collection[float] | Tensor,
        gamma: float,
        tau: float,
        history_size: int,
        mini_batch_size: int | None,
        cache_capacity_pages: int,
        device: torch.device,
    ) -> tuple[DQN, TrainingLog]:
        assert mini_batch_size == None or history_size >= mini_batch_size, \
            'History size must be >= mini batch size'
        assert 0 <= tau <= 1

        # =====================================================================
        # Step 1: Initialize network.
        # =====================================================================
        queries = list(self.query_id_map.keys())
        input_dimension = 2 * len(self.tables) * self.target_cols
        Q = DQN(input_dimension)
        Q.to(device)
        target = DQN(input_dimension)
        target.to(device)
        history: list[tuple[State, Action, float, State, bool]] = []
        steps = 0

        optimizer = torch.optim.Adam(Q.parameters())

        episodes = len(epsilon_schedule)
        pbar = tqdm(total=episodes)
        log = TrainingLog()

        for episode, epsilon in enumerate(epsilon_schedule):
            # =================================================================
            # Step 2: Define initial state.
            # =================================================================
            cache = PageLRUCache(cache_capacity_pages)
            remaining_queries = deepcopy(queries)
            state_counter = 0
            episode_reward = 0.0
            episode_losses = []  # <-- per-step losses within this episode

            while True:
                state_counter += 1
                pbar.set_description(
                    f'Episode {episode}/{episodes-1}, '
                    f'state: {state_counter}, '
                    f'epsilon : {epsilon:.4f}'
                )
                pbar.update(0)
                state: State = (deepcopy(cache), deepcopy(remaining_queries))

                # =============================================================
                # Step 3: Get all possible actions that can be taken in this
                #         state and either:
                #         - choose one randomly, OR
                #         - pick the action with the highest Q-value.
                # =============================================================
                if self.rng.random() < epsilon:
                    idx = random.randrange(len(remaining_queries))
                else:
                    with torch.no_grad():
                        idx = max(
                            range(len(remaining_queries)),
                            key=lambda i: self.q_value(
                                Q, cache, i, device
                            ).item()
                        )

                query_id = remaining_queries[idx]
                query = self.query_id_map[query_id]
                next_remaining_queries = deepcopy(remaining_queries)
                next_remaining_queries.pop(idx)

                # =============================================================
                # Step 4: Execute action and observe reward.
                # =============================================================

                reward, next_cache = self.execute(query, cache)
                episode_reward += reward

                done = len(next_remaining_queries) == 0
                next_state: State = (next_cache, next_remaining_queries)
                history.append((state, idx, reward, next_state, done))
                if len(history) > history_size:
                    history.pop(0)

                # =============================================================
                # Step 5: Sample a batch from the history of actions.
                # =============================================================
                if mini_batch_size is None:
                    mini_batch = history
                else:
                    if len(history) < mini_batch_size:
                        cache = next_cache
                        remaining_queries = next_remaining_queries
                        continue
                    mini_batch = random.sample(history, k=mini_batch_size)

                # =============================================================
                # Step 6: Compute the temporal difference targets (yi) between
                #         the target network's rewards and the actual reward.
                #         Loss is calculated as MSE between the temporal
                #         difference targets and the predicted values.
                # =============================================================
                losses = torch.zeros((len(mini_batch),), device=device)
                for i, (s1, a, r, s2, d) in enumerate(mini_batch):
                    with torch.no_grad():
                        if d:
                            y = r
                        else:
                            c2, rq = s2
                            max_q = max(
                                self.q_value(
                                    target, c2, x, device,
                                ).item() for x in rq
                            )
                            y = r + gamma * max_q
                    c1, _ = s1
                    predicted = self.q_value(Q, c1, a, device)
                    losses[i] = (y - predicted)**2

                loss = losses.mean()
                episode_losses.append(loss.item())
                log.step_loss.append(loss.item())
                # =============================================================
                # Step 7: Backpropagation
                # =============================================================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # =============================================================
                # Step 8: Transfer weights from the Q network to the target
                #         network.
                # =============================================================
                steps += 1
                if steps == update_steps:
                    steps = 0
                    for sp, tp in zip(Q.parameters(), target.parameters()):
                        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

                # =============================================================
                # Step 9: Transition to next state
                # =============================================================
                cache = next_cache
                remaining_queries = next_remaining_queries

                # =============================================================
                # Step 10: If we are in a terminal state, then stop; otherwise,
                #          go back to step 3.
                # =============================================================
                if done:
                    break

            pbar.set_description(f'Episode {episode}/{episodes-1}')
            pbar.set_postfix(
                loss=f'{loss.item():.4e}',
            )
            pbar.update(1)

            arr = np.array(episode_losses)
            log.episode_mean_loss.append(arr.mean())
            log.episode_std_loss.append(arr.std())
            log.episode_min_loss.append(arr.min())
            log.episode_max_loss.append(arr.max())
            log.episode_reward.append(episode_reward)
            log.episode_steps.append(state_counter)
            log.episode_epsilon.append(float(epsilon))

        return Q, log
