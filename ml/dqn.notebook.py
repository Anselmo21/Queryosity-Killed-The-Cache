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
from dataclasses import dataclass
import random
import sys
from typing import Dict, List, Optional, Sequence as Seq

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
from src.simulator.cache_simulator import LRUCache
from src.simulator.access_profile import AccessProfile
from src.simulator.access_profile import build_access_profiles_from_db
from src.utilities.configurations import (
    BASELINE_SEED,
    PG_HOST,
    PG_PASSWORD,
    PG_PORT,
    PG_SCHEMA,
    PG_STATEMENT_TIMEOUT_MS,
    PG_USER,
)
from src.postgres.connection import close_connection, create_connection
from src.simulator.dqn_simulator import build_state
from src.utilities.constants import DB_DEFAULTS, PROJECT_ROOT, WORKLOAD_DIRS
from src.utilities.workload import load_queries

# %%
WORKLOAD = 'tpch'
db_name = DB_DEFAULTS[WORKLOAD]
conn = create_connection(
    db_name=db_name,
    user=PG_USER,
    password=PG_PASSWORD,
    host=PG_HOST,
    port=PG_PORT,
    schema=PG_SCHEMA,
    statement_timeout_ms=PG_STATEMENT_TIMEOUT_MS,
)
queries = load_queries(WORKLOAD)
profiles = build_access_profiles_from_db(queries, conn, analyze=False)
close_connection(conn)


# %%
@dataclass
class Transition:
    state: Tensor # (n,)
    reward: Tensor # (1,)
    next_states: Tensor # (b, n)

@dataclass
class TransitionInfo:
    transitions: List[Transition]
    all_tables: List[str]
    max_pages: Dict[str, int]

def collect_transitions(
    profiles: List[AccessProfile],
    cache_capacity_pages: int,
    num_episodes: int = 500,
    device: Optional[torch.device] = None
) -> TransitionInfo:
    all_tables = sorted(list(set(t for p in profiles for t in p.table_pages)))
    max_pages = {
        t: max(p.table_pages.get(t, 0) for p in profiles)
        for t in all_tables
    }
    transitions: List[Transition] = []

    for _ in range(num_episodes):
        cache = LRUCache(cache_capacity_pages)
        queue = list(range(len(profiles)))
        random.shuffle(queue)

        while queue:
            # build state for each candidate
            candidates = [
                (idx, build_state(cache, profiles[idx], all_tables, max_pages))
                for idx in queue
            ]

            # random action during collection
            chosen_pos = random.randint(0, len(queue) - 1)
            chosen_idx, chosen_state = candidates[chosen_pos]
            queue.pop(chosen_pos)

            # page-weighted reward matching your simulate_schedule logic
            hits = 0
            total = 0
            for table, pages in profiles[chosen_idx].table_pages.items():
                total += pages
                if table in cache._entries:
                    hits += pages
            reward = hits / total if total > 0 else 0.0

            # update cache using your existing LRUCache
            for table, pages in profiles[chosen_idx].table_pages.items():
                cache.access(table, pages)

            next_candidates = [
                build_state(cache, profiles[idx], all_tables, max_pages)
                for idx in queue
            ]

            transitions.append(Transition(
                state = torch.tensor(chosen_state, dtype=torch.float32, device=device),
                reward = torch.tensor(reward, dtype=torch.float32, device=device),
                next_states = torch.tensor(next_candidates, dtype=torch.float32, device=device),
            ))

    return TransitionInfo(
        transitions=transitions,
        all_tables=all_tables,
        max_pages=max_pages,
    )


# %%
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


# %%
@dataclass
class TrainingHistory:
    epoch: List[int]
    loss: List[float]
    mean_q: List[float]
    mean_reward: List[float]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'epoch': self.epoch,
            'loss': self.loss,
            'mean_q': self.mean_q,
            'mean_reward': self.mean_reward,
        })

def train(
    dqn: nn.Module,
    transitions: List[Transition],
    epochs: int = 50,
    gamma: float = 0.95,
    lr: float = 1e-3,
) -> TrainingHistory:
    optimizer = torch.optim.Adam(dqn.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history: TrainingHistory = TrainingHistory(
        epoch = [],
        loss = [],
        mean_q = [],
        mean_reward = [],
    )

    iteration_counter = 0
    pbar = tqdm(total=epochs)

    for epoch in range(epochs):
        random.shuffle(transitions)
        total_loss = 0.0
        current_loss: float = 0.0
        previous_loss: Optional[float] = None
        total_q = 0.0
        total_reward = 0.0

        for t in transitions:
            state = t.state
            reward = t.reward

            if len(t.next_states.shape) == 1:
                target = reward # Terminal state
            else:
                with torch.no_grad():
                    next_qs: Tensor = dqn(t.next_states)
                    max_next_q = next_qs.max()
                target = reward + gamma * max_next_q

            predicted: Tensor = dqn(state)
            predicted = predicted.squeeze()
            loss: Tensor = loss_fn(predicted, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            previous_loss = current_loss
            current_loss = loss.item()
            total_loss += current_loss
            total_q += predicted.item()
            total_reward += reward.item()

        n = len(transitions)
        iteration_counter += 1
        history.epoch.append(epoch)
        history.loss.append(total_loss / n)
        history.mean_q.append(total_q / n)
        history.mean_reward.append(total_reward / n)

        pbar.update(1)
        pbar.set_description(f'Epoch {epoch+1}')
        pbar.set_postfix(
            avg_loss=f'{total_loss / len(transitions):.4e}',
            loss=f'{current_loss:.4e}',
            previous_loss=f'{previous_loss:.4e}',
        )
    return history


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %%
transition_info = collect_transitions(profiles, 100, device=device)
transitions = transition_info.transitions
all_tables = transition_info.all_tables
max_pages = transition_info.max_pages

input_size = len(all_tables)*2
model = DQN(input_size=input_size)
model = model.to(device)

history = train(
    dqn=model,
    transitions=transitions,
    epochs=10,
)

# %%
fig: Figure; axes: Seq[Seq[Axes]]
n_rows = 2; n_cols = 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4))

df = history.to_dataframe()

i_to_series = ['loss', 'mean_q', 'mean_reward']

from itertools import product

for i, j in product(range(n_rows), range(n_cols)):
    series_name = i_to_series[j]
    x = df['epoch']
    if i == 0:
        y = df[series_name]
        title = series_name
    elif i == 1:
        y = np.log10(df[series_name])
        title = f'log10({series_name})'
    axes[i][j].plot(x, y)
    axes[i][j].set_title(title)
    axes[i][j].set_xlabel('Epoch')
    axes[i][j].set_ylabel(title)

plt.tight_layout()
plt.show()

# %%
model.cpu()
dummy_input = torch.zeros(1, input_size)
torch.onnx.export(
    model,
    (dummy_input,),
    'dqn.onnx',
    input_names=['state'],
    output_names=['q_value'],
    dynamic_axes={'state': {0: 'batch_size'}}  # allows variable batch size
)

# %%
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('dqn.onnx')

def dqn_fitness(schedule, profiles, cache_capacity_pages, session, all_tables, max_pages):
    cache = LRUCache(cache_capacity_pages)
    total_q = 0.0

    for idx in schedule:
        state = np.array(
            build_state(cache, profiles[idx], all_tables, max_pages),
            dtype=np.float32
        ).reshape(1, -1)

        q_value = session.run(['q_value'], {'state': state})[0]
        total_q += q_value.item()

        for table, pages in profiles[idx].table_pages.items():
            cache.access(table, pages)

    return total_q
