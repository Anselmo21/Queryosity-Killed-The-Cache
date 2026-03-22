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
from typing import Sequence as Seq

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import torch
import torch.nn as nn
from torch import Tensor

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
def build_state(
    cache: LRUCache,
    query_profile: AccessProfile,
    all_tables: list[str],
    max_pages: dict[str, int]
) -> list[float]:
    buffer_vec = [
        cache._entries.get(t, 0) / max_pages[t] for t in all_tables
    ]
    query_vec = [
        query_profile.table_pages.get(t, 0) / max_pages[t]
        for t in all_tables
    ]
    return buffer_vec + query_vec


# %%
def collect_transitions(profiles: list[AccessProfile], cache_capacity_pages: int, num_episodes=500):
    all_tables = list(set(t for p in profiles for t in p.table_pages))
    max_pages = {
        t: max(p.table_pages.get(t, 0) for p in profiles)
        for t in all_tables
    }
    transitions = []

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

            transitions.append({
                'state': chosen_state,
                'reward': reward,
                'next_states': next_candidates,
            })

    return transitions, all_tables, max_pages


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
def train(
    dqn: nn.Module,
    transitions,
    epochs: int = 50,
    gamma: float = 0.95,
    lr: float = 1e-3,
    device: torch.device | None = None
):
    optimizer = torch.optim.Adam(dqn.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {
        'epoch': [],
        'loss': [],
        'mean_q': [],
        'mean_reward': [],
    }

    for epoch in range(epochs):
        random.shuffle(transitions)
        total_loss = 0.0
        total_q = 0.0
        total_reward = 0.0

        for t in transitions:
            state = torch.tensor(t['state'], dtype=torch.float32).to(device)
            reward = t['reward']

            # Bellman target
            if t['next_states']:
                with torch.no_grad():
                    next_qs = [dqn(torch.tensor(s, dtype=torch.float32).to(device)) for s in t['next_states']]
                    max_next_q = max(q.item() for q in next_qs)
                target = reward + gamma * max_next_q
            else:
                target = reward  # terminal state

            predicted: Tensor = dqn(state)
            predicted = predicted.squeeze()
            target_tensor = torch.tensor(target, dtype=torch.float32).to(device)
            loss: Tensor = loss_fn(predicted, target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_q += predicted.item()
            total_reward += reward

        n = len(transitions)
        history['epoch'].append(epoch + 1)
        history['loss'].append(total_loss / n)
        history['mean_q'].append(total_q / n)
        history['mean_reward'].append(total_reward / n)

        print(f"Epoch {epoch+1}: loss = {total_loss / len(transitions):.4f}")
    return history


# %%
transitions, all_tables, max_pages = collect_transitions(profiles, 100)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DQN(input_size=len(all_tables)*2)
model = model.to(device)

history = train(
    dqn=model,
    transitions=transitions,
    epochs=10,
    device=device
)

# %%
fig: Figure
axes: Seq[Axes]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].plot(history['epoch'], history['loss'])
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')

axes[1].plot(history['epoch'], history['mean_q'])
axes[1].set_title('Mean Q-Value')
axes[1].set_xlabel('Epoch')

axes[2].plot(history['epoch'], history['mean_reward'])
axes[2].set_title('Mean Reward')
axes[2].set_xlabel('Epoch')

plt.tight_layout()
plt.show()
