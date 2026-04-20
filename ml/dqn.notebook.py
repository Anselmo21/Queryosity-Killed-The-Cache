# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: ml_cuda12.2
#     language: python
#     name: python3
# ---

# %%
import random
import sys
import typing
from typing import Sequence as Seq

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import torch
from torch import Tensor

sys.path.insert(0, "..")
from src.simulator.cache_simulator import encode_page_sets
from src.simulator.cache_simulator import PageClockSweepCache
from dqntrainer import DQN, DQNTrainer

# %%
print(
    'Before training the model, we need a name for it to avoid clashing with '
    'other models. I suggest something like "<benchmark>-<size>"'
)
NAME = input('Name: ')

# %%
print('Which benchmark are you using?')
BENCHMARK = input('Benchmark (tpch, tpcds, job): ')

# %%
print('We also need the capacity of the cache so was can simulate it.')
CACHE_CAPACITY_PAGES = int(input('Cache capacity in 8kb pages: '))

# %%
print(
    'And we need the number of training iterations (episodes) to train the '
    'DQN.'
)
N_EPISODES = int(input('Number of episodes: '))

# %%
TARGET_COLS = 1000 # From the paper

# %%
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(f'Using device {device}')

rng = random.Random()

# %%
dfs: list[pd.DataFrame] = []
for i in range(1, 23):
    df = pd.read_csv(f'../page_access/{BENCHMARK}/q{i}.csv')
    df['query'] = i
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

queries: list[list[tuple[str, int]]] = [
    list(zip(g['table'], g['block']))
    for _, g in df.groupby('query')
]
query_id_map = {idx: query for idx, query in enumerate(queries)}

raw_page_sets: list[set[tuple[str, int]]] = []
for query in queries:
    raw_page_sets.append(set(query))

page_sets, page_to_id = encode_page_sets(raw_page_sets)

table2n_blocks: dict[str, int] = {}
for group in queries:
    for table, block in group:
        table2n_blocks[table] = max(table2n_blocks.get(table, 0), block + 1)

tables = table2n_blocks.keys()

# %%
dqn_trainer = DQNTrainer(
    table_to_n_blocks=table2n_blocks,
    queryid_to_tablepages=query_id_map,
    queryid_to_tablepageids=page_sets,
    tablepage_to_tablepageid=page_to_id,
    target_cols=TARGET_COLS,
    rng=rng,
)
dqn, log = dqn_trainer.train(
    update_steps=2,
    epsilon_schedule=np.geomspace(0.9, 0.01, num=N_EPISODES),
    gamma=0.9,
    tau=0.1,
    history_size=100,
    mini_batch_size=None,
    cache_capacity_pages=CACHE_CAPACITY_PAGES,
    device=device,
)

# %%
torch.save(dqn.state_dict(), f'{NAME}.pt')

# %%
episodes = np.arange(len(log.episode_mean_loss))
mean = np.array(log.episode_mean_loss)
std = np.array(log.episode_std_loss)

fig, axes = plt.subplots(2, 2, figsize=(14, 5))
axes = typing.cast(Seq[Seq[Axes]], axes)

ax = axes[0][0]
ax.plot(episodes, mean, label="Mean loss")
ax.fill_between(episodes, mean - std, mean + std, alpha=0.25, label="±1 std", color='red')
ax.set(xlabel="Episode", ylabel="Loss", title="Loss per Episode")
ax.legend()

ax = axes[1][0]
ax.plot(episodes, mean, label="Mean log loss")
ax.fill_between(episodes, mean - std, mean + std, alpha=0.25, label="±1 std", color='red')
ax.set_yscale('log')
ax.set(xlabel="Episode", ylabel="Log loss", title="Log loss per Episode")
ax.legend()

ax = axes[0][1]
ax.plot(log.step_loss)
ax.set(xlabel="Step (global)", ylabel="Loss", title="Loss per Step")

ax = axes[1][1]
ax.plot(log.step_loss)
ax.set_yscale('log')
ax.set(xlabel="Log step (global)", ylabel="Log loss", title="Log loss per step")

fig.tight_layout()
plt.show()

# %%
input_dimension = 2 * len(tables) * TARGET_COLS
dqn = DQN(input_dimension)
dqn.load_state_dict(torch.load(f'{NAME}.pt'))
dqn.to(device)
dqn.eval()

# %%
cache = PageClockSweepCache(CACHE_CAPACITY_PAGES)

remaining_queries = list(query_id_map.keys())
schedule = []

hit_rate_history = []
best_q_value_history = []

# Scheduling loop
while len(remaining_queries) > 0:
    # Get the query with the highest Q-value
    query_id_to_q_value = {
        query_id: dqn_trainer.q_value(dqn, cache, query_id, device).item()
        for query_id in remaining_queries
    }
    best_query_id = max(
        query_id_to_q_value,
        key=lambda qid: query_id_to_q_value[qid]
    )

    # Add the best query to the schedule, remove it from the queue
    schedule.append(best_query_id)
    remaining_queries.remove(best_query_id)
    best_q_value_history.append(query_id_to_q_value[best_query_id])

    # Update cache by simulating the selected query
    hit_rate, next_cache = dqn_trainer.execute(best_query_id, cache)
    hit_rate_history.append(hit_rate)
    cache = next_cache

# %%
print('SCHEDULE')
print([f'q{query_id}' for query_id in schedule])
print(f"\n{'Query ID':<12} {'Q-Value':<12} {'Hit Rate':<12}")
print("-" * 36)
for query_id, q_value, hit_rate in zip(schedule, best_q_value_history, hit_rate_history):
    print(f'{query_id:<12} {q_value:<12.4f} {hit_rate:<12.4f}')
print("-" * 36)
print(f'Average hit rate: {sum(hit_rate_history)/len(hit_rate_history):.4f}')
