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
from dqntrainer import DQNTrainer

# %%
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(f'Using device {device}')

rng = random.Random()

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
query_id_map = {idx: query for idx, query in enumerate(queries)}

raw_page_sets: list[set[tuple[str, int]]] = []
for query in queries:
    raw_page_sets.append(set(query))

page_sets, page_to_id = encode_page_sets(raw_page_sets)

TARGET_COLS = 1000

table2n_blocks: dict[str, int] = {}
for group in queries:
    for table, block in group:
        table2n_blocks[table] = max(table2n_blocks.get(table, 0), block + 1)

tables = table2n_blocks.keys()

# %%
N_EPISODES = 25
dqn_trainer = DQNTrainer(
    table_to_n_blocks=table2n_blocks,
    queryid_to_tablepages=query_id_map,
    queryid_to_tablepageids=page_sets,
    tablepage_to_tablepageid=page_to_id,
    target_cols=10_000,
    rng=rng,
)
dqn, log = dqn_trainer.train(
    update_steps=2,
    epsilon_schedule=np.geomspace(0.9, 0.01, num=N_EPISODES),
    gamma=0.9,
    tau=0.1,
    history_size=100,
    mini_batch_size=None,
    cache_capacity_pages=2000,
    device=device,
)

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
ax.set(xlabel="Log gtep (global)", ylabel="Log loss", title="Log loss per step")

fig.tight_layout()
plt.show()

# %%
dqn.cpu()
input_size = dqn.net[0].in_features
dummy_input = torch.zeros(1, input_size)
torch.onnx.export(
    dqn,
    (dummy_input,),
    'dqn.onnx',
    input_names=['state'],
    output_names=['q_value'],
    dynamic_axes={'state': {0: 'batch_size'}}  # allows variable batch size
)
