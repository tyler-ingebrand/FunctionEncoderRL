import os
from typing import List

import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt

# smooth data without changing size of list
def smooth(y):
    from scipy.signal import savgol_filter
    yhat = savgol_filter(y, 21, 3)  # window size 51, polynomial order 3
    return yhat
def parse_tensorboard(path, scalars:List[str]):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}

logdir = "./log/tag/new_players"

run_data = {}
step = None
for run_dir in reversed(sorted(os.listdir(logdir))):

    # fetch run type by reading from file.
    alg_type = None
    if os.path.isfile(os.path.join(logdir, run_dir, "function_encoder_ppo.txt")):
        alg_type = "FE PPO"
    elif os.path.isfile(os.path.join(logdir, run_dir, "oracle_ppo.txt")):
        alg_type = "OHE PPO"
    elif os.path.isfile(os.path.join(logdir, run_dir, "normal_ppo.txt")):
        alg_type = "PPO"
    elif os.path.isfile(os.path.join(logdir, run_dir, "transformer_ppo.txt")):
        alg_type = "Transformer PPO"
    else:
        print("Skipping run dir", run_dir)
        continue
    if alg_type not in run_data:
        run_data[alg_type] = {}

    # fetch seed by reading from file
    seed = None
    if os.path.isfile(os.path.join(logdir, run_dir, "seed_0.txt")):
        seed = 0
    elif os.path.isfile(os.path.join(logdir, run_dir, "seed_1.txt")):
        seed = 1
    elif os.path.isfile(os.path.join(logdir, run_dir, "seed_2.txt")):
        seed = 2
    elif os.path.isfile(os.path.join(logdir, run_dir, "seed_3.txt")):
        seed = 3
    elif os.path.isfile(os.path.join(logdir, run_dir, "seed_4.txt")):
        seed = 4
    elif os.path.isfile(os.path.join(logdir, run_dir, "seed_5.txt")):
        seed = 5

    if seed is None:
        print("Skipping run dir", run_dir)
        continue
    if seed in run_data[alg_type]:
        print("Skipping repeated alg", run_dir)
        continue
        # raise ValueError("seed already exists")

    # read data, convert to numpy array
    data = parse_tensorboard(os.path.join(logdir, run_dir), ["test/reward"])
    if step is None or len(data["test/reward"]['step'].to_numpy()) > len(step):
        step = data["test/reward"]['step'].to_numpy()
    data = data["test/reward"]['value'].to_numpy()

    # add to dict
    run_data[alg_type][seed] = data


# now plot
# for each alg type in the run data, compute median and 1st and 3rd quartile
fig, ax = plt.subplots()
for alg_type in run_data:
    alg_data = [seed_data for seed_data in run_data[alg_type].values()]
    length = alg_data[1].shape[0]
    alg_data = [a for a in alg_data if a.shape[0] == length]

    alg_data = np.stack(alg_data, axis=0)
    print(alg_type, alg_data.shape)
    quartiles = np.quantile(alg_data, [0.0, 0.5, 1.0], axis=0)

    # plot with shaded area
    color = "blue" if alg_type == "FE PPO" else "red" if alg_type == "PPO" else "green" if alg_type == "Transformer PPO" else "orange" if alg_type == "OHE PPO" else "black"
    ax.plot(step, quartiles[1], label=alg_type, color=color)
    ax.fill_between(
        step,
        quartiles[0],
        quartiles[2],
        alpha=0.2,
        color=color,
        lw=0,
    )

ax.set_xlabel("Steps")
ax.set_ylabel("Reward")
ax.legend()
plt.title("Learning Curve for New Players")
plt.savefig("./log/tag/new_players/graph.png", dpi=400)

# save quartiles to csv to plot in latex later. Use pandas to do one at a time.
df = pd.DataFrame()
df["step"] = step
for alg_type in run_data:
    alg_data = [seed_data for seed_data in run_data[alg_type].values()]
    length = alg_data[1].shape[0]
    alg_data = [a for a in alg_data if a.shape[0] == length]

    alg_data = np.stack(alg_data, axis=0)
    print(alg_type, alg_data.shape)
    quartiles = np.quantile(alg_data, [0.0, 0.5, 1.00], axis=0)
    # df[alg_type] = smooth(quartiles[1])
    # df[alg_type + "_low"] = smooth(quartiles[0])
    # df[alg_type + "_high"] = smooth(quartiles[2])
    df[alg_type] = quartiles[1]
    df[alg_type + "_low"] = quartiles[0]
    df[alg_type + "_high"] = quartiles[2]
df.to_csv("./log/tag/new_players/quartiles.csv", index=False)