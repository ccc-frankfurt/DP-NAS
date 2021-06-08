import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import seaborn as sns

import lib.Models.state_space_parameters as state_space_parameters


def visualize_rolling_reward(csv_path=None, replay_dict=None, num_arch=1, save_path='./runs/'):
    # set variables
    if csv_path is not None:
        df_replay_dict = pd.read_csv(csv_path, delimiter=',')
    else:
        df_replay_dict = replay_dict

    reward_col = 'reward'

    # do not count cases where the reward is 0
    df_replay_dict = df_replay_dict[df_replay_dict[reward_col] > 0.0]

    rolling_mean_window = 20
    net_index_limit = None
    ignore_first_arcs = 0

    fontsize = 24 * 1.25
    titlesize = fontsize + 4
    legendsize = fontsize - 10

    # get epsilon parameters
    epsilon_schedule = state_space_parameters.epsilon_schedule
    epsilon_starts = np.cumsum(np.array(epsilon_schedule)[:, 1])

    if net_index_limit is None:
        reward_replay_dict = df_replay_dict[reward_col].tolist()
    else:
        reward_replay_dict = df_replay_dict[reward_col][:int(net_index_limit)].tolist()

    rolling_mean_reward_replay_dict = list()

    if rolling_mean_window is None:
        for i in range(0, len(reward_replay_dict)):
            rolling_mean_reward_replay_dict.append(np.mean(np.array(reward_replay_dict[0:i+1])))
    else:
        for i in range(int(rolling_mean_window), len(reward_replay_dict)):
            rolling_mean_reward_replay_dict.append(
                np.mean(np.array(reward_replay_dict[i-int(rolling_mean_window):i+1])))

    if len(rolling_mean_reward_replay_dict) == 0:
        return

    # figure
    sns.set("paper", font_scale=2.5)

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)

    eps_text_offset = 2
    eps_alpha = 0.8
    cmin = max(0, min(reward_replay_dict))
    cmax = max(reward_replay_dict)

    if rolling_mean_window is None:
        text_y = min(rolling_mean_reward_replay_dict[int(ignore_first_arcs):]) + 0.01
        sc = ax.scatter(np.arange(int(ignore_first_arcs), len(rolling_mean_reward_replay_dict)),
                        rolling_mean_reward_replay_dict[int(ignore_first_arcs):],
                        cmap='viridis', marker='o', c=np.array(reward_replay_dict)[int(ignore_first_arcs):],
                        vmin=cmin, vmax=cmax, s=400.0, alpha=0.8)
    else:
        text_y = min(rolling_mean_reward_replay_dict) + 0.01
        upper_text_y = cmax - 0.075
        sc = ax.scatter(
            np.arange(int(rolling_mean_window), len(rolling_mean_reward_replay_dict) + int(rolling_mean_window), 1),
            rolling_mean_reward_replay_dict,
            cmap='viridis', marker='o', c=np.array(reward_replay_dict)[int(rolling_mean_window):],
            vmin=cmin, vmax=cmax, s=400.0, alpha=0.8)

    cbar = plt.colorbar(sc, shrink=1.0, aspect=15, pad=0.02)
    cbar.set_label(label='Individual architecture reward', size=fontsize, labelpad=10)
    cbar.ax.tick_params(labelsize=20)

    if rolling_mean_window is None:
        ax.plot(np.arange(int(ignore_first_arcs), len(rolling_mean_reward_replay_dict)),
                rolling_mean_reward_replay_dict[int(ignore_first_arcs):], '--k', linewidth=2.5, alpha=0.8)
    else:
        ax.plot(np.arange(int(rolling_mean_window), len(rolling_mean_reward_replay_dict) + int(rolling_mean_window), 1),
                rolling_mean_reward_replay_dict, '-x', linewidth=2.5, alpha=0.8)

    for i in range(0, len(epsilon_starts) - 1, 2):
        ax.axvline(x=epsilon_starts[i], linewidth=1, color='k', linestyle='dashed', alpha=eps_alpha)
        text(epsilon_starts[i] + eps_text_offset, text_y, "$\epsilon$ = " + str(epsilon_schedule[i][0]), rotation=90,
             verticalalignment='center', fontsize=legendsize, alpha=eps_alpha)

    text(10, upper_text_y, "Exploration phase", verticalalignment='center', fontsize=legendsize, alpha=eps_alpha)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(fontsize)

    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize - 4)
        item.set_fontweight('bold')

    plt.ylabel('Moving average reward', fontsize=fontsize, fontweight='bold')
    plt.xlabel('Architecture index', fontsize=fontsize, fontweight='bold')
    plt.title('DP-NAS', fontsize=titlesize, y=1.05, fontweight='bold')

    plt.tight_layout()
    print('SAVING: plot moving reward average')
    plt.savefig(save_path + '/MetaQNN_moving_avg_epoch_' + str(num_arch) + '.pdf')
    plt.close(fig)
