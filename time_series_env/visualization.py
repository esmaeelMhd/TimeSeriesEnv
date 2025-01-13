import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from typing import Any, Tuple, List

# Set configurations
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

HALF_TEXT_SIZE = 10

def initialize_plots(n_act: int, n_targets: int, figsize: Tuple[float, float]) -> Tuple[plt.Figure, List[plt.Axes], List[plt.Axes], plt.Axes]:
    """Initialize the figure and axes for plotting."""
    fig, axes = plt.subplots(n_act + n_targets + 1, 1, figsize=figsize, 
                             constrained_layout=True, sharex=True)
    target_axes = axes[:n_targets]
    action_axes = axes[n_targets:-1]
    reward_ax = axes[-1]
    return fig, target_axes, action_axes, reward_ax

def update_plot(ax: plt.Axes, data: np.ndarray, label: str, color_idx: int = 0, 
                ylabel: str = '', xlabel: str = '', additional_data: np.ndarray = None, 
                additional_label: str = '', colors: List[str] = None) -> None:
    """Update plot for given axes with NumPy arrays."""
    # Set local configurations for this plot using rc_context
    with plt.rc_context({
        'font.family': ['Times New Roman', 'serif'],  # Font family with fallback to serif
        'font.size': HALF_TEXT_SIZE,              # General font size
        'axes.linewidth': 0.25,                    # Line width for plot axes
        'axes.xmargin': 0.02,                     # Margin on x-axis
        'axes.ymargin': 0.04,                     # Margin on y-axis
        'axes.labelsize': HALF_TEXT_SIZE,         # Font size for x and y axis labels
        'axes.titlesize': HALF_TEXT_SIZE,         # Font size for axis titles
        'xtick.labelsize': HALF_TEXT_SIZE,        # Font size for x-tick labels
        'ytick.labelsize': HALF_TEXT_SIZE,        # Font size for y-tick labels
        'legend.fontsize': HALF_TEXT_SIZE,        # Font size for legends
        'lines.markersize': 1.5                     # Marker size for the plot
    }):
        ax.clear()
        ax.plot(data, 'x-', color=colors[color_idx], label=label)
        if additional_data is not None:
            ax.plot(additional_data, '--', color='black', label=additional_label)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_ylabel(ylabel, labelpad=10)
    
        if xlabel:
            ax.set_xlabel(xlabel)
            
        handles, labels = ax.get_legend_handles_labels()
        if any(label != '_nolegend_' for label in labels):
            ax.legend()

def plot_live(env, figure: plt.Figure, target_axes: List[plt.Axes], action_axes: List[plt.Axes], 
              reward_ax: plt.Axes, colors: List[str]) -> plt.Figure:
    """Plots the live environment data using Matplotlib with dynamic updates."""      
    # Directly use the NumPy arrays without conversion
    actions = env.actions[:env.round]
    targets = env.targets[:env.round]
    rewards = env.rewards[:env.round]
    actual_actions = env.actual_actions[:env.round]
    actual_targets = env.actual_targets[:env.round]
    actual_rewards_ep = env.actual_rewards_ep[:env.round]

    for i, ax in enumerate(target_axes):
        update_plot(ax, targets[:, i], label='Simulation',
                    color_idx=1, ylabel=f'{env.target_names[i]}',
                    additional_data=actual_targets[:, i], 
                    additional_label='Data', colors=colors)

    for i, ax in enumerate(action_axes):
        update_plot(ax, actions[:, i], label='Agent',
                    color_idx=2, ylabel=f'{env.act_names[i]}',
                    additional_data=actual_actions[:, i], 
                    additional_label='Data', colors=colors)

    update_plot(reward_ax, rewards, 'Agent', color_idx=3, ylabel='Reward', xlabel='Steps',
                additional_data=actual_rewards_ep, additional_label='Data', colors=colors)

    return figure


def plot_not_live(env, figure: plt.Figure, target_axes: List[plt.Axes], action_axes: List[plt.Axes], 
                  reward_ax: plt.Axes, colors: List[str]) -> plt.Figure:
    """Render the final state of the environment for the episode."""
    # Use NumPy arrays directly
    actions = env.actions
    targets = env.targets
    rewards = env.rewards
    actual_actions = env.actual_actions
    actual_targets = env.actual_targets
    actual_rewards_ep = env.actual_rewards_ep
    
    for i, ax in enumerate(target_axes):
        update_plot(ax, targets[:, i], label='Simulation',
                    color_idx=1, ylabel=f'{env.target_names[i]}',
                    additional_data=actual_targets[:, i], 
                    additional_label='Data', colors=colors)

    for i, ax in enumerate(action_axes):
        update_plot(ax, actions[:, i], label='Agent',
                    color_idx=2, ylabel=f'{env.act_names[i]}',
                    additional_data=actual_actions[:, i], 
                    additional_label='Data', colors=colors)

    update_plot(reward_ax, rewards, 'Agent', color_idx=3, ylabel='Reward', xlabel='Steps',
                additional_data=actual_rewards_ep, additional_label='Data', colors=colors)
    
    return figure

def finalize_and_save_plot(fig: Any, results_folder: str, agent_name: str, fig_name: str) -> None:
    """Finalize plots and save to file, supporting both matplotlib and plotly figures."""
    results_path = os.path.join(results_folder, agent_name)
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)
    
    if isinstance(fig, plt.Figure):
        # Handling for matplotlib figures
        for ax in fig.get_axes():
            ax.grid(visible=True, which='major', color='gray', linewidth=0.5)
            ax.grid(visible=True, which='minor', color='gray', linewidth=0.5)
        
        # Save the matplotlib figure
        fig.savefig(os.path.join(results_path, f'{fig_name}.pdf'), dpi=300)

    else:
        raise ValueError("Unsupported figure type. The figure must be either a matplotlib or plotly figure.")
