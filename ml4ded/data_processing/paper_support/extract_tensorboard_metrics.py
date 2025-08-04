import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Customize these to your actual run directories
RUN_DIRS = {
    "baseline_no_registers": "/home/jordan/omscs/cs8903/SegDefDepth/data/paper_experiment_runs/baseline_no_registers",
    "baseline_w_registers": "/home/jordan/omscs/cs8903/SegDefDepth/data/paper_experiment_runs/baseline_w_registers",
    "unstable_temporal_results": "/home/jordan/omscs/cs8903/SegDefDepth/data/paper_experiment_runs/unstable_temporal_results",
    "stable_temporal_results": "/home/jordan/omscs/cs8903/SegDefDepth/data/paper_experiment_runs/stable_temporal_results",
}

TAGS = {
    "validation mIoU": "mIoU",
    "validation pixAcc": "Pixel Accuracy",
    "validation weighted mIoU": "Weighted mIoU",
}

def load_event_data(path, tags):
    ea = EventAccumulator(path)
    ea.Reload()
    scalars = {}
    for tag in tags:
        try:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            scalars[tag] = (steps, values)
        except KeyError:
            print(f"[Warning] Tag '{tag}' not found in {path}")
    return scalars

def annotate_best(ax, steps, values, label):
    if not steps or not values:
        return
    best_idx = max(range(len(values)), key=lambda i: values[i])
    best_step = steps[best_idx]
    best_val = values[best_idx]
    ax.annotate(f"{label}\n{best_val:.3f} @ {best_step}",
                xy=(best_step, best_val),
                xytext=(5, -10),
                textcoords="offset points",
                fontsize=8,
                arrowprops=dict(arrowstyle='->', lw=0.5))

def plot_group(runs_data, group_name):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)
    fig.suptitle(f"{group_name} Validation Metrics", fontsize=14)

    for i, (tag, label) in enumerate(TAGS.items()):
        ax = axs[i]
        for run_name, data in runs_data.items():
            if tag in data:
                steps, values = data[tag]
                ax.plot(steps, values, label=run_name)
                annotate_best(ax, steps, values, run_name)
        ax.set_ylabel(label)
        ax.grid(True)
        ax.legend()
    axs[-1].set_xlabel("Steps")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    all_runs_data = {}

    for run_name, log_dir in RUN_DIRS.items():
        if os.path.exists(log_dir):
            run_data = load_event_data(log_dir, TAGS.keys())
            all_runs_data[run_name] = run_data
        else:
            print(f"[Error] Directory not found: {log_dir}")

    # Group 1: Baselines
    baseline_runs = {k: v for k, v in all_runs_data.items() if "baseline" in k}
    plot_group(baseline_runs, "Baseline")

    # Group 2: Temporal
    temporal_runs = {k: v for k, v in all_runs_data.items() if "temporal" in k}
    plot_group(temporal_runs, "Temporal")

    # Group 3: Combined
    plot_group(all_runs_data, "Combined")