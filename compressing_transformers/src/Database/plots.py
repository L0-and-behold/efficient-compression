import csv
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

exp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../input/")

# ── Configure here ────────────────────────────────────────────────────────────
FILES = [
    (exp_path+"6159990784.csv", "Big dataset"),
    (exp_path+"1231945728.csv", "Medium dataset"),
    (exp_path+"299958272.csv", "Small dataset"),
]
MOVING_AVG = True   # True = plot smoothed, False = plot raw
WINDOW     = 100     # moving average window size
OUTPUT     = "loss_curve_comparison.svg"
# ─────────────────────────────────────────────────────────────────────────────

def load(csv_file):
    iterations, losses, epoch_iterations = [], [], []
    offset, prev = 0, None
    with open(csv_file) as f:
        for row in csv.reader(f):
            if len(row) < 2 or not row[0].strip():
                continue
            it = int(row[0])
            if prev is not None and it < prev:
                offset += prev
                epoch_iterations.append(it + offset)
            iterations.append(it + offset)
            losses.append(float(row[1]))
            prev = it
    return iterations, losses, epoch_iterations

def moving_avg(values, window):
    """Compute a centered moving average over a 1-D array."""
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result

def plot():
    """Plot loss curves for all runs in a single figure."""
    plt.figure(figsize=(12, 6))

    for csv_file, label in FILES:
        iterations, losses, epoch_iterations = load(csv_file)
        if MOVING_AVG:
            losses = moving_avg(losses, WINDOW)
            lw, ms = 1.5, 0
        else:
            lw, ms = 0.8, 4
        line, = plt.plot(iterations, losses, linewidth=lw, label=label)
        if ms:
            plt.scatter(iterations, losses, s=ms, color=line.get_color(), zorder=3)
        
        plt.vlines(epoch_iterations[0], min(losses), max(losses)/2, colors=line.get_color())

    plt.xlabel("Iteration")
    plt.ylabel("Loss" + (f" (moving avg, w={WINDOW})" if MOVING_AVG else ""))
    
    # Get current number of ticks and suggest double
    ax = plt.gca()
    current_ticks = len(ax.get_xticks())
    ax.xaxis.set_major_locator(MaxNLocator(nbins=current_ticks * 2))
    
    plt.title("Loss Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT, format="svg")
    print(f"Saved to {OUTPUT}")

if __name__ == "__main__":
    plot()