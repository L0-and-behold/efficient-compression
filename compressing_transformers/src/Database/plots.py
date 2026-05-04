import csv
import os
import matplotlib.pyplot as plt

exp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../experiment-results/")

# ── Configure here ────────────────────────────────────────────────────────────
FILES = [
    (exp_path+"exp_name/artifacts/run-y102/train_loss.csv", "Name_of_run"),
    (exp_path+"exp_name/artifacts/run-v405/train_loss.csv", "Name_of_run_2"),
]
MOVING_AVG = True   # True = plot smoothed, False = plot raw
WINDOW     = 100     # moving average window size
OUTPUT     = "comparison.svg"
# ─────────────────────────────────────────────────────────────────────────────

def load(csv_file):
    iterations, losses = [], []
    offset, prev = 0, None
    with open(csv_file) as f:
        for row in csv.reader(f):
            if len(row) < 2 or not row[0].strip():
                continue
            it = int(row[0])
            if prev is not None and it < prev:
                offset += prev
            iterations.append(it + offset)
            losses.append(float(row[1]))
            prev = it
    return iterations, losses

def moving_avg(values, window):
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result

def plot():
    plt.figure(figsize=(12, 6))

    for csv_file, label in FILES:
        iterations, losses = load(csv_file)
        if MOVING_AVG:
            losses = moving_avg(losses, WINDOW)
            lw, ms = 1.5, 0
        else:
            lw, ms = 0.8, 4
        line, = plt.plot(iterations, losses, linewidth=lw, label=label)
        if ms:
            plt.scatter(iterations, losses, s=ms, color=line.get_color(), zorder=3)

    plt.xlabel("Iteration")
    plt.ylabel("Loss" + (f" (moving avg, w={WINDOW})" if MOVING_AVG else ""))
    plt.title("Loss Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT, format="svg")
    print(f"Saved to {OUTPUT}")

if __name__ == "__main__":
    plot()