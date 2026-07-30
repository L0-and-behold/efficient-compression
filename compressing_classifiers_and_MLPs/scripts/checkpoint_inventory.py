#!/usr/bin/env python3
"""Scan FINISHED checkpoints, match to logs and runs.csv, produce a summary table."""

import os, glob, re, csv

EXP_DATA = "/projects/latentSimplicity/experiment_data"
REPORTS = "/projects/latentSimplicity/code/efficient-compression/compressing_classifiers_and_MLPs/reports"

# 1. Find all FINISHED checkpoints
checkpoints = sorted(glob.glob(os.path.join(EXP_DATA, "*/checkpoints/FINISHED_*.jld2")))

# Extract animal names from checkpoint filenames
ckpt_info = []
for path in checkpoints:
    fname = os.path.basename(path)                     # FINISHED_wild-lynx.jld2
    animal = fname.replace("FINISHED_", "").replace(".jld2", "")  # wild-lynx
    exp_dir = path.split("/checkpoints/")[0]
    exp_name = os.path.basename(exp_dir)
    ckpt_info.append({"path": path, "animal": animal, "exp_dir": exp_dir, "exp_name": exp_name})

# 2. Search all log files for "Checkpoint <animal> → FINISHED" pattern
log_files = sorted(glob.glob(os.path.join(REPORTS, "*.log")))

# Build index: animal -> list of log files
animal_to_logs = {}
for lf in log_files:
    with open(lf, 'r', errors='replace') as f:
        content = f.read()
    # Find all "Checkpoint <name> → FINISHED" or "Checkpoint <name> -> FINISHED"
    matches = re.findall(r'Checkpoint (\S+)\s*(?:→|->)\s*FINISHED', content)
    for m in matches:
        animal_to_logs.setdefault(m, []).append(lf)

# 3. For each checkpoint, find the matching log and extract info
results = []
for ci in ckpt_info:
    animal = ci["animal"]
    matched_logs = animal_to_logs.get(animal, [])
    
    row = {
        "exp_name": ci["exp_name"],
        "animal": animal,
        "log_file": "",
        "procedure": "",
        "run_id": "",
        "pre_prune_val": "",
        "post_prune_val": "",
        "compression": "",
        "cr_x": "",
        "seed": "",
        "alpha": "",
        "log_matches_exp": "",
    }
    
    if not matched_logs:
        row["log_file"] = "NOT FOUND"
        results.append(row)
        continue
    
    # Use the last (most recent) matching log
    log_path = matched_logs[-1]
    log_basename = os.path.basename(log_path)
    row["log_file"] = log_basename
    
    # Check if log name matches experiment name
    # e.g. log "alpha-sweep_278633_9.log" should match exp "alpha-sweep-v1_9-13"
    # Just store both for manual verification
    row["log_matches_exp"] = "check"
    
    with open(log_path, 'r', errors='replace') as f:
        lines = f.readlines()
    
    # Find the LAST line with "Checkpoint <animal> → FINISHED" (last = after resume)
    finished_line_idx = None
    for i, line in enumerate(lines):
        if re.search(rf'Checkpoint {re.escape(animal)}\s*(?:→|->)\s*FINISHED', line):
            finished_line_idx = i
    
    if finished_line_idx is not None:
        # Look in a small window before the FINISHED marker
        # Pattern: "CR report @ ep N: compression=X%, L0=post/pre, post-prune val acc=Y%"
        # immediately followed by "Checkpoint <animal> → FINISHED"
        window_start = max(0, finished_line_idx - 8)
        window_end = min(len(lines), finished_line_idx + 1)
        window = lines[window_start:window_end]
        window_text = "".join(window)
        
        # CR report line: "CR report @ ep 90: compression=85.2%, L0=3770639/25557032, post-prune val acc=68.86%"
        cr_match = re.search(r'CR report.*?compression=([\d.]+)%.*?L0=(\d+)/(\d+).*?post-prune val acc=([\d.]+)%', window_text)
        if cr_match:
            comp_pct = float(cr_match.group(1))
            post_l0 = int(cr_match.group(2))
            pre_l0 = int(cr_match.group(3))
            post_prune_acc = float(cr_match.group(4))
            row["compression"] = f"{comp_pct:.1f}%"
            row["cr_x"] = f"{pre_l0/post_l0:.2f}x" if post_l0 > 0 else "inf"
            row["post_prune_val"] = f"{post_prune_acc:.2f}%"
        
        # Pre-pruning val accuracy: the "Validation accuracy" line before "Pruning..."
        val_match = re.search(r'Validation accuracy:\s*([\d.]+)\s*%', window_text)
        if val_match:
            row["pre_prune_val"] = f"{float(val_match.group(1)):.2f}%"
    
    # Extract procedure and run_id from "Start training for run-..." line
    for line in lines:
        m = re.search(r"Start training for (run-\w+) with architecture '(\w+)'.*optimization procedure '(\w+)'", line)
        if m:
            row["run_id"] = m.group(1)
            row["procedure"] = m.group(3)
            break
    
    # Also try "Starting run with values:" pattern
    if not row["procedure"]:
        for line in lines:
            m = re.search(r'Starting run with values:.*optimization_procedure:\s*(\w+)', line)
            if m:
                row["procedure"] = m.group(1)
            m2 = re.search(r'Start training for (run-\w+)', line)
            if m2:
                row["run_id"] = m2.group(1)
            if row["procedure"] and row["run_id"]:
                break
    
    # Also try "Resuming from" pattern if run_id not found yet
    if not row["run_id"]:
        for line in lines:
            m = re.search(r'for (run-\w+)', line)
            if m:
                row["run_id"] = m.group(1)
                break
    
    results.append(row)

# 4. Match run_ids to runs.csv for seed and alpha
for row in results:
    if not row["run_id"]:
        continue
    
    exp_dir = os.path.join(EXP_DATA, row["exp_name"])
    runs_csv = os.path.join(exp_dir, "runs.csv")
    if not os.path.exists(runs_csv):
        continue
    
    with open(runs_csv, 'r') as f:
        reader = csv.DictReader(f)
        for csvrow in reader:
            if csvrow.get("run_id") == row["run_id"]:
                row["seed"] = csvrow.get("seed", "")
                row["alpha"] = csvrow.get("α", csvrow.get("alpha", ""))
                # Also get procedure from CSV if not found in log
                if not row["procedure"]:
                    row["procedure"] = csvrow.get("optimization_procedure", "")
                break

# 5. Print table
print(f"{'Experiment':<45} {'Animal':<18} {'Procedure':<18} {'Run ID':<16} {'α':<12} {'Pre-prune':<12} {'Post-prune':<12} {'CR':<10} {'CR(x)':<8} {'Seed':<6} {'Log File':<45}")
print("-" * 220)
for r in results:
    print(f"{r['exp_name']:<45} {r['animal']:<18} {r['procedure']:<18} {r['run_id']:<16} {r['alpha']:<12} {r['pre_prune_val']:<12} {r['post_prune_val']:<12} {r['compression']:<10} {r['cr_x']:<8} {r['seed']:<6} {r['log_file']:<45}")

# 6. Filter summary: post-prune acc >= 65% and CR(x) >= 2x
print("\n\n=== FILTERED: post-prune acc >= 65% AND CR(x) >= 2x ===\n")
filtered = []
for r in results:
    # Parse post_prune_val
    if not r["post_prune_val"]:
        continue
    acc = float(r["post_prune_val"].replace("%", ""))
    if acc < 65.0:
        continue
    # Parse cr_x
    if not r["cr_x"] or r["cr_x"] == "inf":
        continue
    cr = float(r["cr_x"].replace("x", ""))
    if cr < 2.0:
        continue
    filtered.append((r, acc, cr))

print(f"{'Procedure':<18} {'Animal':<18} {'α':<12} {'Post-prune':<12} {'CR(x)':<8} {'Experiment':<45}")
print("-" * 120)
for r, acc, cr in sorted(filtered, key=lambda x: (x[0]["procedure"], -x[2])):
    print(f"{r['procedure']:<18} {r['animal']:<18} {r['alpha']:<12} {r['post_prune_val']:<12} {r['cr_x']:<8} {r['exp_name']:<45}")

# Count per method
from collections import Counter
method_counts = Counter(r["procedure"] for r, _, _ in filtered)
print(f"\nTotal filtered checkpoints: {len(filtered)}")
print("Per method:")
for method, count in sorted(method_counts.items()):
    print(f"  {method}: {count}")
