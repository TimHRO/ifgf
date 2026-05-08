import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt

# Path to your executable
EXECUTABLE = "./test/test_ifgf"

# Fixed parameters:
# ./test/test_ifgf 10.0 10.0 8 1 10000 100 1e-16
#
# We vary:
# - first parameter: from 0 to 100  -> separate plots
# - last parameter: x-axis

second_param = 50
third_param = 10
fourth_param = 1
fifth_param = 1000
sixth_param = 10

# Example values for first parameter
first_param_values = [50]

# Example values for last parameter (x-axis)
last_param_values = np.logspace(-16, -5, 30)


def run_program(first_param, last_param):
    cmd = [
        EXECUTABLE,
        str(first_param),
        str(second_param),
        str(third_param),
        str(fourth_param),
        str(fifth_param),
        str(sixth_param),
        str(last_param),
    ]

    print("Running:", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    output = result.stdout

    # Regex extraction
    patterns = {
        "to_far": r"^to far:\s+(\d+)$",
        "far": r"^far:\s+(\d+)$",
        "near": r"^near:\s+(\d+)$",
        "e_abs": r"^summary:\s+e abs=\s+([0-9eE\.\-\+]+)$"
    }

    values = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.MULTILINE)
        if match:
            values[key] = float(match.group(1))
        else:
            values[key] = np.nan
            print(f"Warning: Could not find {key}")

    return values


for first_param in first_param_values:
    x_vals = []
    to_far_vals = []
    far_vals = []
    near_vals = []
    e_abs_vals = []

    for last_param in last_param_values:
        data = run_program(first_param, last_param)

        x_vals.append(last_param)
        to_far_vals.append(data["to_far"])
        far_vals.append(data["far"])
        near_vals.append(data["near"])
        e_abs_vals.append(data["e_abs"])

    print(far_vals)
    print(to_far_vals)

    # Create plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left Y-axis -> counters
    ax1.set_xscale("log")
    #ax1.set_yscale("log")
    ax1.grid(True)
    ax1.plot(x_vals, to_far_vals, marker="o", label="to far")
    ax1.plot(x_vals, far_vals, marker="s", label="far")
    ax1.plot(x_vals, near_vals, marker="^", label="near")

    ax1.set_xlabel("Cut Off Threshold")
    ax1.set_ylabel("Interaction Counter")
    ax1.tick_params(axis="y")

    # Right Y-axis -> e_abs
    ax2 = ax1.twinx()
    ax2.semilogy(x_vals, e_abs_vals, marker="d", linestyle="--", label="e abs")
    ax2.set_ylabel("absolute error")
    ax2.tick_params(axis="y")

    # Combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.title(f"k= {first_param} + i{second_param}")
    plt.tight_layout()
    plt.show()