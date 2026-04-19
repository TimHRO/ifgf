import subprocess
import re
import csv
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Parameters
# ---------------------------

#x_values = [10**exp for exp in np.linspace(-1,2,20)]   # sweep variable
#real = [0.0, np.pi*0.1, np.pi*0.2, np.pi*0.45, np.pi*0.48, np.pi*0.5]   

H_values = [10**exp for exp in np.linspace(-1,2,20)]
kappa = 1
angle_values = [0.0, np.pi/8, np.pi/4, np.pi/2]
order = 3
segments = 1
sigma = 0.5
h = np.sqrt(3)/2
s_min = 1e-15
#s_min = h/r_max
print(f"r_max {h/s_min}, s_min{s_min}")

executable = "./test/test_fact_osz"
csv_filename = "./test/kappa_results.csv"

results = []

# ---------------------------
# Run experiment
# ---------------------------

for angle in angle_values:
    for H in H_values:
        c = kappa * np.sin(angle)
        r = kappa * np.cos(angle)

        cmd = [executable, str(H), str(r), str(c), str(segments), str(order), str(s_min), str(sigma), str(1e-14)]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        output = proc.stdout

        match = re.search(r"Absolute L2 Error: \s*([0-9eE\.\+\-]+)", output)

        if match:
            error = float(match.group(1))
            results.append((H, kappa, r, c, angle, order, error))
            print(f"H={H}, norm={kappa}, r={r}, c={c}, realp={angle}, order={order}, error={error}")
        else:
            print(f"Error not found for H={H}, k.c={r}")
            print(output)

# ---------------------------
# Save CSV
# ---------------------------

with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["H", "x", "r", "c", "realp", "order", "absolute_L2_error"])
    writer.writerows(results)

print(f"Saved results to {csv_filename}")

# ---------------------------
# Plot results
# ---------------------------

plt.figure()

for angle in angle_values:
    xs = [r[0]*r[1] for r in results if r[4] == angle]
    errs = [r[6] for r in results if r[4] == angle]

    plt.loglog(xs, errs, marker='o', label=f"arg(kappa) = {angle}")

plt.xlabel("|kappa| H")
plt.ylabel("Absolute L2 error")
plt.title("Interpolation Error vs kappa H")
plt.grid(True)
plt.legend()

plt.savefig("./test/kappa_error_plot.pdf")
plt.show()