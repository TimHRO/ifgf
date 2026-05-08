import subprocess
import numpy as np
import matplotlib.pyplot as plt
import re

def run_cq(order, N, lam, exe_path, params):
    cmd = [exe_path] + [str(p) for p in params] + [str(order), str(N), str(lam)]
    err_val = None
    s_vals = [] 
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if "Absolute Error:" in line:
                match = re.search(r"Absolute Error:\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", line)
                if match: err_val = float(match.group(1))
            
            # Captures every occurrence of "s: (x,y)" in the output
            if "s:" in line:
                s_match = re.search(r"s:\s*\(\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\)", line)
                if s_match: 
                    s_vals.append(complex(float(s_match.group(1)), float(s_match.group(2))))
        
        return err_val, s_vals
    except Exception as e:
        print(f"Error for N={N}: {e}")
        return None, []

def main():
    exe_path = "./test_CQIFGF"
    orders, N, dt = [7, 8, 9, 10, 11, 12], 200, 5
    fixed_params = [30, 1, 1000, dt] # [tarPerBox, base_n, num_points, dt]

    all_errors, all_s = [], []

    #for o in orders:
    #N_values = [20,80,100,120,140,160,200]
    o = 9
    #for N in N_values:
    for o in orders:
        lam = 10.0 ** (-8.0 / N)
        err, s_list = run_cq(o, N, lam, exe_path, fixed_params)
        all_errors.append(err if err is not None else np.nan)
        all_s.extend(s_list) # Collect all s points across all runs
        print(f"O={o}, N={N}, Captured {len(s_list)} s-values, error={err}")
        s_rel_max = max([z.imag/z.real for z in s_list])
        print(s_rel_max)
        print(s_list)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Error Plot
    alpha = 2
    y_exp = [1e5*np.exp(-2*o) for o in orders]
    #ifgf_error = [0.048, 0.0084, 0.000478, 7.469e-5,9.1148e-6,1.801e-6]
    ifgf_error = [0.0059,0.000997,5.26e-5,1.064e-5,1.75771e-6,3.96e-7]
    ax1.semilogy(orders, ifgf_error, label = "theoretical error: exp(-order)")
    ax1.semilogy(orders, all_errors, 'o-', label = "simulation error")
    ax1.legend()
    ax1.set_title('CQ Error vs Interpolation Order')
    ax1.grid(True, alpha=0.5)

    # Complex Plane Plot (Scattering all N s-values)
    if all_s:
        ax2.scatter([val.real for val in all_s], [val.imag for val in all_s], 
                    s=5, alpha=0.6, edgecolors='none', label=f'Total s points: {len(all_s)}')
    
    ax2.axhline(0, color='black', lw=1)
    ax2.axvline(0, color='black', lw=1)
    ax2.set_title(f'Distribution of s values ($N={N}$)')
    ax2.set_xlabel('Re(s)')
    ax2.set_ylabel('Im(s)')
    ax2.axis('equal')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig("./CQ_vs_O.pdf")
    plt.show()

if __name__ == "__main__":
    main()