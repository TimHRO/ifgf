import subprocess
import numpy as np
import matplotlib.pyplot as plt

def run_cq_error(eps, exe_path="./test_CQWeights"):
    try:
        result = subprocess.run([exe_path, str(eps)], capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error for eps={eps}: {e}")
        return None

def main():
    eps_values = np.logspace(-10, -2, num=20)   # 1e-8 to 1e-2
    errors = []

    print("eps\t\terror")
    for eps in eps_values:
        err = run_cq_error(eps)
        if err is not None:
            errors.append(err)
            print(f"{eps:.2e}\t{err:.6e}")
        else:
            errors.append(np.nan)

    # Remove NaNs
    eps_clean = eps_values[~np.isnan(errors)]
    errors_clean = np.array(errors)[~np.isnan(errors)]

    # Plot
    plt.figure(figsize=(8,6))
    plt.loglog(eps_clean, errors_clean, 'o-', label='Measured error')
    plt.loglog(eps_clean, 1e-0*np.sqrt(eps_clean), '--', label=r'$\sqrt{\varepsilon}$')
    plt.loglog(eps_clean, (eps_clean), '--', label=r'${\varepsilon}$')
    plt.xlabel(r'IFGF relative error $\varepsilon$')
    plt.ylabel('Maximum absolute error in convolution weights')
    plt.title('Convolution quadrature error vs. IFGF accuracy')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('cq_error_convergence.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()