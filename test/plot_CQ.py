import subprocess
import re
import numpy as np

# Path to executable
EXECUTABLE = "./test_ifgf"

# Fixed parameters
fourth_param = 1
fifth_param = 1000
sixth_param = 10
cut = 0

# --------------------------------------------------
# Complex test points (theta values already mapped to z)
# --------------------------------------------------

theta_values = [(0.308169-0j), (0.313413+0.0850179j), (0.329064+0.168747j), (0.354887+0.249918j), (0.390488+0.327301j), (0.43533+0.399722j), (0.488731+0.466084j), (0.549882+0.52538j), (0.617857+0.576712j), (0.691624+0.619302j), (0.770067+0.652503j), (0.851994+0.675814j), (0.936166+0.68888j), (1.0213+0.691503j), (1.10612+0.683643j), (1.18933+0.665421j), (1.26967+0.637111j), (1.34592+0.599143j), (1.41692+0.552093j), (1.48161+0.496674j), (1.53899+0.433725j), (1.58821+0.364202j), (1.6285+0.289158j), (1.65927+0.20973j), (1.68005+0.127124j), (1.69052+0.0425897j), (1.69052-0.0425897j), (1.68005-0.127124j), (1.65927-0.20973j), (1.6285-0.289158j), (1.58821-0.364202j), (1.53899-0.433725j), (1.48161-0.496674j), (1.41692-0.552093j), (1.34592-0.599143j), (1.26967-0.637111j), (1.18933-0.665421j), (1.10612-0.683643j), (1.0213-0.691503j), (0.936166-0.68888j), (0.851994-0.675814j), (0.770067-0.652503j), (0.691624-0.619302j), (0.617857-0.576712j), (0.549882-0.52538j), (0.488731-0.466084j), (0.43533-0.399722j), (0.390488-0.327301j), (0.354887-0.249918j), (0.329064-0.168747j), (0.313413-0.0850179j)]

# --------------------------------------------------
# Parse executable output
# --------------------------------------------------

def run_program(real_part, imag_part, third_param):
    cmd = [
        EXECUTABLE,
        str(real_part),       # first parameter = real part
        str(imag_part),       # second parameter = imaginary part
        str(third_param),     # third parameter (sweeping 7 -> 12)
        str(fourth_param),
        str(fifth_param),
        str(sixth_param),
        str(cut),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    output = result.stdout

    # Match:
    # summary: e abs= <number>
    pattern = r"^\s*summary:\s+e abs=\s+([0-9eE.+\-]+)\s*$"

    match = re.search(pattern, output, re.MULTILINE)

    if match:
        return float(match.group(1))
    else:
        print(
            f"Warning: Could not find error for "
            f"third_param={third_param}, z={real_part}+i{imag_part}"
        )
        return 0


# --------------------------------------------------
# Sweep third_param = 7 ... 12
# Return maximum absolute error over all theta values
# --------------------------------------------------

for third_param in range(7, 13):
    errors = []

    for z in theta_values:
        real_part = z.real
        imag_part = z.imag

        err = run_program(real_part, imag_part, third_param)
        errors.append(err)

    max_abs_error = np.max(np.abs(errors))

    # stdout only: third_param and corresponding max error
    print(f"{third_param} {max_abs_error}")