"""
PEMD code library.

Developed by: Tan Shendong
Date: 2024.03.17
"""


import os
import glob
import subprocess
import numpy as np
import pandas as pd
from importlib import resources
from PEMD.model import PEMD_lib
from scipy.optimize import curve_fit


def homo_lumo_energy(sorted_df, unit_name, out_dir, length):
    homo_energy, lumo_energy = None, None
    found_homo = False
    log_file_path = sorted_df.iloc[0]['File_Path']

    with open(log_file_path, 'r') as f:
        for line in f:
            if "Alpha  occ. eigenvalues" in line:
                # Convert the last value to float and assign to HOMO energy
                homo_energy = float(line.split()[-1])
                found_homo = True
            elif "Alpha virt. eigenvalues" in line and found_homo:
                # Convert the fifth value to float and assign to LUMO energy
                lumo_energy = float(line.split()[4])
                # No need to break here, as we want the last occurrence
                found_homo = False

    result_df = pd.DataFrame({
        'out_dir': [out_dir],
        'HOMO_Energy_eV': [homo_energy],
        'LUMO_Energy_eV': [lumo_energy]
    })

    # to csv file
    csv_filepath = f'{out_dir}/{unit_name}_N{length}_HOMO_LUMO.csv'

    # 将DataFrame保存为CSV文件
    result_df.to_csv(csv_filepath, index=False)

    return result_df


def dipole_moment(sorted_df, unit_name, out_dir, length):
    found_dipole_moment = None
    log_file_path = sorted_df.iloc[0]['File_Path']

    with open(log_file_path, 'r') as file:
        for line in file:
            if "Dipole moment (field-independent basis, Debye):" in line:
                found_dipole_moment = next(file)  # Keep updating until the last occurrence

    if  found_dipole_moment:
        parts =  found_dipole_moment.split()
        # Extracting the X, Y, Z components and the total dipole moment
        dipole_moment = parts[7]
        dipole_moment_df = pd.DataFrame({'dipole_moment': [dipole_moment]})

        # to csv file
        csv_filepath = f'{out_dir}/{unit_name}_N{length}_dipole_moment.csv'

        # 将DataFrame保存为CSV文件
        dipole_moment_df.to_csv(csv_filepath, index=False)

        return dipole_moment_df


def RESP_fit_Multiwfn(unit_name, length, out_dir, numconf, method,):

    origin_dir = os.getcwd()
    resp_dir = os.path.join(out_dir, 'resp_work')
    os.chdir(resp_dir)

    chk_files = glob.glob('*.chk')
    for chk_file in chk_files:
        PEMD_lib.convert_chk_to_fchk(chk_file)

    # 初始化DataFrame
    resp_chg_df = pd.DataFrame()

    # 使用importlib.resources获取脚本路径
    with resources.path("PEMD.analysis", "calcRESP.sh") as script_path:
        for i in range(numconf):
            if method == 'resp':
                command = ["bash", str(script_path), f"SP_gas_conf_{i}.fchk"]
            elif method == 'resp2':
                command = ["bash", str(script_path), f"SP_gas_conf_{i}.fchk", f"SP_solv_conf_{i}.fchk"]
            else:
                raise ValueError("Unsupported method. Please choose 'resp' or 'resp2'.")

            # 使用subprocess模块调用脚本
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # 输出命令执行结果
            if process.returncode == 0:
                print(f"RESP fitting for the {i + 1}-th structure has been successfully completed.")
            else:
                print(f"RESP fitting for the {i + 1}-th structure failed : {process.stderr}")

            if method == 'resp':
                with open('SP_solv.chg', 'r') as file:
                    lines = file.readlines()
            elif method == 'resp2':
                with open('RESP2.chg', 'r') as file:
                    lines = file.readlines()

            # 第一次循环时读取原子和电荷，后续只更新电荷
            if i == 0:
                data = []
                for line in lines:
                    parts = line.split()
                    if len(parts) == 5:  # 假设格式为：Atom X Y Z Charge
                        atom_name = parts[0]
                        charge = float(parts[-1])
                        data.append((atom_name, charge))

                resp_chg_df = pd.DataFrame(data, columns=['atom', f'charge_{i}'])
            else:
                charges = []
                for line in lines:
                    parts = line.split()
                    if len(parts) == 5:
                        charge = float(parts[-1])
                        charges.append(charge)

                # 将新的电荷数据添加为DataFrame的新列
                resp_chg_df[f'charge_{i}'] = charges

    # 计算所有charge列的平均值，并将结果存储在新列'charge'中
    charge_columns = [col for col in resp_chg_df.columns if 'charge' in col]
    resp_chg_df['charge'] = resp_chg_df[charge_columns].mean(axis=1)
    # 删除原始的charge列
    resp_chg_df.drop(columns=charge_columns, inplace=True)

    os.chdir(origin_dir)

    # to csv file
    csv_filepath = os.path.join(resp_dir, f'{unit_name}_N{length}_{method}_chg.csv')
    resp_chg_df.to_csv(csv_filepath, index=False)

    return resp_chg_df


def dens_temp(out_dir, tpr_file, edr_file, module_soft='GROMACS', initial_time=500, time_gap=4000, duration=1000,
              temp_initial=600, temp_decrement=20, max_time=102000, summary_file="dens_tem.csv"):
    # go to dir
    current_path = os.getcwd()
    MD_dir = os.path.join(current_path, out_dir)
    os.chdir(MD_dir)

    # Load GROMACS module before starting the loop
    subprocess.run(f"module load {module_soft}", shell=True)

    # Initialize a list to store the data
    results = []

    # Loop until time exceeds max_time ps
    time = initial_time
    temp = temp_initial

    while time <= max_time:
        start_time = time
        end_time = time + duration

        print(f"Processing temperature: {temp}K, time interval: {start_time} to {end_time}ps")

        # Use gmx_mpi energy to extract density data and extract the average density value from the output
        command = f"echo Density | gmx_mpi energy -f {edr_file} -s {tpr_file} -b {start_time} -e {end_time}"
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        density_lines = [line for line in result.stdout.split('\n') if "Density" in line]
        density = round(float(density_lines[0].split()[1]) / 1000, 4) if density_lines else "N/A"

        # Append the extracted density value and corresponding temperature to the results list
        results.append({"Temperature": temp, "Density": density})

        # Update time and temperature for the next loop iteration
        time += time_gap
        temp -= temp_decrement

    # Convert the list of results to a DataFrame
    df = pd.DataFrame(results)
    # Save the DataFrame to a CSV file
    df.to_csv(summary_file, index=False)

    os.chdir(current_path)
    print("Density extraction and summary for all temperature points completed.")

    return df


def fit_tg(df, param_file="fitting_tg.csv"):
    # Define the fitting function
    def fit_function(T, a, b, c, Tg, xi):
        return a*T + b - c*(T - Tg) * (1 + (T - Tg) / np.sqrt((T - Tg)**2 + xi**2))

    # Extract temperature and density from the DataFrame
    temperatures = df['Temperature'].to_numpy()
    densities = df['Density'].to_numpy()

    # Initial guess for the fitting parameters
    initial_guess = [1, 1, 1, 300, 1]

    # Set parameter bounds
    bounds = ([-np.inf, -np.inf, -np.inf, 100, 0], [np.inf, np.inf, np.inf, 600, np.inf])

    # Perform the curve fitting
    popt, pcov = curve_fit(
        fit_function,
        temperatures,
        densities,
        p0=initial_guess,
        maxfev=5000,
        bounds=bounds
    )

    # Extracting fitted parameters
    a_fit, b_fit, c_fit, Tg_fit, xi_fit = popt
    print(f"Fitted parameters: a = {a_fit}, b = {b_fit}, c = {c_fit}, Tg = {Tg_fit}, xi = {xi_fit}")
    print(f"Estimated Tg from fit: {Tg_fit} K")

    # Save fitting parameters to CSV file
    param_df = pd.DataFrame({
        'Parameter': ['a', 'b', 'c', 'Tg', 'xi'],
        'Value': [a_fit, b_fit, c_fit, Tg_fit, xi_fit]
    })
    param_df.to_csv(param_file, index=False)

    return param_df




