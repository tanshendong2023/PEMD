"""
PEMD code library.

Developed by: Tan Shendong
Date: 2024.03.17
"""


import os
import glob
import pandas as pd
import subprocess
from importlib import resources
from PEMD.model import PEMD_lib


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


def RESP_fit_Multiwfn(unit_name, length, out_dir, method='resp',):

    origin_dir = os.getcwd()
    resp_dir = os.path.join(out_dir, 'resp_work')
    os.chdir(resp_dir)

    chk_files = glob.glob('*.chk')
    for chk_file in chk_files:
        PEMD_lib.convert_chk_to_fchk(chk_file)

    # 使用importlib.resources获取脚本路径
    with resources.path("PEMD.analysis", "calcRESP.sh") as script_path:
        if method == 'resp':
            command = ["bash", str(script_path), "SP_solv.fchk"]
        elif method == 'resp2':
            command = ["bash", str(script_path), "SP_gas.fchk", "SP_solv.fchk"]
        else:
            raise ValueError("Unsupported method. Please choose 'resp' or 'resp2'.")

        # 使用subprocess模块调用脚本
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 输出命令执行结果
        if process.returncode == 0:
            print("RESP fitting completed successfully.")
        else:
            print(f"Error during RESP fitting: {process.stderr}")

    if method == 'resp':
        with open('SP_solv.chg', 'r') as file:
            lines = file.readlines()
    elif method == 'resp2':
        with open('RESP2.chg', 'r') as file:
            lines = file.readlines()
    # Extract atom names and charges
    data = []
    for line in lines:
        parts = line.split()
        if len(parts) == 5:  # Atom X Y Z Charge
            atom_name = parts[0]
            charge = float(parts[-1])
            data.append((atom_name, charge))

    # Create a DataFrame
    resp_chg_df = pd.DataFrame(data, columns=['atom', 'charge'])
    os.chdir(origin_dir)

    # to csv file
    csv_filepath = f'{out_dir}/{unit_name}_N{length}_{method}_chg.csv'
    resp_chg_df.to_csv(csv_filepath, index=False)

    return resp_chg_df





