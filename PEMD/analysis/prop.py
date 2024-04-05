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

    # 初始化DataFrame
    resp_chg_df = pd.DataFrame()

    # 使用importlib.resources获取脚本路径
    with resources.path("PEMD.analysis", "calcRESP.sh") as script_path:
        for i in range(10):
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






