"""
PEMD code library.

Developed by: Tan Shendong
Date: 2024.03.17
"""


import os
import pandas as pd
import subprocess


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


def RESPchg_fit_Multiwfn():
    current_directory = os.getcwd()
    for item in os.listdir(current_directory):
        if os.path.isdir(item) and item.startswith("RESP"):
            dir_path = os.path.join(current_directory, item)
            # 输出开始信息
            print(f"开始执行 {item} 目录的RESP电荷拟合")
            command = "calcRESP.sh SP_gas.chk SP_solv.chk"
            os.chdir(dir_path)
            try:
                subprocess.run(command, shell=True, check=True)
                # 输出结束信息
                print(f"{item} 目录的RESP电荷拟合完毕")
            except subprocess.CalledProcessError as e:
                print(f"命令在目录 {dir_path} 中执行失败: {e}")
            # 返回到原始目录
            os.chdir(current_directory)
