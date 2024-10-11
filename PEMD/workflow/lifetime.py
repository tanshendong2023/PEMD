# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import os
import numpy as np
import pandas as pd
import MDAnalysis as mda
from tqdm.auto import tqdm

def load_md_trajectory(work_dir, tpr_filename='nvt_prod.tpr', xtc_filename='nvt_prod.xtc'):
    data_tpr_file = os.path.join(work_dir, tpr_filename)
    data_xtc_file = os.path.join(work_dir, xtc_filename)
    u = mda.Universe(data_tpr_file, data_xtc_file)
    return u

def times_array(run, run_start, run_end, time_step=5):
    times = []
    for step, _ts in enumerate(run.trajectory[run_start:run_end]):
        times.append(step * time_step)
    return np.array(times)

def calc_acf(h_values: dict[str, np.ndarray]) -> np.ndarray:
    # 获取所有分子的 h_ij(t) 数组，组成一个二维数组
    h_matrix = np.array(list(h_values.values()))
    num_mols, num_times = h_matrix.shape
    max_lag = num_times  # 最大时间滞后

    # 初始化自相关函数数组
    acf = np.zeros(max_lag)

    # 计算归一化因子
    h0_squared_mean = np.mean(h_matrix[:, 0] ** 2)

    for lag in range(max_lag):
        # 对于每个时间滞后，计算 h_ij(t) * h_ij(t + lag)
        products = []
        for h_ij in h_matrix:
            if num_times - lag > 0:
                h_t = h_ij[:num_times - lag]
                h_t_lag = h_ij[lag:]
                product = h_t * h_t_lag
                products.extend(product)
        # 计算平均值并归一化
        if h0_squared_mean != 0:
            acf[lag] = np.mean(products) / h0_squared_mean
        else:
            acf[lag] = 0
    return acf

def calc_neigh_corr(run, distance_dict, select_dict, run_start, run_end):
    acf_avg = {}
    center_atoms = run.select_atoms('resname LIP and name Li')
    for kw in distance_dict:
        acf_all = []
        for atom in tqdm(center_atoms):
            distance = distance_dict.get(kw)
            assert distance is not None
            h_values = {}
            for time_count, _ts in enumerate(run.trajectory[run_start:run_end]):
                if kw in select_dict:
                    selection = (
                        "("
                        + select_dict[kw]
                        + ") and (around "
                        + str(distance)
                        + " index "
                        + str(atom.id - 1)
                        + ")"
                    )
                    shell = run.select_atoms(selection)
                else:
                    raise ValueError("Invalid species selection")
                # 获取这些原子所属的分子ID
                mols = set(atom.residue for atom in shell)
                for mol in mols:
                    mol_id = str(mol.resid)
                    if mol_id not in h_values:
                        h_values[mol_id] = np.zeros(run_end - run_start)
                    h_values[mol_id][time_count] = 1
            # 对于每个锂离子-分子对，计算时间自相关函数
            acf = calc_acf(h_values)
            acf_all.append(acf)
        # 对所有锂离子取平均
        acf_avg[kw] = np.mean(acf_all, axis=0)
    return acf_avg


if __name__ == '__main__':
    work_dir = '../'
    tpr_file = 'nvt_prod.tpr'
    xtc_file = 'nvt_prod.xtc'
    run_start = 0
    run_end = 10001  # step

    distance_dict = {"polymer": 3.7, "anion": 3.5, 'SN':3.7}        # Li-PEO 3.7; Li-TFSI 3.3; Li-SN 3.7
    select_dict = {
        "cation": "resname LIP and name Li",
        "anion": "resname NSC and name OBT",
        "polymer": "resname MOL and name O",
        "SN": 'resname SN and name N'
    }

    run = load_md_trajectory(work_dir, tpr_file, xtc_file)
    times = times_array(run, run_start, run_end, time_step=5)
    acf_avg = calc_neigh_corr(run, distance_dict, select_dict, run_start, run_end)

    # 归一化自相关函数
    acf_avg_norm = {}
    species_list = list(acf_avg.keys())
    for kw in species_list:
        if kw in acf_avg:
            acf_avg_norm[kw] = acf_avg[kw] / acf_avg[kw][0]  # 防止除以零的错误处理

    # 准备将时间和归一化后的自相关函数保存在同一个CSV文件中
    acf_df = pd.DataFrame({'Time (ps)': times})
    for key, value in acf_avg_norm.items():
        acf_df[key + ' ACF'] = pd.Series(value)

    # 将数据保存到CSV文件
    acf_df.to_csv('residence_time.csv', index=False)







