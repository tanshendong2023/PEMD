# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from PEMD.analysis import coordination
from statsmodels.tsa.stattools import acovf

def times_array(run, run_start, run_end, time_step=5):
    times = []
    for step, _ts in enumerate(run.trajectory[run_start:run_end]):
        times.append(step * time_step)
    return np.array(times)

def calc_acf(a_values: dict[str, np.ndarray]) -> list[np.ndarray]:
    acfs = []
    for neighbors in a_values.values():  # for _atom_id, neighbors in a_values.items():
        acfs.append(acovf(neighbors, demean=False, adjusted=True, fft=True))
    return acfs

def calc_neigh_corr(run, distance_dict, select_dict, run_start, run_end):
    acf_avg = {}
    center_atoms = run.select_atoms('resname LIP and name Li')
    for kw in distance_dict:
        acf_all = []
        for atom in tqdm(center_atoms[::]):
            distance = distance_dict.get(kw)
            assert distance is not None
            bool_values = {}
            for time_count, _ts in enumerate(run.trajectory[run_start:run_end:]):
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
                for atom in shell.atoms:
                    if str(atom.id) not in bool_values:
                        bool_values[str(atom.id)] = np.zeros(int((run_end - run_start) / 1))
                    bool_values[str(atom.id)][time_count] = 1
            acfs = calc_acf(bool_values)
            acf_all.extend(list(acfs))
        acf_avg[kw] = np.mean(acf_all, axis=0)
    return acf_avg

if __name__ == '__main__':
    work_dir = './'
    tpr_file = 'nvt_prod.tpr'
    xtc_file = 'nvt_prod.xtc'
    run_start = 0
    run_end = 20001  # step

    distance_dict = {"polymer": 3.625, "anion": 3.175, 'SN': 3.375}
    select_dict = {
        "cation": "'resname LIP and name Li'",
        "anion": "resname NSC and name OBT",
        "polymer": "resname MOL and name O",
        "SN": 'resname SN and name N'

    }

    run = coordination.load_md_trajectory(work_dir, tpr_file, xtc_file)
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
    acf_df.to_csv('residence_time_normalized.csv', index=False)





