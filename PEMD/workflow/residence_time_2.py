# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import os
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
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
                for shell_atom in shell.atoms:
                    if str(shell_atom.id) not in bool_values:
                        bool_values[str(shell_atom.id)] = np.zeros(int((run_end - run_start) / 1))
                    bool_values[str(shell_atom.id)][time_count] = 1
            acfs = calc_acf(bool_values)
            acf_all.extend(list(acfs))
        acf_avg[kw] = np.mean(acf_all, axis=0)
    return acf_avg

def exponential_func(
    x: float | np.floating | np.ndarray,
    a: float | np.floating | np.ndarray,
    b: float | np.floating | np.ndarray,
    c: float | np.floating | np.ndarray,
) -> np.floating | np.ndarray:

    return a * np.exp(-b * x) + c

def fit_residence_time(
    times: np.ndarray,
    acf_avg_dict: dict[str, np.ndarray],
    cutoff_time: int,
    time_step: float,
    save_curve: str | bool = False,
) -> dict[str, np.floating]:

    acf_avg_norm = {}
    popt = {}
    pcov = {}
    tau = {}
    species_list = list(acf_avg_dict.keys())

    # Exponential fit of solvent-Li ACF
    for kw in species_list:
        acf_avg_norm[kw] = acf_avg_dict[kw] / acf_avg_dict[kw][0]
        popt[kw], pcov[kw] = curve_fit(
            exponential_func,
            times[:cutoff_time],
            acf_avg_norm[kw][:cutoff_time],
            p0=(1, 1e-4, 0),
        )
        tau[kw] = 1 / popt[kw][1]  # ps

    # Plot ACFs
    colors = ["b", "g", "r", "c", "m", "y"]
    line_styles = ["-", "--", "-.", ":"]
    for i, kw in enumerate(species_list):
        plt.plot(times, acf_avg_norm[kw], label=kw, color=colors[i])
        fitted_x = np.linspace(0, cutoff_time * time_step, cutoff_time)
        fitted_y = exponential_func(np.linspace(0, cutoff_time * time_step, cutoff_time), *popt[kw])
        save_decay = np.vstack(
            (
                times[:cutoff_time],
                acf_avg_norm[kw][:cutoff_time],
                fitted_x,
                fitted_y,
            )
        )
        if save_curve:
            if save_curve is True:
                np.savetxt(f"decay{i}.csv", save_decay.T, delimiter=",")
            elif os.path.exists(str(save_curve)):
                np.savetxt(str(save_curve) + f"decay{i}.csv", save_decay.T, delimiter=",")
            else:
                raise ValueError("Please specify a bool or a path in string.")
        plt.plot(
            fitted_x,
            fitted_y,
            line_styles[i],
            color="k",
            label=kw + " Fit",
        )

    plt.xlabel("Time (ps)")
    plt.legend()
    plt.ylabel("Neighbor Auto-correlation Function")
    plt.ylim(0, 1)
    plt.xlim(0, cutoff_time * time_step)
    plt.show()

    return tau

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    work_dir = './'
    tpr_file = 'nvt_prod.tpr'
    xtc_file = 'nvt_prod.xtc'
    run_start = 0
    run_end = 40001  # step

    # Load the trajectory
    u = coordination.load_md_trajectory(work_dir, tpr_file, xtc_file)
    volume = u.coord.volume

    # Select the atoms of interest
    li_atoms = u.select_atoms('resname LIP and name Li')
    peo_atoms = u.select_atoms('resname MOL and name O')
    tfsi_atoms = u.select_atoms('resname NSC and name OBT')
    sn_atoms = u.select_atoms('resname SN and name N')

    # Perform RDF and coordination number calculation
    bins_peo, rdf_peo, coord_num_peo = coordination.calc_rdf_coord(li_atoms, peo_atoms, volume)
    bins_tfsi, rdf_tfsi, coord_num_tfsi = coordination.calc_rdf_coord(li_atoms, tfsi_atoms, volume)
    bins_sn, rdf_sn, coord_num_sn = coordination.calc_rdf_coord(li_atoms, sn_atoms, volume)

    # obtain the coordination number and first solvation shell distance
    r_li_peo, y_coord_peo = coordination.obtain_rdf_coord(bins_peo, rdf_peo, coord_num_peo)
    r_li_tfsi, y_coord_tfsi = coordination.obtain_rdf_coord(bins_tfsi, rdf_tfsi, coord_num_tfsi)
    r_li_sn, y_coord_sn = coordination.obtain_rdf_coord(bins_sn, rdf_sn, coord_num_sn)

    distance_dict = {"polymer": r_li_peo, "anion": r_li_tfsi, 'SN': r_li_sn}
    select_dict = {
        "cation": "resname LIP and name Li",
        "anion": "resname NSC and name OBT",
        "polymer": "resname MOL and name O",
        "SN": 'resname SN and name N'
    }

    run = coordination.load_md_trajectory(work_dir, tpr_file, xtc_file)
    times = times_array(run, run_start, run_end, time_step=5)
    acf_avg = calc_neigh_corr(run, distance_dict, select_dict, run_start, run_end)

    residence_time = fit_residence_time(times, acf_avg, 1000, 5)
    save_to_json(residence_time, 'residence_time.json')

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





