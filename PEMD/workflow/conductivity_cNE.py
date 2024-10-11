# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import os
import numpy as np
import MDAnalysis as mda
from tqdm.auto import tqdm
from PEMD.analysis import coordination
from concurrent.futures import ProcessPoolExecutor, as_completed


def distance(x0, x1, box_length):
    delta = x1 - x0
    delta = np.where(delta > 0.5 * box_length, delta - box_length, delta)
    delta = np.where(delta < -0.5 * box_length, delta + box_length, delta)
    return delta

def calc_population_frame(ts, cations, anions, index):
    all_atoms = cations + anions
    merged_list = list(range(len(all_atoms)))
    box_size = ts.dimensions[0]
    all_clusters = []

    while merged_list:
        this_cluster = [merged_list[0]]
        merged_list.remove(merged_list[0])

        for i in this_cluster:
            for j in merged_list:
                d_vec = distance(all_atoms.positions[i], all_atoms.positions[j], box_size)
                d = np.linalg.norm(d_vec)
                if d <= 3.4:
                    this_cluster.append(j)
                    merged_list.remove(j)

        all_clusters.append(this_cluster)

    type_id = 'Li'
    type_id3 = 'NBT'
    pop_matrix = np.zeros((50, 50, 1))

    for cluster in all_clusters:
        cations_count = 0
        anions_count = 0

        for atom_id in cluster:
            if all_atoms[atom_id].type == type_id:
                cations_count += 1
            if all_atoms[atom_id].type == type_id3:
                anions_count += 1

        if cations_count < 50 and anions_count < 50:
            pop_matrix[cations_count][anions_count] += 1

    return pop_matrix, index

def calc_population_parallel(run, run_start, run_end, select_cations, select_anions):
    stacked_population = np.array([])

    with ProcessPoolExecutor() as executor:
        futures = []
        for idx, ts in enumerate(run.trajectory[run_start:run_end]):
            cations = run.select_atoms(select_cations)
            anions = run.select_atoms(select_anions)
            futures.append(executor.submit(calc_population_frame, ts, cations, anions, idx))

        results = [future.result() for future in
                   tqdm(as_completed(futures), total=len(futures), desc='Processing trajectory')]

    # Sort results by index and combine
    sorted_results = sorted(results, key=lambda x: x[1])
    for current_population, _ in sorted_results:
        if stacked_population.size == 0:
            stacked_population = current_population
        else:
            stacked_population = np.dstack((stacked_population, current_population))

    avg_population = np.mean(stacked_population, axis=2)
    return avg_population

def get_position(work_dir, data_tpr_file, dcd_xtc_file, select_cations, select_anions, dt, dt_collection, run_start,
                 nsteps, format='GROMACS'):
    data_tpr_file_path = os.path.join(work_dir, data_tpr_file)
    dcd_xtc_file_path = os.path.join(work_dir, dcd_xtc_file)

    run = mda.Universe(data_tpr_file_path, dcd_xtc_file_path)

    cations = run.select_atoms(select_cations).residues
    anions = run.select_atoms(select_anions).residues

    cations_list = cations.atoms.split("residue")
    anions_list = anions.atoms.split("residue")

    t_total = nsteps - run_start

    times = None
    if format == 'GROMACS':
        times = np.arange(0, t_total * dt + 1, dt * dt_collection, dtype=int)
    elif format == 'LAMMPS':
        times = np.arange(0, t_total * dt, dt * dt_collection, dtype=int)

    return run, cations, cations_list, anions, anions_list, times

def create_position_arrays(run, cations_list, anions_list, times, run_start, dt_collection):
    time = 0

    cation_positions = np.zeros((len(times), len(cations_list), 3))
    anion_positions = np.zeros((len(times), len(anions_list), 3))

    for ts in enumerate(tqdm(run.trajectory[int(run_start / dt_collection):])):
        system_com = run.atoms.center_of_mass(wrap=True)
        for index, cation in enumerate(cations_list):
            cation_positions[time, index, :] = cation.center_of_mass() - system_com
        for index, anion in enumerate(anions_list):
            anion_positions[time, index, :] = anion.center_of_mass() - system_com
        time += 1

    return cation_positions, anion_positions

def calc_slope_msd(times_array, msd_array, dt_collection, dt, interval_time=10000, step_size=10):
    # 确保传入对数函数的值是正值
    valid_times = times_array[1:]  # 排除时间为零的情况
    valid_msds = msd_array[1:]  # 同样，排除任何非正MSD值
    log_time = np.log(valid_times)
    log_msd = np.log(np.maximum(valid_msds, 1e-10))  # 防止对零或负值取对数

    dt_ = dt_collection * dt
    interval_msd = int(interval_time / dt_)

    time_range = (None, None)
    min_slope_sum = float('inf')

    for i in range(0, len(log_time) - interval_msd, step_size):
        if i + interval_msd > len(log_time):
            break
        local_slope = np.gradient(log_msd[i:i + interval_msd], log_time[i:i + interval_msd])
        slope_difference_sum = np.sum(np.abs(local_slope - 1))
        if slope_difference_sum < min_slope_sum:
            min_slope_sum = slope_difference_sum
            time_range = (times_array[i], times_array[i + interval_msd])

    # 检查 time_range 是否被有效更新
    if None in time_range:
        raise ValueError("未找到合适的时间范围进行坡度计算。")

    final_slope = (msd_array[int(time_range[1] / dt_)] - msd_array[int(time_range[0] / dt_)]) / (
                time_range[1] - time_range[0])

    return final_slope, time_range

def compute_self_diffusion(atom_positions, times, dt_collection, dt, interval_time, step_size):
    n_atoms = np.shape(atom_positions)[1]
    msd = calc_Lii_self(atom_positions, times) / n_atoms  # mean for particle

    # Utilize the common slope calculation function
    slope, time_range = calc_slope_msd(times, msd, dt_collection, dt, interval_time, step_size)

    A2cm = 1e-8  # Angstroms to cm
    ps2s = 1e-12  # picoseconds to seconds
    convert = (A2cm ** 2) / ps2s  # cm^2/s
    D = slope * convert / 6

    return msd, D, time_range

def autocorrFFT(x):
    N = len(x)
    F = np.fft.fft(x, n=2 * N)
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real
    n = N * np.ones(N) - np.arange(0, N)
    acf = res / n
    return acf

def msd_fft(r):
    N = len(r)
    D = np.square(r).sum(axis=1)
    D = np.append(D, 0)
    S2 = sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    msd = S1 - 2 * S2
    return msd

def calc_Lii_self(atom_positions, times):
    Lii_self = np.zeros(len(times))
    n_atoms = np.shape(atom_positions)[1]
    for atom_num in tqdm(range(n_atoms), desc="Calculating MSD"):
        r = atom_positions[:, atom_num, :]
        msd_temp = msd_fft(np.array(r))
        Lii_self += msd_temp
    msd = np.array(Lii_self)
    return msd

if __name__ == '__main__':

    work_dir = '../'
    tpr_file = 'nvt_prod.tpr'
    xtc_wrap_file = 'nvt_prod.xtc'
    select_cations = 'name Li'
    select_anions = 'name NBT or name OBT or name SBT'
    run_start = 0
    run_end = 40001


    data_tpr_file = 'nvt_prod.tpr'
    dcd_xtc_file = 'nvt_prod_unwrap.xtc'
    select_cations = 'resname LIP'
    select_anions = 'resname NSC'
    dt = 0.001
    dt_collection = 5000
    run_start = 0
    nsteps = 4e8
    T = 450  # K
    interval_time = 10000  # slope range 5ns
    step_size = 10

    run = coordination.load_md_trajectory(work_dir, tpr_file, xtc_wrap_file)
    avg_population = calc_population_parallel(run, run_start, run_end, select_cations, select_anions)

    (
        run,
        cations,
        cations_list,
        anions,
        anions_list,
        times,
    ) = get_position(
        work_dir,
        data_tpr_file,
        dcd_xtc_file,
        select_cations,
        select_anions,
        dt,
        dt_collection,
        run_start,
        nsteps,
        format='GROMACS',
    )







