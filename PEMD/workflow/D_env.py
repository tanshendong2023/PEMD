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
from concurrent.futures import ProcessPoolExecutor, as_completed

def distance(x0, x1, box_length):
    delta = x1 - x0
    delta = np.where(delta > 0.5 * box_length, delta - box_length, delta)
    delta = np.where(delta < -0.5 * box_length, delta + box_length, delta)
    return delta

def load_data_traj(work_dir, data_tpr_file, dcd_xtc_file, select_atoms, run_start, run_end, dt, dt_collection,
                   cutoff_radii):
    data_tpr_file_path = os.path.join(work_dir, data_tpr_file)
    dcd_xtc_file_path = os.path.join(work_dir, dcd_xtc_file)

    u = mda.Universe(data_tpr_file_path, dcd_xtc_file_path)

    # Selections
    li_atoms = u.select_atoms(select_atoms['cation'])
    tfsi_atoms = u.select_atoms(select_atoms['anion'])
    oe_atoms = u.select_atoms(select_atoms['PEO'])
    sn_atoms = u.select_atoms(select_atoms['SN'])

    num_li = int(len(li_atoms))
    li_positions = np.zeros((run_end - run_start, num_li, 3))
    li_env = np.zeros((run_end - run_start, num_li))

    # Trajectory analysis
    for ts in tqdm(u.trajectory[run_start:run_end], desc='Processing trajectory'):

        box_size = ts.dimensions[0]
        li_positions[ts.frame, :, :] = li_atoms.positions[:, :]

        for n, li in enumerate(li_atoms):
            distances_oe_vec = distance(oe_atoms.positions, li.position, box_size)
            distances_oe = np.linalg.norm(distances_oe_vec, axis=1)
            close_oe = np.where(distances_oe < cutoff_radii['PEO'])[0]

            distances_tfsi_vec = distance(tfsi_atoms.positions, li.position, box_size)
            distances_tfsi = np.linalg.norm(distances_tfsi_vec, axis=1)
            close_tfsi = np.where(distances_tfsi < cutoff_radii['TFSI'])[0]

            distances_sn_vec = distance(sn_atoms.positions, li.position, box_size)
            distances_sn = np.linalg.norm(distances_sn_vec, axis=1)
            close_sn = np.where(distances_sn < cutoff_radii['SN'])[0]

            # Assign environment type
            if close_oe.any() and not close_tfsi.any() and not close_sn.any():
                li_env[ts.frame, n] = 1
            elif close_oe.any() and (close_tfsi.any() or close_sn.any()):
                li_env[ts.frame, n] = 2
            elif not close_oe.any() and (close_tfsi.any() or close_sn.any()):
                li_env[ts.frame, n] = 3

    return li_positions, li_env

def compute_delta_d_square(dt, li_positions, li_env, env, run_start, run_end, threshold):
    msd_in_dt = []
    count_li_dt = []
    if dt == 0:
        return 0, 0  # MSD at dt=0 is 0 as Δn would be 0

    for t in range(run_start, run_end - dt):
        delta_d = li_positions[t + dt] - li_positions[t]
        delta_d_square = np.square(delta_d)

        i = (li_env[t] == env)  # 确保 t 和 t + dt 都在 li_env 的有效范围内
        j = (li_env[t + dt] == env)
        bound_counts = np.sum(np.abs(li_env[t:t + dt] - li_env[t]) < 1, axis=0)
        h = (bound_counts / dt) >= threshold
        li_intersection = i & j & h

        count_li_dt.append(np.sum(li_intersection))

        delta_d_square_filtered = delta_d_square[li_intersection]
        if delta_d_square_filtered.size > 0:
            msd_in_dt.append(np.mean(delta_d_square_filtered))

    return np.mean(msd_in_dt) if msd_in_dt else 0, np.mean(count_li_dt) if count_li_dt else 0

def compute_msd_parallel(li_positions, li_env, env, run_start, run_end, time_window, dt, dt_collection, threshold=0.95):
    times = np.arange(0, time_window * dt * dt_collection, dt * dt_collection, dtype=int)
    msd = []
    count_li = []

    with ProcessPoolExecutor(max_workers=32) as executor:
        futures = {
            executor.submit(compute_delta_d_square, dt, li_positions, li_env, env, run_start, run_end, threshold): dt
            for dt in range(time_window)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating MSD"):
            dt = futures[future]
            msd_result, count_li_result = future.result()
            msd.append((dt, msd_result))
            count_li.append((dt, count_li_result))

    msd.sort(key=lambda x: x[0])
    count_li.sort(key=lambda x: x[0])

    return np.array([result for _, result in msd]), np.array([result for _, result in count_li]), times

if __name__ == '__main__':
    work_dir = './'

    data_tpr_file = 'nvt_prod.tpr'
    dcd_xtc_file = 'nvt_prod_unwrap.xtc'

    select_atoms = {
        'cation': 'resname LIP and name Li',
        'anion': 'resname NSC and name OBT',
        'PEO': 'resname MOL and name O',
        'SN': 'resname SN and name N',
    }

    cutoff_radii = {
        'PEO': 3.6,
        'TFSI': 3.2,
        'SN': 3.7,
    }

    run_start = 0
    run_end = 80001  # step
    dt = 0.001
    dt_collection = 5000  # step
    time_window = 10001
    env = 1  # 1: PEO  2: PEO/SN/TFSI 3: SN/TFSI

    #  loads and processes trajectory data
    (
        li_positions,
        li_env,
    ) = load_data_traj(
        work_dir,
        data_tpr_file,
        dcd_xtc_file,
        select_atoms,
        run_start,
        run_end,
        dt,
        dt_collection,
        cutoff_radii,
    )

    (
        msd,
        count_li,
        times
    ) = compute_msd_parallel(
        li_positions,
        li_env,
        env,
        run_start,
        run_end,
        time_window,
        dt,
        dt_collection
    )

    # 创建包含三个列的DataFrame
    df = pd.DataFrame({
        'Time': times,
        'MSD': msd,
        'Count_Li': count_li
    })

    # 写入CSV文件
    df.to_csv(f'msd_{env}.csv', index=False)

    # Print success message
    print("CSV file has been successfully saved!")

