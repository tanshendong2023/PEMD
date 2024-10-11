# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import os
import time
import numpy as np
import pandas as pd
import MDAnalysis as mda
from tqdm.auto import tqdm
from numba import njit, prange

def distance(x0, x1, box_length):
    delta = x1 - x0
    delta -= box_length * np.round(delta / box_length)
    return delta

def load_data_traj(work_dir, tpr_file, xtc_wrap_file, xtc_unwrap_file, select_atoms, run_start, run_end, cutoff_radii):
    tpr_filepath = os.path.join(work_dir, tpr_file)
    xtc_wrap_filepath = os.path.join(work_dir, xtc_wrap_file)
    xtc_unwrap_filepath = os.path.join(work_dir, xtc_unwrap_file)

    run_wrap = mda.Universe(tpr_filepath, xtc_wrap_filepath)
    run_unwrap = mda.Universe(tpr_filepath, xtc_unwrap_filepath)

    # Selections
    cations_wrap = run_wrap.select_atoms(select_atoms['cation'])
    cations_unwrap = run_unwrap.select_atoms(select_atoms['cation'])
    anions_wrap = run_wrap.select_atoms(select_atoms['anion'])
    PEO_wrap = run_wrap.select_atoms(select_atoms['PEO'])
    SN_wrap = run_wrap.select_atoms(select_atoms['SN'])

    n_cations = len(cations_wrap)
    total_frames = run_end - run_start
    li_positions = np.zeros((total_frames, n_cations, 3), dtype=np.float64)
    li_env = np.zeros((total_frames, n_cations), dtype=np.int32)

    # Trajectory analysis for positions
    for ts in tqdm(run_unwrap.trajectory[run_start:run_end], desc='Processing positions'):
        li_positions[ts.frame - run_start] = cations_unwrap.positions.copy()

    # Trajectory analysis for environments
    for ts in tqdm(run_wrap.trajectory[run_start:run_end], desc='Processing environments'):
        box_size = ts.dimensions[0]

        li_pos = cations_wrap.positions.copy()
        PEO_pos = PEO_wrap.positions.copy()
        anions_pos = anions_wrap.positions.copy()
        SN_pos = SN_wrap.positions.copy()

        for n in range(n_cations):
            li = li_pos[n]

            distances_oe = np.linalg.norm(distance(PEO_pos, li, box_size), axis=1)
            close_oe = np.any(distances_oe < cutoff_radii['PEO'])

            distances_tfsi = np.linalg.norm(distance(anions_pos, li, box_size), axis=1)
            close_tfsi = np.any(distances_tfsi < cutoff_radii['TFSI'])

            distances_sn = np.linalg.norm(distance(SN_pos, li, box_size), axis=1)
            close_sn = np.any(distances_sn < cutoff_radii['SN'])

            # Assign environment type
            if close_oe and not (close_tfsi or close_sn):
                li_env[ts.frame - run_start, n] = 1
            elif close_oe and (close_tfsi or close_sn):
                li_env[ts.frame - run_start, n] = 2
            elif not close_oe and (close_tfsi or close_sn):
                li_env[ts.frame - run_start, n] = 3

    return li_positions, li_env

@njit(parallel=True)
def compute_msd(li_positions, li_env, env, time_window, threshold):
    total_frames, n_cations, _ = li_positions.shape
    msd_array = np.zeros(time_window)
    count_li_array = np.zeros(time_window)
    for dt in prange(1, time_window):
        msd_sum = 0.0
        msd_count = 0
        count_li_sum = 0.0
        count_li_entries = 0

        max_t = total_frames - dt
        for t in range(max_t):
            delta_d = li_positions[t + dt] - li_positions[t]
            delta_d_square = np.sum(delta_d ** 2, axis=1)

            i = li_env[t] == env
            j = li_env[t + dt] == env

            # Calculate how long the Li ion stays in the same environment
            bound_counts = np.zeros(n_cations, dtype=np.int32)
            # never_env_3 = np.ones(n_cations, dtype=np.bool_)

            for k in range(dt):
                same_env = li_env[t + k] == li_env[t]
                bound_counts += same_env.astype(np.int32)

                # not_env_3 = li_env[t + k] != 3
                # never_env_3 &= not_env_3

            h = (bound_counts / dt) >= threshold

            li_intersection = i & j & h # & never_env_3

            count_li = np.sum(li_intersection)
            if count_li > 0:
                msd_sum += np.sum(delta_d_square[li_intersection])
                msd_count += count_li
                count_li_sum += count_li
                count_li_entries += 1

        if msd_count > 0:
            msd_array[dt] = msd_sum / msd_count
        else:
            msd_array[dt] = 0.0

        if count_li_entries > 0:
            count_li_array[dt] = count_li_sum / count_li_entries
        else:
            count_li_array[dt] = 0.0

    return msd_array, count_li_array

if __name__ == '__main__':
    work_dir = '../'

    tpr_file = 'nvt_prod.tpr'
    xtc_wrap_file = 'nvt_prod.xtc'
    xtc_unwrap_file = 'nvt_prod_unwrap.xtc'

    select_atoms = {
        'cation': 'resname LIP and name Li',
        'anion': 'resname NSC and name OBT',
        'PEO': 'resname MOL and name O',
        'SN': 'resname SN and name N',
    }

    cutoff_radii = {
        'PEO': 3.7,
        'TFSI': 3.5,
        'SN': 3.7,
    }

    run_start = 0
    run_end = 80001  # frames
    time_step = 0.001  # ps per MD step
    dt_collection = 5000  # MD steps between frames
    time_per_frame = time_step * dt_collection  # ps per frame
    time_window = 2001  # number of dt indices
    env = 1  # 1: PEO  2: PEO/SN/TFSI 3: SN/TFSI

    start_time = time.time()  # 获取当前时间

    # 文件名
    li_positions_file = 'li_positions.npy'
    li_env_file = 'li_env.npy'

    # 检查文件是否存在
    if os.path.exists(li_positions_file) and os.path.exists(li_env_file):
        print("加载已有的 li_positions 和 li_env 数据...")
        li_positions = np.load(li_positions_file)
        li_env = np.load(li_env_file)
    else:
        print("计算 li_positions 和 li_env...")
        # 加载并处理轨迹数据
        li_positions, li_env = load_data_traj(
            work_dir,
            tpr_file,
            xtc_wrap_file,
            xtc_unwrap_file,
            select_atoms,
            run_start,
            run_end,
            cutoff_radii,
        )
        # 保存数据到文件
        np.save(li_positions_file, li_positions)
        np.save(li_env_file, li_env)
        print("li_positions 和 li_env 已保存到文件。")

    # 计算 MSD
    msd, count_li = compute_msd(
        li_positions,
        li_env,
        env,
        time_window,
        threshold=0.95
    )

    # 时间数组
    times = np.arange(time_window) * time_per_frame

    end_time = time.time()  # 获取当前时间
    print(f"代码执行时间：{end_time - start_time} 秒")

    # 创建包含三个列的DataFrame
    df = pd.DataFrame({
        'Time': times,
        'MSD': msd,
        'Count_Li': count_li
    })

    # 写入CSV文件
    df.to_csv(f'msd_{env}.csv', index=False)

    # 打印成功信息
    print("CSV file has been successfully saved!")



