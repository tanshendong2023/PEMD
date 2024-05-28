#!/usr/bin/env python


import numpy as np
import MDAnalysis as mda
from tqdm.auto import tqdm

# Load the trajectory
u = mda.Universe('nvt_prod.tpr', 'unwrapped_traj.xtc')


def distance(x0, x1, box_length):
    delta = x1 - x0
    delta = np.where(delta > 0.5 * box_length, delta - box_length, delta)
    delta = np.where(delta < -0.5 * box_length, delta + box_length, delta)
    return delta


run_start = 0
run_end = 80001  # 10 ps * 10000 = 100 ns

rc = 3.575
li_atoms = u.select_atoms('resname LIP and name Li')
oe_atoms = u.select_atoms('resname MOL and name O')

# Precompute box size and other invariant properties
box_size = u.dimensions[0]

msd_data = []
ave_oe_mols = []

# Process trajectory
for ts in tqdm(u.trajectory[run_start:run_end], desc="Processing"):
    ave_o_mols_list = []
    avg_n_list = []

    li_positions = li_atoms.positions
    oe_positions = oe_atoms.positions
    oe_resids = oe_atoms.resids

    for li_pos in li_positions:
        distances_vec = distance(oe_positions, li_pos, box_size)
        distances = np.linalg.norm(distances_vec, axis=1)
        close_o_indices = distances < rc

        if np.any(close_o_indices):
            o_resids = oe_resids[close_o_indices]
            ave_o_mols = np.mean(o_resids)
            ave_o_mols_list.append(ave_o_mols)

            if np.all(o_resids == o_resids[0]):  # All Oes belong to the same molecule
                ave_n = np.mean(np.where(close_o_indices)[0])
                avg_n_list.append(ave_n)
            else:
                avg_n_list.append(0)
        else:
            avg_n_list.append(0)
            ave_o_mols_list.append(0)

    ave_oe_mols.append(ave_o_mols_list)
    msd_data.append(avg_n_list)

msd_data = np.array(msd_data)
ave_oe_mols = np.array(ave_oe_mols)

time_window = 8000
msd = []

for dt in tqdm(range(time_window), desc="Processing"):
    msd_in_dt = []
    for t in range(run_start, run_end-dt):
        delta_n = msd_data[t+dt] - msd_data[t]
        delta_n_square = np.square(delta_n)

        # 使用布尔索引直接过滤无效的数据点
        valid_mask = (msd_data[t+dt] != 0) & (msd_data[t] != 0)
        valid_mask &= ((ave_oe_mols[t+dt] == ave_oe_mols[t]) |
                       (np.floor(ave_oe_mols[t+dt]) != ave_oe_mols[t+dt]) |
                       (np.floor(ave_oe_mols[t]) != ave_oe_mols[t]))

        # 应用有效掩码
        valid_delta_n_square = delta_n_square[valid_mask]

        if valid_delta_n_square.size > 0:
            msd_in_dt.append(np.mean(valid_delta_n_square))
        else:
            msd_in_dt.append(0)  # 如果没有有效的数据点，则添加0

    if msd_in_dt:
        msd_avg_in_dt = np.mean(msd_in_dt)
        msd.append(msd_avg_in_dt)
    else:
        msd.append(0)  # 确保即使没有数据点也添加0

msd = np.array(msd)
# print(msd.shape)

np.savetxt('msd.txt', msd)

