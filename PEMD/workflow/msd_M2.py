#!/usr/bin/env python


import os
import numpy as np
import MDAnalysis as mda
from tqdm.auto import tqdm
from PEMD.analysis import residence_time, coordination
from concurrent.futures import ProcessPoolExecutor, as_completed

work_dir = './'

data_tpr_file = 'nvt_prod.tpr'
dcd_xtc_file = 'nvt_prod_unwrap.xtc'

select_atoms = {
    'cation': 'resname LIP and name Li',
    'anion': 'resname NSC and name OBT',
    'polymer': 'resname MOL and name O',
}

# Load the trajectory
u = coordination.load_md_trajectory(work_dir, 'nvt_prod.tpr', 'nvt_prod.xtc')

# Select the atoms of interest
li_atoms = u.select_atoms('resname LIP and name Li')
peo_atoms = u.select_atoms('resname MOL and name O')
tfsi_atoms = u.select_atoms('resname NSC and name OBT')

# Perform RDF and coordination number calculation
bins_peo, rdf_peo, coord_num_peo = coordination.calculate_rdf_and_coordination(u, li_atoms, peo_atoms)
bins_tfsi, rdf_tfsi, coord_num_tfsi = coordination.calculate_rdf_and_coordination(u, li_atoms, tfsi_atoms)

# obtain the coordination number and first solvation shell distance
r_li_peo, y_rdf_peo, y_coord_peo = coordination.obtain_rdf_coord(bins_peo, rdf_peo, coord_num_peo)
r_li_tfsi, y_rdf_tfsi, y_coord_tfsi = coordination.obtain_rdf_coord(bins_tfsi, rdf_tfsi, coord_num_tfsi)

# setting the first solvation shell distance
cutoff_radii = {
    'PEO': r_li_peo,
    'TFSI': r_li_tfsi,
}

run_start = 0
run_end = 80001  # step
dt = 0.001
dt_collection = 5000 # step
num_li = 50
num_oe = 50
chains = 20
time_window = 4001

def distance(x0, x1, box_length):
    delta = x1 - x0
    delta = np.where(delta > 0.5 * box_length, delta - box_length, delta)
    delta = np.where(delta < -0.5 * box_length, delta + box_length, delta)
    return delta

def load_data_traj(work_dir, data_tpr_file, dcd_xtc_file, select_atoms, run_start, run_end, dt, dt_collection,
                   cutoff_radii):

    # load trajectory
    data_tpr_file_path = os.path.join(work_dir, data_tpr_file)
    dcd_xtc_file_path = os.path.join(work_dir, dcd_xtc_file)

    u = mda.Universe(data_tpr_file_path, dcd_xtc_file_path)

    # obtain the time array
    # t_total = run_end - run_start  # total simulation steps, minus equilibration time
    # times = np.arange(0, t_total * dt * dt_collection, dt * dt_collection, dtype=int)

    li_atoms = u.select_atoms(select_atoms['cation'])
    oe_atoms = u.select_atoms(select_atoms['polymer'])

    # 初始化参数
    num_li = int(len(li_atoms))
    chain_n = len(np.unique(oe_atoms.resids))  # 链的条数

    oe_ave_n = np.zeros((run_end - run_start, num_li))  # 初始化锂周围氧的平均索引
    li_positions = np.zeros((run_end - run_start, num_li, 3))  # 初始化锂离子坐标的数组

    for ts in tqdm(u.trajectory[run_start: run_end], desc='processing'):

        li_positions[ts.frame, :, :] = li_atoms.positions[:,:]

        box_size = ts.dimensions[0]

        for n, li in enumerate(li_atoms):

            distances_oe_vec = distance(oe_atoms.positions, li.position, box_size)
            distances_oe = np.linalg.norm(distances_oe_vec, axis=1)
            close_oe_index = np.where(distances_oe < cutoff_radii['PEO'])[0]

            if len(close_oe_index) > 0:  # 确保选择的Li都和醚氧相互作用
                o_resids = oe_atoms[close_oe_index].resids  # 找到醚氧所在的链

                # 检查所有的醚氧和同一个聚合物相互作用
                if np.all(o_resids == o_resids[0]):
                    oe_ave_n[ts.frame, n] = np.mean(close_oe_index)  # 锂周围氧的平均索引

    return oe_ave_n, li_positions


def compute_delta_n_square(dt, li_positions, oe_ave_n, run_start, run_end, threshold):
    msd_in_dt = []
    count_li_dt = []
    if dt == 0:
        return 0, 0  # MSD at dt=0 is 0 as Δn would be 0
    for t in range(run_start, run_end - dt):
        delta_n = li_positions[t + dt] - li_positions[t]
        delta_n_square = np.sum(np.square(delta_n), axis=1)

        i = (np.abs(oe_ave_n[dt + t] - oe_ave_n[t]) <= 1) & (oe_ave_n[t] != 0)
        bound_counts = np.sum(np.abs(oe_ave_n[t:t + dt] - oe_ave_n[t]) <= 1, axis=0)
        j = (bound_counts / dt) >= threshold
        li_intersection = i & j

        count_li_dt.append(np.sum(li_intersection))

        delta_n_square_filtered = delta_n_square[li_intersection]

        if delta_n_square_filtered.size > 0:
            msd_in_dt.append(np.mean(delta_n_square_filtered))

    return np.mean(msd_in_dt) if msd_in_dt else 0, np.mean(count_li_dt) if count_li_dt else 0


def compute_dn_msd_parallel(li_positions, oe_ave_n, run_start, run_end, time_window, dt, dt_collection, threshold=0.85):
    times = np.arange(0, time_window * dt * dt_collection, dt * dt_collection, dtype=int)
    msd = []
    count_li = []
    with ProcessPoolExecutor() as executor:
        # 使用字典存储future以保持顺序
        futures = {
            executor.submit(compute_delta_n_square, dt, li_positions, oe_ave_n, run_start, run_end, threshold): dt for
            dt in range(time_window)}
        # 确保结果按照dt顺序处理
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculate MSD"):
            dt = futures[future]
            msd_result, count_li_result = future.result()
            msd.append((dt, msd_result))
            count_li.append((dt, count_li_result))
    # 按dt排序并提取MSD值
    msd.sort(key=lambda x: x[0])
    count_li.sort(key=lambda x: x[0])
    return np.array([result for _, result in msd]), np.array([result for _, result in count_li]), times


if __name__ == '__main__':
    #  loads and processes trajectory data
    oe_ave_n, li_positions, = load_data_traj(work_dir, data_tpr_file, dcd_xtc_file, select_atoms, run_start, run_end,
                                             dt, dt_collection, cutoff_radii,)

    msd, count_li, times = compute_dn_msd_parallel(li_positions, oe_ave_n, run_start, run_end, time_window, dt,
                                                   dt_collection)

    np.savetxt('msd_M2.txt', msd)
    np.savetxt('count_li_M2.txt', count_li)
    np.savetxt('times_M2.txt', times)





