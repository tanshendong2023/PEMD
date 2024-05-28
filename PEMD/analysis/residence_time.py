import os
import numpy as np
import MDAnalysis as mda
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from MDAnalysis.analysis import distances
from MDAnalysis.lib.distances import distance_array


def distance(x0,x1,box_length):
    delta = x1 - x0
    delta = np.where(delta > 0.5 * box_length, delta - box_length, delta)
    delta = np.where(delta < -0.5 * box_length, delta + box_length, delta)
    return delta


def ms_endtoend_distance(data_tpr_file, dcd_xtc_file, run_start, dt_collection, chains, select_atom='name O'):
    # Load the trajectory
    u = mda.Universe(data_tpr_file, dcd_xtc_file)
    box_size = u.dimensions[0]

    # Preselecting atoms
    oxygen_atoms = u.select_atoms(select_atom)  # Select all oxygen atoms at once
    re_all = [] # inital the end to end distance list

    for ts in tqdm(u.trajectory[int(run_start/dt_collection):], desc="Processing"):
        ts_vectors = []

        for mol_id in range(1, int(chains)+1):  # Assuming 20 molecules
            chain_indices = np.where(oxygen_atoms.resids == mol_id)[0]  # 获得所有聚合物链醚氧的index，并每一条链单独储存一个index
            if len(chain_indices) > 1:  # Ensure there is more than one oxygen atom
                chain_coor = oxygen_atoms.positions[chain_indices]  # 获得每条聚合物链醚氧的坐标
                chain1_coor = chain_coor[1:]
                chain2_coor = chain_coor[:-1]
                b0_array = distance(chain1_coor, chain2_coor, box_size)  # 生成每个间隔醚氧的向量
                re_vector = np.sum(b0_array, axis=0)  # 所有向量加和
                re = np.linalg.norm(re_vector)  # 对向量进行模长的计算
                ts_vectors.append(re)

        if ts_vectors:
            ts_vectors = np.square(ts_vectors)
            re_mean = np.mean(ts_vectors)
            re_all.append(re_mean)

    return re_all


def plot_ms_endtoend_distance(re_all, run_start, run_end, dt_collection, system="PEO-LiTFSI"):
    print(f"Mean squared end-to-end distance: {np.mean(re_all)}")
    print(f"stddev: {np.std(re_all)}")

    font_list = {"title": 20, "label": 18, "legend": 16, "ticket": 18, "data": 14}
    linewidth = 1.5
    color_list = ["#DF543F", "#2286A9", "#FBBF7C", "#3C3846"]

    t = np.arange(int(run_start/dt_collection), int(run_end/dt_collection)+1)

    fig, ax = plt.subplots()
    ax.plot(t, re_all, '-', linewidth=linewidth, color=color_list[2], label=system)

    ax.legend(fontsize=font_list["legend"], frameon=False)
    ax.set_xlabel(r'$t$ (ps)', fontsize=font_list["label"])
    ax.set_ylabel(r'Average $R{_e}{^2}$', fontsize=font_list["label"])
    ax.tick_params(axis='x', labelsize=font_list["ticket"], direction='in')
    ax.tick_params(axis='y', labelsize=font_list["ticket"], direction='in')
    ax.grid(True, linestyle='--')
    fig.set_size_inches(6, 5)
    plt.tight_layout()


def get_ether_oxygen_position(data_tpr_file, dcd_xtc_file, run_start, dt, nsteps, dt_collection):
    # load the trajectory
    u = mda.Universe(data_tpr_file, dcd_xtc_file)

    # Select the atoms of interest
    oe = u.select_atoms('resname MOL and name O')
    t_total = nsteps - run_start  # total simulation steps, minus equilibration time
    times = np.arange(0, t_total * dt + 1, dt * dt_collection, dtype=int)

    # 初始化参数
    chains = 20  # 链的条数
    oe_per_chain = 50  # 每链中的OE数目

    # 初始化用于存储位置的数组
    atom_positions = np.zeros((len(times), len(oe), 3))

    # 处理每个时间步
    time = 0
    for ts in tqdm(u.trajectory[int(run_start/dt_collection):], desc="Processing"):  # 遍历每个链
        for i in range(int(chains)):
            # 选择当前链中的原子
            oe_in_onechain = u.select_atoms(f'resid {i + 1}')
            # 计算当前链的原子相对于质心的位置
            atom_positions[time, oe_per_chain * i:oe_per_chain * (i + 1), :] = oe.positions[oe_per_chain * i:oe_per_chain * (i + 1),:] - oe_in_onechain.center_of_mass()
        time += 1

    return atom_positions, times



















