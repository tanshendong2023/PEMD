import os
import numpy as np
import MDAnalysis as mda
from tqdm.auto import tqdm
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor, as_completed


class PolymerIonDynamics:

    def __init__(self, work_dir, tpr_file, xtc_wrap_file, xtc_unwrap_file, cation_atoms, anion_atoms, polymer_atoms,
                 run_start, run_end, dt, dt_collection, time_window_M1, time_window_M2):
        self.work_dir = work_dir
        self.tpr_file = tpr_file
        self.xtc_wrap_file = xtc_wrap_file
        self.xtc_unwrap_file = xtc_unwrap_file
        self.run_start = run_start
        self.run_end = run_end
        self.dt = dt
        self.dt_collection = dt_collection
        self.time_window_M1 = time_window_M1
        self.time_window_M2 = time_window_M2

        self.u_wrap = self.load_trajectory(self.xtc_wrap_file)
        self.u_unwrap = self.load_trajectory(self.xtc_unwrap_file)

        self.cation_atoms_wrap = self.u_wrap.select_atoms(cation_atoms)
        self.anion_atoms_wrap = self.u_wrap.select_atoms(anion_atoms)
        self.polymer_atoms_wrap = self.u_wrap.select_atoms(polymer_atoms)

        self.cation_atoms_unwrap = self.u_unwrap.select_atoms(cation_atoms)
        self.anion_atoms_unwrap = self.u_unwrap.select_atoms(anion_atoms)
        self.polymer_atoms_unwrap = self.u_unwrap.select_atoms(polymer_atoms)

        self.num_cations = len(self.cation_atoms_unwrap)
        self.num_o_polymer = len(self.polymer_atoms_unwrap)
        self.num_chain = len(np.unique(self.polymer_atoms_unwrap.resids))
        self.num_o_chain = int(self.num_o_polymer // self.num_chain)

        self.times_M1 = self.times_range(self.time_window_M1)
        self.times_M2 = self.times_range(self.time_window_M2)
        self.times_MR = self.times_range(self.run_end - self.run_start)

        self.initialize_simulation()

    def load_trajectory(self, xtc_file):
        tpr_path = os.path.join(self.work_dir, self.tpr_file)
        xtc_path = os.path.join(self.work_dir, xtc_file)
        return mda.Universe(tpr_path, xtc_path)

    def initialize_simulation(self):
        self.box_size = self.u_unwrap.dimensions[0]
        self.poly_o_ave_n, self.poly_n, self.bound_o_n, self.poly_o_positions= self.process_traj()
        self.re_all = self.ms_endtoend_distance()
        self.msd_M2 = self.calculate_msd_parallel(self.calculate_msd_M2, self.time_window_M2)

    def distance(self, x0, x1, box_length):
        delta = x1 - x0
        delta = np.where(delta > 0.5 * box_length, delta - box_length, delta)
        delta = np.where(delta < -0.5 * box_length, delta + box_length, delta)
        return delta

    def process_traj(self,):

        poly_o_ave_n = np.zeros((len(self.times_MR), self.num_cations))  # Initialize array
        poly_n = np.zeros((len(self.times_MR), self.num_cations))  # Initialize array
        bound_o_n = np.full((len(self.times_MR), self.num_cations, 10), -1, dtype=int)  # 初始化bound氧的索引
        poly_o_positions = np.zeros((len(self.times_MR), self.num_o_polymer, 3))  # 初始化氧坐标的数组

        for ts in tqdm(self.u_unwrap.trajectory[self.run_start: self.run_end], desc='Processing trajectory'):

            for n, li in enumerate(self.cation_atoms_unwrap):
                distances_oe_vec = self.distance(self.polymer_atoms_unwrap.positions, li.position,
                                                 self.box_size)
                distances_oe = np.linalg.norm(distances_oe_vec, axis=1)
                close_oe_index = np.where(distances_oe <= 3.575)[0]

                if len(close_oe_index) > 0:
                    o_resids = self.polymer_atoms_unwrap[close_oe_index].resids

                    if np.all(o_resids == o_resids[0]):
                        poly_o_ave_n[ts.frame, n] = np.mean(close_oe_index)
                        poly_n[ts.frame, n] = o_resids[0]
                        bound_o_n[ts.frame, n, :len(close_oe_index)] = close_oe_index  # bound氧的索引
                    else:
                        poly_n[ts.frame, n] = -1

            for i in range(int(self.num_chain)):
                oe_in_onechain = self.u_unwrap.select_atoms(f'resid {i + 1}')   # 选择当前链中的原子
                # 计算当前链的原子相对于质心的位置
                poly_o_positions[ts.frame, self.num_o_chain * i:self.num_o_chain * (i + 1),
                :] = self.polymer_atoms_unwrap.positions[self.num_o_chain * i:self.num_o_chain * (i + 1),
                     :] - oe_in_onechain.center_of_mass()

        return poly_o_ave_n, poly_n, bound_o_n, poly_o_positions

    def ms_endtoend_distance(self):
        re_all = []

        for ts in tqdm(self.u_unwrap.trajectory[self.run_start: self.run_end], desc="Processing"):
            ts_vectors = []

            for mol_id in range(1, int(self.num_chain) + 1):  # Assuming 20 molecules
                chain_indices = np.where(self.polymer_atoms_unwrap.resids == mol_id)[0]  # 获得所有聚合物链醚氧的index，并每一条链单独储存一个index
                if len(chain_indices) > 1:  # Ensure there is more than one oxygen atom
                    chain_coor = self.polymer_atoms_unwrap.positions[chain_indices]  # 获得每条聚合物链醚氧的坐标
                    chain1_coor = chain_coor[1:]
                    chain2_coor = chain_coor[:-1]
                    b0_array = self.distance(chain1_coor, chain2_coor, self.box_size)  # 生成每个间隔醚氧的向量
                    re_vector = np.sum(b0_array, axis=0)  # 所有向量加和
                    re = np.linalg.norm(re_vector)  # 对向量进行模长的计算
                    ts_vectors.append(re)

            if ts_vectors:
                ts_vectors = np.square(ts_vectors)
                re_mean = np.mean(ts_vectors)
                re_all.append(re_mean)

        return re_all

    def times_range(self, t_end):
        times = np.arange(0, t_end * self.dt * self.dt_collection, self.dt * self.dt_collection, dtype=int)
        return times

    def rouse_model(self, t, tau, Re_square,):
        """计算 Rouse 模型的理论值，用于拟合 MSD 数据。"""
        sum_part = sum([(1 - np.exp(-p ** 2 * t / tau)) / p ** 2 for p in range(1, self.num_chain - 1)])
        return (2 * Re_square / np.pi ** 2) * sum_part

    def fit_rouse_model(self, times, msd):
        """计算 Rouse 时间常数并拟合 MSD 数据。"""
        Re_square = np.mean(self.re_all)  # 平均平方端到端距离
        tau,_ = curve_fit(lambda t, tau: self.rouse_model(t, tau, Re_square), times, msd)
        return tau[0] / 1000

    def calculate_msd_M2(self, dt):
        msd_in_t = []
        if dt == 0:
            return 0
        for t in range(self.run_start, self.run_end - dt):
            delta_n = self.poly_o_positions[t + dt] - self.poly_o_positions[t]
            delta_n_square = np.sum(np.square(delta_n), axis=1)

            i = np.where((np.abs(self.poly_o_ave_n[dt + t] - self.poly_o_ave_n[t]) <= 1) & (self.poly_o_ave_n[t] != 0))[0]

            bound_counts = np.sum(np.abs(self.poly_o_ave_n[t:t + dt] - self.poly_o_ave_n[t]) <= 1, axis=0)
            j = np.where((bound_counts / dt) >= 0.85)[0]
            li_intersection = np.intersect1d(i, j)

            all_bound_oe_indices = []  # 创建一个列表来收集所有有效的索引
            for idx in li_intersection:
                valid_indices = self.bound_o_n[t, idx][self.bound_o_n[t, idx] != -1]
                if valid_indices.size > 0:
                    all_bound_oe_indices.extend(valid_indices)  # 收集所有有效的索引

            if all_bound_oe_indices:
                msd_in_t.append(np.mean(delta_n_square[all_bound_oe_indices]))

        return np.mean(msd_in_t) if msd_in_t else 0

    def calculate_msd_parallel(self, calculate_msd, time_window):
        """Calculate the parallel computation of MSD over multiple time deltas."""
        msd = []
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(calculate_msd, dt,): dt for dt in range(time_window)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Calculate MSD"):
                dt = futures[future]
                msd_result = future.result()
                msd.append((dt, msd_result))
        msd.sort(key=lambda x: x[0])
        return np.array([result for _, result in msd])




