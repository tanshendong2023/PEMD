"""
PEMD code library.

Developed by: Tan Shendong
Date: 2024.03.17
"""


import os
import glob
import subprocess
import numpy as np
import pandas as pd
import MDAnalysis as mda
from tqdm.auto import tqdm
from PEMD.model import PEMD_lib
import matplotlib.pyplot as plt
from importlib import resources
from scipy.optimize import curve_fit


def homo_lumo_energy(sorted_df, unit_name, out_dir, length):
    homo_energy, lumo_energy = None, None
    found_homo = False
    log_file_path = sorted_df.iloc[0]['File_Path']

    with open(log_file_path, 'r') as f:
        for line in f:
            if "Alpha  occ. eigenvalues" in line:
                # Convert the last value to float and assign to HOMO energy
                homo_energy = float(line.split()[-1])
                found_homo = True
            elif "Alpha virt. eigenvalues" in line and found_homo:
                # Convert the fifth value to float and assign to LUMO energy
                lumo_energy = float(line.split()[4])
                # No need to break here, as we want the last occurrence
                found_homo = False

    result_df = pd.DataFrame({
        'out_dir': [out_dir],
        'HOMO_Energy_eV': [homo_energy],
        'LUMO_Energy_eV': [lumo_energy]
    })

    # to csv file
    csv_filepath = f'{out_dir}/{unit_name}_N{length}_HOMO_LUMO.csv'

    # 将DataFrame保存为CSV文件
    result_df.to_csv(csv_filepath, index=False)

    return result_df


def dipole_moment(sorted_df, unit_name, out_dir, length):
    found_dipole_moment = None
    log_file_path = sorted_df.iloc[0]['File_Path']

    with open(log_file_path, 'r') as file:
        for line in file:
            if "Dipole moment (field-independent basis, Debye):" in line:
                found_dipole_moment = next(file)  # Keep updating until the last occurrence

    if  found_dipole_moment:
        parts =  found_dipole_moment.split()
        # Extracting the X, Y, Z components and the total dipole moment
        dipole_moment = parts[7]
        dipole_moment_df = pd.DataFrame({'dipole_moment': [dipole_moment]})

        # to csv file
        csv_filepath = f'{out_dir}/{unit_name}_N{length}_dipole_moment.csv'

        # 将DataFrame保存为CSV文件
        dipole_moment_df.to_csv(csv_filepath, index=False)

        return dipole_moment_df


def RESP_fit_Multiwfn(out_dir, numconf, method,):

    origin_dir = os.getcwd()
    resp_dir = os.path.join(out_dir, 'resp_work')
    os.chdir(resp_dir)

    chk_files = glob.glob('*.chk')
    for chk_file in chk_files:
        PEMD_lib.convert_chk_to_fchk(chk_file)

    # 初始化DataFrame
    resp_chg_df = pd.DataFrame()

    # 使用importlib.resources获取脚本路径
    with resources.path("PEMD.analysis", "calcRESP.sh") as script_path:
        for i in range(numconf):
            if method == 'resp':
                command = ["bash", str(script_path), f"SP_gas_conf_{i}.fchk"]
            elif method == 'resp2':
                command = ["bash", str(script_path), f"SP_gas_conf_{i}.fchk", f"SP_solv_conf_{i}.fchk"]
            else:
                raise ValueError("Unsupported method. Please choose 'resp' or 'resp2'.")

            # 使用subprocess模块调用脚本
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # 输出命令执行结果
            if process.returncode == 0:
                print(f"RESP fitting for the {i + 1}-th structure has been successfully completed.")
            else:
                print(f"RESP fitting for the {i + 1}-th structure failed : {process.stderr}")

            if method == 'resp':
                with open('SP_solv.chg', 'r') as file:
                    lines = file.readlines()
            elif method == 'resp2':
                with open('RESP2.chg', 'r') as file:
                    lines = file.readlines()

            # 第一次循环时读取原子和电荷，后续只更新电荷
            if i == 0:
                data = []
                for line in lines:
                    parts = line.split()
                    if len(parts) == 5:  # 假设格式为：Atom X Y Z Charge
                        atom_name = parts[0]
                        charge = float(parts[-1])
                        data.append((atom_name, charge))

                resp_chg_df = pd.DataFrame(data, columns=['atom', f'charge_{i}'])
            else:
                charges = []
                for line in lines:
                    parts = line.split()
                    if len(parts) == 5:
                        charge = float(parts[-1])
                        charges.append(charge)

                # 将新的电荷数据添加为DataFrame的新列
                resp_chg_df[f'charge_{i}'] = charges

    # 计算所有charge列的平均值，并将结果存储在新列'charge'中
    charge_columns = [col for col in resp_chg_df.columns if 'charge' in col]
    resp_chg_df['charge'] = resp_chg_df[charge_columns].mean(axis=1)
    # 删除原始的charge列
    resp_chg_df.drop(columns=charge_columns, inplace=True)

    os.chdir(origin_dir)

    # to csv file
    csv_filepath = os.path.join(resp_dir, f'{method}_chg.csv')
    resp_chg_df.to_csv(csv_filepath, index=False)

    return resp_chg_df


def dens_temp(out_dir, tpr_file, edr_file, module_soft='GROMACS', initial_time=500, time_gap=4000, duration=1000,
              temp_initial=600, temp_decrement=20, max_time=102000, summary_file="dens_tem.csv"):
    # go to dir
    current_path = os.getcwd()
    MD_dir = os.path.join(current_path, out_dir)
    os.chdir(MD_dir)

    # Load GROMACS module before starting the loop
    subprocess.run(f"module load {module_soft}", shell=True)

    # Initialize a list to store the data
    results = []

    # Loop until time exceeds max_time ps
    time = initial_time
    temp = temp_initial

    while time <= max_time:
        start_time = time
        end_time = time + duration

        print(f"Processing temperature: {temp}K, time interval: {start_time} to {end_time}ps")

        # Use gmx_mpi energy to extract density data and extract the average density value from the output
        command = f"echo Density | gmx_mpi energy -f {edr_file} -s {tpr_file} -b {start_time} -e {end_time}"
        result = subprocess.run(command, shell=True, text=True, capture_output=True)
        density_lines = [line for line in result.stdout.split('\n') if "Density" in line]
        density = round(float(density_lines[0].split()[1]) / 1000, 4) if density_lines else "N/A"

        # Append the extracted density value and corresponding temperature to the results list
        results.append({"Temperature": temp, "Density": density})

        # Update time and temperature for the next loop iteration
        time += time_gap
        temp -= temp_decrement

    # Convert the list of results to a DataFrame
    df = pd.DataFrame(results)
    # Save the DataFrame to a CSV file
    df.to_csv(summary_file, index=False)

    os.chdir(current_path)
    print("Density extraction and summary for all temperature points completed.")

    return df


def fit_tg(df, param_file="fitting_tg.csv"):
    # Define the fitting function
    def fit_function(T, a, b, c, Tg, xi):
        return a*T + b - c*(T - Tg) * (1 + (T - Tg) / np.sqrt((T - Tg)**2 + xi**2))

    # Extract temperature and density from the DataFrame
    temperatures = df['Temperature'].to_numpy()
    densities = df['Density'].to_numpy()

    # Initial guess for the fitting parameters
    initial_guess = [1, 1, 1, 300, 1]

    # Set parameter bounds
    bounds = ([-np.inf, -np.inf, -np.inf, 100, 0], [np.inf, np.inf, np.inf, 600, np.inf])

    # Perform the curve fitting
    popt, pcov = curve_fit(
        fit_function,
        temperatures,
        densities,
        p0=initial_guess,
        maxfev=5000,
        bounds=bounds
    )

    # Extracting fitted parameters
    a_fit, b_fit, c_fit, Tg_fit, xi_fit = popt
    print(f"Fitted parameters: a = {a_fit}, b = {b_fit}, c = {c_fit}, Tg = {Tg_fit}, xi = {xi_fit}")
    print(f"Estimated Tg from fit: {Tg_fit} K")

    # Save fitting parameters to CSV file
    param_df = pd.DataFrame({
        'Parameter': ['a', 'b', 'c', 'Tg', 'xi'],
        'Value': [a_fit, b_fit, c_fit, Tg_fit, xi_fit]
    })
    param_df.to_csv(param_file, index=False)

    return param_df


def get_position(data_tpr_file, dcd_xtc_file, select_cations, select_anions, dt, dt_collection, run_start, nsteps,
                 format='GROMACS'):

    run = mda.Universe(data_tpr_file, dcd_xtc_file)

    cations = run.select_atoms(select_cations).residues
    anions = run.select_atoms(select_anions).residues

    cations_list = cations.atoms.split("residue")
    anions_list = anions.atoms.split("residue")

    t_total = nsteps - run_start  # total simulation steps, minus equilibration time

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

    for ts in enumerate(tqdm(run.trajectory[int(run_start/dt_collection):])):
        system_com = run.atoms.center_of_mass(wrap=True)
        for index, cation in enumerate(cations_list):
            cation_positions[time, index, :] = cation.center_of_mass() - system_com
        for index, anion in enumerate(anions_list):
            anion_positions[time, index, :] = anion.center_of_mass() - system_com
        time += 1

    return cation_positions, anion_positions


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


def cross_corr(x, y):

    N = len(x)
    F1 = np.fft.fft(x, n=2 ** (N * 2 - 1).bit_length())
    F2 = np.fft.fft(y, n=2 ** (N * 2 - 1).bit_length())
    PSD = F1 * F2.conjugate()
    res = np.fft.ifft(PSD)
    res = (res[:N]).real
    n = N * np.ones(N) - np.arange(0, N)
    cf = res / n
    return cf


def msd_fft_cross(r, k):

    N = len(r)
    D = np.multiply(r, k).sum(axis=1)
    D = np.append(D, 0)
    S2 = sum([cross_corr(r[:, i], k[:, i]) for i in range(r.shape[1])])
    S3 = sum([cross_corr(k[:, i], r[:, i]) for i in range(k.shape[1])])
    Q = 2 * D.sum()
    S1 = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    msd = S1 - S2 - S3
    return msd


def calc_Lii_self(atom_positions, times):

    Lii_self = np.zeros(len(times))
    n_atoms = np.shape(atom_positions)[1]
    for atom_num in (range(n_atoms)):
        r = atom_positions[:,atom_num, :]
        msd_temp = msd_fft(np.array(r))
        Lii_self += msd_temp
    msd = np.array(Lii_self)
    return msd


def calc_Lii(atom_positions, times):

    r_sum = np.sum(atom_positions, axis = 1)
    msd = msd_fft(r_sum)
    return np.array(msd)


def calc_Lij(cation_positions, anion_positions, times):

    r_cat = np.sum(cation_positions, axis = 1)
    r_an = np.sum(anion_positions, axis = 1)
    msd = msd_fft_cross(np.array(r_cat),np.array(r_an))
    return np.array(msd)


def compute_all_Lij(cation_positions, anion_positions, times):

    msd_self_cation = calc_Lii_self(cation_positions, times)
    msd_self_anion =  calc_Lii_self(anion_positions, times)
    msd_cation = calc_Lii(cation_positions, times)
    msd_anion = calc_Lii(anion_positions, times)
    msd_distinct_catAn = calc_Lij(cation_positions, anion_positions, times)
    msds_all = [msd_cation, msd_self_cation, msd_anion, msd_self_anion, msd_distinct_catAn]
    return msds_all


def compute_slope_msd(msd, times, dt_collection, dt, interval_time=5000): # 5ns
    log_time = np.log(times)
    log_msd = np.log(msd)

    dt_ = dt_collection * dt
    interval_msd = int(interval_time / dt_)
    small_interval = 200 # 1ns

    average_slopes = []  # 存储每个大间隔的平均斜率
    x_ranges = []  # 存储每个大间隔的x坐标范围
    closest_slope = float('inf')
    time_range = (None, None)

    for i in range(0, len(log_time) - interval_msd, interval_msd):
        slopes_log = []
        for j in range(i, i + interval_msd - small_interval, small_interval):
            delta_y = log_msd[j + small_interval] - log_msd[j]
            delta_x = log_time[j + small_interval] - log_time[j]
            if delta_x != 0:
                slope_log = delta_y / delta_x
                slopes_log.append(slope_log)

        # 计算当前大间隔内的平均斜率
        if slopes_log:
            average_slope = np.mean(slopes_log)
            average_slopes.append(average_slope)
            x_ranges.append((times[i], times[i + interval_msd]))

            # 更新最接近1的平均斜率及其范围
            if abs(average_slope - 1) < abs(closest_slope - 1):
                closest_slope = average_slope
                time_range = (times[i], times[i + interval_msd])

    slope = (msd[int(time_range[1] / dt_)] - msd[int(time_range[0] / dt_)]) / (time_range[1] - time_range[0])
    return slope, time_range


def compute_self_diffusion(atom_positions, times, dt_collection, dt, interval_time):

    n_atoms = np.shape(atom_positions)[1]
    msd = calc_Lii_self(atom_positions, times) / n_atoms  # mean for particle

    # Utilize the common slope calculation function
    slope, time_range = compute_slope_msd(msd, times, dt_collection, dt, interval_time)

    A2cm = 1e-8  # Angstroms to cm
    ps2s = 1e-12  # picoseconds to seconds
    convert = (A2cm ** 2) / ps2s   # cm^2/s
    D = slope * convert / 6

    return msd, D, time_range


def plot_msd(msd_data, times, time_ranges, dt_collection, dt, labels, save_file):
    font_list = {"title": 20, "label": 18, "legend": 16, "ticket": 18, "data": 14}
    color_list = ["#DF543F", "#2286A9", "#FBBF7C", "#3C3846"]

    dt_ = dt_collection * dt
    fig, ax = plt.subplots()

    # 判断是单个MSD还是双MSD
    if isinstance(msd_data, list):  # 假设msd_data是列表，包含两个MSD数组
        for i, msd in enumerate(msd_data):
            mid_time = (time_ranges[i][1] + time_ranges[i][0]) / 2
            start = int(10 ** (np.log10(mid_time) - 0.15) / dt_)
            end = int(10 ** (np.log10(mid_time) + 0.15) / dt_)
            scale = (msd[int(mid_time / dt_)] + 40) / mid_time

            x_log = times[start:end]
            y_log = x_log * scale

            ax.plot(times[1:], msd[1:], '-', linewidth=1.5, color=color_list[i], label=labels[i])
            ax.plot(x_log, y_log, '--', linewidth=2, color=color_list[i])
    else:  # 单个MSD
        mid_time = (time_ranges[1] + time_ranges[0]) / 2
        start = int(10 ** (np.log10(mid_time) - 0.15) / dt_)
        end = int(10 ** (np.log10(mid_time) + 0.15) / dt_)
        scale = (msd_data[int(mid_time / dt_)] + 40) / mid_time

        x_log = times[start:end]
        y_log = x_log * scale

        ax.plot(times[1:], msd_data[1:], '-', linewidth=1.5, color=color_list[0], label=labels)
        ax.plot(x_log, y_log, '--', linewidth=2, color="grey")

    ax.legend(fontsize=font_list["legend"], frameon=False)
    ax.set_xlabel(r'$t$ (ps)', fontsize=font_list["label"])
    ax.set_ylabel(r'MSD ($\AA^2$)', fontsize=font_list["label"])
    ax.tick_params(axis='both', which='both', direction='in', labelsize=font_list["ticket"])

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(1e2,)

    ax.grid(True, linestyle='--')
    fig.set_size_inches(5.5, 4)
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'{save_file}', bbox_inches='tight', dpi=300)
    plt.show()


def compute_conductivity(run, run_start, dt_collection, cations_list, anions_list, times, dt, T, interval_time=5000):

    # compute sum over all charges and positions
    qr = []
    for _ts in tqdm(run.trajectory[int(run_start / dt_collection):]):
        qr_temp = np.zeros(3)
        for cation in cations_list:
            qr_temp += cation.center_of_mass() * int(1)
        for anion in anions_list:
            qr_temp += anion.center_of_mass() * int(-1)
        qr.append(qr_temp)
    msd = msd_fft(np.array(qr))

    # Utilize the common slope calculation function
    slope, time_range = compute_slope_msd(msd, times, dt_collection, dt, interval_time)

    A2cm = 1e-8  # Angstroms to cm
    ps2s = 1e-12  # picoseconds to seconds
    e2c = 1.60217662e-19  # elementary charge to Coulomb
    kb = 1.38064852e-23  # Boltzmann Constant, J/K
    convert = e2c * e2c / ps2s / A2cm * 1000
    v = (run.dimensions[0]) ** 3.0

    cond = slope / 6 / kb / T / v * convert   # "mS/cm"

    return msd, cond, time_range


def compute_transfer_number(run, dt_collection, cation_positions, anion_positions, times, dt, T, interval_time):

    msds_all = compute_all_Lij(cation_positions, anion_positions, times)

    # Utilize the common slope calculation function
    slope_plusplus, time_range_plusplus = compute_slope_msd(msds_all[0], times, dt_collection, dt, interval_time)
    slope_minusminus, time_range_minusminus = compute_slope_msd(msds_all[2], times, dt_collection, dt, interval_time)
    slope_plusminus, time_range_plusminus = compute_slope_msd(msds_all[4], times, dt_collection, dt, interval_time)

    A2cm = 1e-8  # Angstroms to cm
    ps2s = 1e-12  # picoseconds to seconds
    e2c = 1.60217662e-19  # elementary charge to Coulomb
    kb = 1.38064852e-23  # Boltzmann Constant, J/K
    convert = e2c * e2c / ps2s / A2cm * 1000
    v = (run.dimensions[0]) ** 3.0

    t = (slope_plusplus + slope_minusminus - 2 * slope_plusminus) / 6 / kb / T / v * convert   # mS/cm

    return t














