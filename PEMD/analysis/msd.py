import os
import numpy as np
import MDAnalysis as mda
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def get_position(work_dir, data_tpr_file, dcd_xtc_file, select_cations, select_anions, dt, dt_collection, run_start,
                 nsteps, format='GROMACS'):
    """
    Calculate the positions of ions in a molecular dynamics simulation.

    Parameters:
    work_dir : str
        The working directory where files are stored.
    data_tpr_file : str
        The topology file for the simulation (GROMACS TPR file).
    dcd_xtc_file : str
        The trajectory file for the simulation (DCD or XTC file).
    select_cations : str
        Selection string for cations.
    select_anions : str
        Selection string for anions.
    dt : float
        Time step of the simulation.
    dt_collection : int
        Interval of steps for collecting data.
    run_start : int
        The starting step of data collection to skip initial equilibration.
    nsteps : int
        Total number of steps in the simulation.
    format : str, optional
        Specifies the simulation software used ('GROMACS' or 'LAMMPS'). Default is 'GROMACS'.

    Returns:
    tuple
        Contains the simulation universe, cation and anion residue groups, lists of cation and anion atoms, and the times array.
    """

    # Construct full paths to the data and trajectory files
    data_tpr_file_path = os.path.join(work_dir, data_tpr_file)
    dcd_xtc_file_path = os.path.join(work_dir, dcd_xtc_file)

    # Load the simulation data
    run = mda.Universe(data_tpr_file_path, dcd_xtc_file_path)

    # Select cations and anions based on user-defined criteria
    cations = run.select_atoms(select_cations).residues
    anions = run.select_atoms(select_anions).residues

    # Split atoms into lists by residue for cations and anions
    cations_list = cations.atoms.split("residue")
    anions_list = anions.atoms.split("residue")

    # Calculate total number of steps to analyze, after accounting for initial equilibration
    t_total = nsteps - run_start

    # Initialize time array for data collection based on the format and collection interval
    times = None
    if format == 'GROMACS':
        times = np.arange(0, t_total * dt + 1, dt * dt_collection, dtype=int)
    elif format == 'LAMMPS':
        times = np.arange(0, t_total * dt, dt * dt_collection, dtype=int)

    # Return all collected data
    return run, cations, cations_list, anions, anions_list, times


def create_position_arrays(run, cations_list, anions_list, times, run_start, dt_collection):
    """
    Generates arrays of position data for cations and anions relative to the center of mass of the system.

    Parameters:
    run : MDA.Universe
        The simulation data loaded with MDAnalysis.
    cations_list : list
        List of cation residue groups.
    anions_list : list
        List of anion residue groups.
    times : np.array
        Array of time steps at which positions are recorded.
    run_start : int
        Starting step of the simulation from which to begin data collection.
    dt_collection : float
        The time step interval at which data is collected.

    Returns:
    tuple
        Two numpy arrays containing the positions of cations and anions across the specified times.
    """

    # Initialize the time counter to zero
    time = 0

    # Create arrays to store the positions of cations and anions. Each position is a 3D coordinate (x, y, z).
    cation_positions = np.zeros((len(times), len(cations_list), 3))
    anion_positions = np.zeros((len(times), len(anions_list), 3))

    # Iterate over each time step in the trajectory starting from 'run_start' adjusted by 'dt_collection'
    for ts in enumerate(tqdm(run.trajectory[int(run_start / dt_collection):])):
        # Calculate the center of mass of the entire system, considering periodic boundary conditions
        system_com = run.atoms.center_of_mass(wrap=True)

        # Compute the position of each cation relative to the system's center of mass and store in array
        for index, cation in enumerate(cations_list):
            cation_positions[time, index, :] = cation.center_of_mass() - system_com

        # Compute the position of each anion in the same way and store in array
        for index, anion in enumerate(anions_list):
            anion_positions[time, index, :] = anion.center_of_mass() - system_com

        # Increment the time index for the next step in the time series
        time += 1

    # Return the arrays of positions for cations and anions
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
    for atom_num in tqdm(range(n_atoms), desc="Calculating MSD"):
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
    log_time = np.log(times[1:])
    log_msd = np.log(msd[1:])

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
        for i, msd_ in enumerate(msd_data):
            mid_time = (time_ranges[i][1] + time_ranges[i][0]) / 2
            start = int(10 ** (np.log10(mid_time) - 0.15) / dt_)
            end = int(10 ** (np.log10(mid_time) + 0.15) / dt_)
            scale = (msd_[int(mid_time / dt_)] + 40) / mid_time

            x_log = times[start:end]
            y_log = x_log * scale

            ax.plot(times[1:], msd_[1:], '-', linewidth=1.5, color=color_list[i], label=labels[i])
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




