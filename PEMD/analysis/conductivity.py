import numpy as np
from tqdm.auto import tqdm
from PEMD.analysis import msd


def calc_cond_msd(run, cations, anions, run_start, ):

    # Split atoms into lists by residue for cations and anions
    cations_list = cations.atoms.split("residue")
    anions_list = anions.atoms.split("residue")

    # compute sum over all charges and positions
    qr = []
    for _ts in tqdm(run.trajectory[run_start:], desc='Calculating conductivity'):
        qr_temp = np.zeros(3)
        for cation in cations_list:
            qr_temp += cation.center_of_mass() * int(1)
        for anion in anions_list:
            qr_temp += anion.center_of_mass() * int(-1)
        qr.append(qr_temp)
    return msd.msd_fft(np.array(qr))

def calculate_slope_msd(times_array, msd_array, dt_collection, dt, interval_time=5000, step_size=20):
    # 对数变换
    log_time = np.log(times_array[1:])
    log_msd = np.log(msd_array[1:])

    # 计算时间间隔
    dt_ = dt_collection * dt
    interval_msd = int(interval_time / dt_)

    # 初始化列表存储每个大间隔的平均斜率
    average_slopes = []
    closest_slope = float('inf')
    time_range = (None, None)

    # 使用滑动窗口计算每个大间隔的平均斜率
    for i in range(0, len(log_time) - interval_msd, step_size):
        if i + interval_msd > len(log_time):  # 确保不越界
            break
        # 使用 polyfit 计算一阶线性拟合的斜率
        coeffs = np.gradient(log_msd[i:i + interval_msd], log_time[i:i + interval_msd])
        slope = np.mean(coeffs)
        average_slopes.append(slope)

        # 更新最接近1的平均斜率及其范围
        if abs(slope - 1) < abs(closest_slope - 1):
            closest_slope = slope
            time_range = (times_array[i], times_array[i + interval_msd])

    # 计算最终斜率
    if time_range[0] is not None and time_range[1] is not None:
        final_slope = (msd_array[int(time_range[1] / dt_)] - msd_array[int(time_range[0] / dt_)]) / (time_range[1] - time_range[0])
    else:
        final_slope = None  # 无法计算斜率

    return final_slope, time_range

def calculate_conductivity(slope, v, T):
    # 从斜率计算电导率
    A2cm = 1e-8  # Angstroms to cm
    ps2s = 1e-12  # picoseconds to seconds
    e2c = 1.60217662e-19  # elementary charge to Coulomb
    kb = 1.38064852e-23  # Boltzmann Constant, J/K
    convert = e2c * e2c / ps2s / A2cm * 1000

    cond = slope / 6 / kb / T / v * convert   # "mS/cm"

    return cond





