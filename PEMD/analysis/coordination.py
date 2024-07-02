import os
import numpy as np
import MDAnalysis as mda
from tqdm.auto import tqdm
from MDAnalysis.analysis import rdf


def calculate_rdf_and_coordination(u, group1, group2, nbins=200, range_rdf=(0.0, 10.0)):
    # Initialize RDF analysis
    rdf_analysis = rdf.InterRDF(group1, group2, nbins=nbins, range=range_rdf)
    rdf_analysis.run()

    # Calculate coordination numbers
    volume = u.coord.volume
    rho = group2.n_atoms / volume  # Density of the second group

    bins = rdf_analysis.results.bins
    rdf_values = rdf_analysis.results.rdf
    coord_numbers = np.cumsum(4 * np.pi * bins**2 * rdf_values * np.diff(np.append(0, bins)) * rho)

    return bins, rdf_values, coord_numbers


def load_md_trajectory(work_dir, tpr_filename='nvt_prod.tpr', xtc_filename='nvt_prod.xtc'):
    data_tpr_file = os.path.join(work_dir, tpr_filename)
    data_xtc_file = os.path.join(work_dir, xtc_filename)
    u = mda.Universe(data_tpr_file, data_xtc_file)
    return u


def obtain_rdf_coord(bins, rdf, coord_numbers):
    # 计算一阶导数的符号变化
    deriv_sign_changes = np.diff(np.sign(np.diff(rdf)))
    # 寻找第一个峰值：导数从正变负
    peak_index = np.where(deriv_sign_changes < 0)[0] + 1  # +1 是因为diff的结果比原数组短1
    if len(peak_index) == 0:
        raise ValueError("No peak found in RDF data.")
    first_peak_index = peak_index[0]

    # 寻找第一个峰值后的第一个极小点
    min_after_peak_index = np.where(deriv_sign_changes[first_peak_index:] > 0)[0] + first_peak_index + 1
    if len(min_after_peak_index) == 0:
        raise ValueError("No minimum found after the first peak in RDF data.")
    first_min_index = min_after_peak_index[0]

    # 极小点处的坐标和配位数
    x_val = round(float(bins[first_min_index]), 3)  # 显式转换为 float 后四舍五入至三位小数
    y_rdf = round(float(rdf[first_min_index]), 3)   # 显式转换为 float 后四舍五入至三位小数
    y_coord = round(float(np.interp(x_val, bins, coord_numbers)), 3)  # 显式转换为 float 后四舍五入至三位小数

    return x_val, y_rdf, y_coord


def distance(x0, x1, box_length):
    """Calculate minimum image distance accounting for periodic boundary conditions."""
    delta = x1 - x0
    delta = np.where(delta > 0.5 * box_length, delta - box_length, delta)
    delta = np.where(delta < -0.5 * box_length, delta + box_length, delta)
    return delta


def analyze_coordination(universe, li_atoms, molecule_groups, cutoff_radii, run_start, run_end):
    num_timesteps = run_end - run_start
    num_li_atoms = len(li_atoms)
    coordination = np.zeros((num_timesteps, num_li_atoms), dtype=int)

    for ts_index, ts in enumerate(tqdm(universe.trajectory[run_start:run_end], desc='Processing')):
        box_size = ts.dimensions[0:3]
        for li_index, li in enumerate(li_atoms):
            encoded_coordination = 0
            factor = 10**(len(molecule_groups) - 1)  # Factor for encoding counts at different decimal places
            for group_name, group_atoms in molecule_groups.items():
                d_vec = distance(group_atoms.positions, li.position, box_size)
                d = np.linalg.norm(d_vec, axis=1)
                close_atoms_index = np.where(d < cutoff_radii[group_name])[0]
                unique_resids = len(np.unique(group_atoms[close_atoms_index].resids))
                encoded_coordination += unique_resids * factor
                factor //= 10  # Increment factor for the next group encoding
            coordination[ts_index, li_index] = encoded_coordination

    return coordination


