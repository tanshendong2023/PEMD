import os
import numpy as np
import matplotlib.pyplot as plt
from PEMD.analysis import coordination


work_dir = './'
data_tpr_file='nvt_prod.tpr'
dcd_xtc_file='nvt_prod.xtc'

# Load the trajectory
u = coordination.load_md_trajectory(work_dir, data_tpr_file, dcd_xtc_file)

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

# Example usage of the function
run_start = 0
run_end = 80001

# Select the atoms of interest
target_groups = {
    'PEO': peo_atoms,
    'TFSI': tfsi_atoms,
}

# setting the first solvation shell distance
cutoff_radii = {
    'PEO': r_li_peo,
    'TFSI': r_li_tfsi,
}

# analysis the coordination for targed compound
coord = coordination.analyze_coordination(u, li_atoms, target_groups, cutoff_radii, run_start, run_end)

# Calculate the distribution as percentages
unique, counts = np.unique(coord, return_counts=True)
total_counts = (run_end - run_start) * len(li_atoms)
percentages = np.round((counts / total_counts) * 100, 2)

distribution = dict(zip(unique, percentages))

# 使用repr()将字典转换为字符串
frequencies_str = repr(distribution)

# 写入文件
file_path = 'coordination.txt'
with open(file_path, 'w') as file:
    file.write(frequencies_str)



