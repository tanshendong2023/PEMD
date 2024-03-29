"""
Polymer model building tools.

Developed by: Tan Shendong
Date: 2024.03.26
"""


import os
from foyer import Forcefield
import parmed as pmd
from PEMD.model import PEMD_lib
from PEMD.sim import qm
import importlib.resources as pkg_resources


def gen_gmx_oplsaa(unit_name, out_dir, length):
    current_path = os.getcwd()
    filepath = current_path + '/' + out_dir

    top_filename = filepath + '/' + f'{unit_name}{length}.top'
    gro_filename = filepath + '/' + f'{unit_name}{length}.gro'

    pdb_filename = None

    for file in os.listdir(filepath):
        if file.endswith(".xyz"):
            file_base = '{}_N{}'.format(unit_name, length)
            xyz_filename = out_dir + '/' + f'{file_base}_gmx.xyz'
            pdb_filename = out_dir + '/' + f'{file_base}_gmx.pdb'

            PEMD_lib.convert_xyz_to_pdb(xyz_filename, pdb_filename, f'{unit_name}', f'{unit_name}')

    untyped_str = pmd.load_file(pdb_filename, structure=True)

    with pkg_resources.path("PEMD.sim", "oplsaa.xml") as oplsaa_path:
        oplsaa = Forcefield(forcefield_files=str(oplsaa_path))
    typed_str = oplsaa.apply(untyped_str)

    # Save to any format supported by ParmEd
    typed_str.save(top_filename)
    typed_str.save(gro_filename)

    nonbonditp_filename = out_dir + '/' + f'{unit_name}_nonbonded.itp'
    bonditp_filename = out_dir + '/' + f'{unit_name}_bonded.itp'

    PEMD_lib.extract_from_top(top_filename, nonbonditp_filename, nonbonded=True, bonded=False)

    PEMD_lib.extract_from_top(top_filename, bonditp_filename, nonbonded=False, bonded=True)

    try:
        os.remove(top_filename)
    except Exception:
        pass  # 忽略任何异常

    try:
        os.remove(gro_filename)
    except Exception:
        pass  # 忽略任何异常


def apply_chg_to_gmx(itp_file, resp_chg_df, repeating_unit, num_repeating):

    (end_ave_chg_noH_df, mid_ave_chg_noH_df, end_ave_chg_H_df, mid_ave_chg_H_df) \
        = qm.ave_chg_to_df(resp_chg_df, repeating_unit, num_repeating)

    atoms_df = PEMD_lib.read_sec_from_gmxitp_to_df(itp_file, '[ atoms ]')
    atoms_chg_df = atoms_df[['atom','charge']]

    # 更新非氢原子的电荷
    for idx, row in end_ave_chg_noH_df.iterrows():
        atom_name = row['Atom']
        charge = row['Charge']
        atoms_chg_df.loc[atoms_chg_df['atom'] == atom_name, 'charge'] = charge






    # # 读取力场原始文件
    # bonditp_file = f'{unit_name}_bonded.itp'
    # with open(bonditp_file, 'r') as f:
    #     lines = f.readlines()
    #
    # with open(bonditp_file_update, 'w') as f:
    #     atom_section = False
    #     for line in lines:
    #         # 检查是否到达了[ atoms ]部分
    #         if line.startswith('[ atoms ]'):
    #             atom_section = True
    #             f.write(line)  # 写入[ atoms ]行
    #             continue
    #
    #         # 如果在[ atoms ]部分，而且行不是空行或注释行
    #         if atom_section and not line.startswith(';') and not line.strip() == '':
    #             split_line = line.split()
    #             # 根据原子序号找到对应的电荷值
    #             atom_idx = int(split_line[0]) - 1  # DataFrame 索引从0开始，而GROMACS从1开始
    #             charge = df.at[atom_idx, 'charge']
    #             # 重写含有电荷值的行
    #             new_line = f"{'  '.join(split_line[:6])}  {charge: .8f}  {'  '.join(split_line[7:])}\n"
    #             file.write(new_line)
    #         elif atom_section and (line.startswith(';') or line.strip() == ''):
    #             # 如果到达了[ atoms ]部分的末尾（即遇到了注释行或空行）
    #             atom_section = False
    #             file.write(line)  # 写入注释行或空行
    #         else:
    #             # 对于其他所有行，直接写入文件
    #             file.write(line)






