"""
Polymer model building tools.

Developed by: Tan Shendong
Date: 2024.03.26
"""


import os
import re
import pandas as pd
import parmed as pmd
from rdkit import Chem
from foyer import Forcefield
from PEMD.model import PEMD_lib
import importlib.resources as pkg_resources


def gen_gmx_oplsaa(unit_name, out_dir, length):

    current_path = os.getcwd()
    relax_polymer_lmp_dir = os.path.join(current_path, out_dir, 'relax_polymer_lmp')

    file_base = f"{unit_name}_N{length}"
    top_filename = os.path.join(relax_polymer_lmp_dir, f"{file_base}.top")
    gro_filename = os.path.join(relax_polymer_lmp_dir, f"{file_base}.gro")

    pdb_filename = None

    for file in os.listdir(relax_polymer_lmp_dir):
        if file.endswith(".xyz"):
            xyz_filename = os.path.join(relax_polymer_lmp_dir, f"{file_base}_gmx.xyz")
            pdb_filename = os.path.join(relax_polymer_lmp_dir, f"{file_base}_gmx.pdb")

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


def apply_chg_to_gmx(unit_name, out_dir, length, resp_chg_df, repeating_unit, num_repeating):

    (end_ave_chg_noH_df, mid_ave_chg_noH_df, end_ave_chg_H_df, mid_ave_chg_H_df) \
        = PEMD_lib.ave_chg_to_df(resp_chg_df, repeating_unit, num_repeating)

    relax_polymer_lmp_dir = os.path.join(out_dir, 'relax_polymer_lmp')

    xyz_file_path = os.path.join(relax_polymer_lmp_dir, f'{unit_name}_N{length}_gmx.xyz')
    atoms_chg_df = PEMD_lib.xyz_to_df(xyz_file_path)

    # 处理末端非氢原子
    top_noH_df = end_ave_chg_noH_df
    tail_noH_df = top_noH_df.iloc[::-1].reset_index(drop=True)

    # 处理mid非氢原子
    atoms_chg_noH_df = atoms_chg_df[atoms_chg_df['atom'] != 'H']

    cleaned_smiles = repeating_unit.replace('[*]', '')
    molecule = Chem.MolFromSmiles(cleaned_smiles)
    atom_count = molecule.GetNumAtoms()
    N = atom_count * num_repeating

    mid_atoms_chg_noH_df = atoms_chg_noH_df.drop(
        atoms_chg_noH_df.head(N).index.union(atoms_chg_noH_df.tail(N).index)).reset_index(drop=True)

    # 遍历中间原子的 DataFrame
    for idx, row in mid_atoms_chg_noH_df.iterrows():
        # 计算当前原子在周期单元中的位置
        position_in_cycle = idx % atom_count
        # 找到对应位置原子的平均电荷值
        ave_chg_noH = mid_ave_chg_noH_df.iloc[position_in_cycle]['charge']
        # 更新电荷值
        mid_atoms_chg_noH_df.at[idx, 'charge'] = ave_chg_noH

    # 处理末端氢原子
    top_H_df = end_ave_chg_H_df
    tail_H_df = top_H_df.iloc[::-1].reset_index(drop=True)

    # 处理mid氢原子
    atoms_chg_H_df = atoms_chg_df[atoms_chg_df['atom'] == 'H']

    molecule_with_h = Chem.AddHs(molecule)
    num_H_repeating = molecule_with_h.GetNumAtoms() - molecule.GetNumAtoms() - 2
    N_H = num_H_repeating * num_repeating + 1

    mid_atoms_chg_H_df = atoms_chg_H_df.drop(
        atoms_chg_H_df.head(N_H).index.union(atoms_chg_H_df.tail(N_H).index)).reset_index(drop=True)

    # 遍历中间原子的 DataFrame
    for idx, row in mid_atoms_chg_H_df.iterrows():
        # 计算当前原子在周期单元中的位置
        position_in_cycle = idx % num_H_repeating
        # 找到对应位置原子的平均电荷值
        avg_chg_H = mid_ave_chg_H_df.iloc[position_in_cycle]['charge']
        # 更新电荷值
        mid_atoms_chg_H_df.at[idx, 'charge'] = avg_chg_H

    charge_update_df = pd.concat([top_noH_df, mid_atoms_chg_noH_df, tail_noH_df, top_H_df, mid_atoms_chg_H_df,
                                tail_H_df], ignore_index=True)

    itp_filepath = os.path.join(out_dir, f'{unit_name}_bonded.itp')

    # 读取.itp文件
    with open(itp_filepath, 'r') as file:
        lines = file.readlines()

    # 找到[ atoms ]部分的开始和结束
    in_section = False  # 标记是否处于指定部分
    section_pattern = r'\[\s*.+\s*\]'  # 用于匹配部分标题的正则表达式
    start_index = end_index = 0
    for i, line in enumerate(lines):
        if line.startswith('[ atoms ]'):
            start_index = i + 3  # 跳过部分标题和列标题
            in_section = True
            continue
        elif in_section and re.match(section_pattern, line.strip()):
            end_index = i - 1
            break

    # 更新电荷，这里假设charge_update_df中的电荷顺序与.itp文件中的原子顺序一致
    charge_index = 0  # 用于跟踪DataFrame中当前的电荷索引
    for i in range(start_index, end_index):
        parts = lines[i].split()
        if charge_index < len(charge_update_df):
            new_charge = charge_update_df.iloc[charge_index]['charge']
            parts[6] = f'{new_charge:.8f}'  # 更新电荷值，假设电荷值在第7个字段
            lines[i] = ' '.join(parts) + '\n'
            charge_index += 1

    # 保存为新的.itp文件
    new_itp_filepath = os.path.join(out_dir, f'{unit_name}_bonded_updated.itp')
    with open(new_itp_filepath, 'w') as file:
        file.writelines(lines)
















