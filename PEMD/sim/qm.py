"""
Polymer model building tools.

Developed by: Tan Shendong
Date: 2024.01.18
"""


import os
import re
import time
import glob
import subprocess
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from simple_slurm import Slurm
from PEMD.model import PEMD_lib
from PEMD.sim_API import gaussian
from PEMD.analysis import prop


def unit_conformer_search_crest(mol, unit_name, out_dir, length, numconf=10, core= 32, ):

    out_dir = out_dir + '/'
    PEMD_lib.build_dir(out_dir)

    mol2 = Chem.AddHs(mol)
    NAttempt = 100000

    cids = []
    for i in range(10):
        cids = AllChem.EmbedMultipleConfs(
            mol2,
            numConfs=10,
            numThreads=64,
            randomSeed=i,
            maxAttempts=NAttempt,
        )

        if len(cids) > 0:
            break

    cid = cids[0]
    AllChem.UFFOptimizeMolecule(mol2, confId=cid)

    file_base = '{}_N{}'.format(unit_name, length)
    pdb_file = os.path.join(out_dir, file_base + '.pdb')
    xyz_file = os.path.join(out_dir, file_base + '.xyz')

    Chem.MolToPDBFile(mol2, pdb_file, confId=cid)
    Chem.MolToXYZFile(mol2, xyz_file, confId=cid)

    crest_dir = os.path.join(out_dir, 'crest_work')
    os.makedirs(crest_dir, exist_ok=True)
    origin_dir = os.getcwd()
    os.chdir(crest_dir)

    xyz_file_path = os.path.join(origin_dir, xyz_file)

    slurm = Slurm(J='crest', N=1, n=f'{core}', output=f'slurm.{Slurm.JOB_ARRAY_MASTER_ID}.out')
    job_id = slurm.sbatch(f'crest {xyz_file_path} --gfn2 --T {core} --niceprint')
    time.sleep(10)

    while True:
        status = PEMD_lib.get_slurm_job_status(job_id)
        if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
            print("crest finish, executing the gaussian task...")
            order_structures = PEMD_lib.orderxyz_energy_crest('crest_conformers.xyz', numconf)
            break
        else:
            print("crest conformer search not finish, waiting...")
            time.sleep(30)

    os.chdir(origin_dir)
    return order_structures


def poly_conformer_search(mol, out_dir, unit_name, length, max_conformers=1000, top_n_MMFF=100, top_n_xtb=10,
                          epsilon=30, ):

    out_dir = out_dir + '/'
    PEMD_lib.build_dir(out_dir)

    # """从分子构象中搜索能量最低的构象
    mol = Chem.AddHs(mol)
    # 生成多个构象
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=max_conformers, randomSeed=1)
    props = AllChem.MMFFGetMoleculeProperties(mol)

    # 对每个构象进行能量最小化，并收集能量值
    minimized_conformers = []
    for conf_id in ids:
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        energy = ff.Minimize()
        minimized_conformers.append((conf_id, energy))

    # 按能量排序并选择前 top_n_MMFF 个构象
    minimized_conformers.sort(key=lambda x: x[1])
    top_conformers = minimized_conformers[:top_n_MMFF]

    xtb_dir = os.path.join(out_dir, 'xtb_work')
    os.makedirs(xtb_dir, exist_ok=True)
    origin_dir = os.getcwd()
    os.chdir(xtb_dir)

    for conf_id, _ in top_conformers:
        xyz_filename = f'conf_{conf_id}.xyz'
        output_filename = f'conf_{conf_id}_xtb.xyz'
        PEMD_lib.mol_to_xyz(mol, conf_id, xyz_filename)

        try:
            # 使用xtb进行进一步优化
            # xyz_file_path = os.path.join(origin_dir, xyz_filename)
            subprocess.run(['xtb', xyz_filename, '--opt', f'--gbsa={epsilon}', '--ceasefiles'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.rename('xtbopt.xyz', output_filename)
            PEMD_lib.std_xyzfile(output_filename)

        except subprocess.CalledProcessError as e:
            print(f'Error during optimization with xtb: {e}')

    # 匹配当前目录下所有后缀为xtb.xyz的文件
    filenames = glob.glob('*_xtb.xyz')
    # 输出文件名
    output_filename = 'merged_xtb.xyz'
    # 合并文件
    with open(output_filename, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                # 读取并写入文件内容
                outfile.write(infile.read())
    order_structures = PEMD_lib.orderxyz_energy_crest(output_filename, top_n_xtb)

    os.chdir(origin_dir)
    return order_structures


def conformer_search_gaussian(out_dir, structures, unit_name, charge=0, multiplicity=1, core = 32, memory= '64GB',
                              chk=True, opt_method='B3LYP', opt_basis='6-311+g(d,p)',dispersion_corr='em=GD3BJ',
                              freq='freq', solv_model='scrf=(pcm,solvent=generic,read)', custom_solv='eps=5.0 \nepsinf=2.1', ):

    current_directory = os.getcwd()
    job_ids = []
    structure_directory = current_directory + '/' + out_dir + '/' + f'{unit_name}_conf_g16'
    os.makedirs(structure_directory, exist_ok=True)

    for i, structure in enumerate(structures):

        # create xyz file
        file_path = os.path.join(structure_directory, f"{unit_name}_{i + 1}.xyz")

        with open(file_path, 'w') as file:
            for line in structure:
                file.write(f"{line}\n")

        gaussian.gaussian(files=file_path,
                          charge=f'{charge}',
                          mult=f'{multiplicity}',
                          suffix='',
                          prefix='',
                          program='gaussian',
                          mem=f'{memory}',
                          nprocs=f'{core}',
                          chk=chk,
                          qm_input=f'opt {freq} {opt_method} {opt_basis} {dispersion_corr} {solv_model}',
                          qm_end=f'{custom_solv}',
                          chk_path=structure_directory,
                          destination=structure_directory,
                          )

        slurm = Slurm(J='g16',
                      N=1,
                      n=f'{core}',
                      output=f'{structure_directory}/slurm.{Slurm.JOB_ARRAY_MASTER_ID}.out'
                      )

        job_id = slurm.sbatch(f'g16 {structure_directory}/{unit_name}_{i + 1}.gjf')
        time.sleep(10)
        job_ids.append(job_id)

    # check the status of the gaussian job
    while True:
        all_completed = True
        for job_id in job_ids:
            status = PEMD_lib.get_slurm_job_status(job_id)
            if status not in ['COMPLETED', 'FAILED', 'CANCELLED']:
                all_completed = False
                break
        if all_completed:
            print("All gaussian tasks finished, order structure with energy calculated by gaussian...")
            # order the structures by energy calculated by gaussian
            sorted_df = PEMD_lib.orderlog_energy_gaussian(structure_directory)
            break
        else:
            print("g16 conformer search not finish, waiting...")
            time.sleep(10)  # wait for 30 seconds
    return sorted_df


def calc_resp_gaussian(unit_name, length, out_dir, sorted_df, core=16, memory='64GB', eps=5.0, epsinf=2.1,):

    resp_dir = os.path.join(out_dir, 'resp_work')
    os.makedirs(resp_dir, exist_ok=True)

    log_file_path = sorted_df.iloc[0]['File_Path']
    chk_name = log_file_path.replace('.log', '.chk')

    # RESP template
    file_contents = f"nprocshared={core}\n"
    file_contents += f"%mem={memory}\n"
    file_contents += f"%oldchk={chk_name}\n"
    file_contents += f"%chk={resp_dir}/SP_gas.chk\n"
    file_contents += "# B3LYP/def2TZVP em=GD3BJ geom=allcheck\n\n"

    file_contents += "--link1--\n"
    file_contents += f"nprocshared={core}\n"
    file_contents += f"%mem={memory}\n"
    file_contents += f"%oldchk={chk_name}\n"
    file_contents += f"%chk={resp_dir}/SP_solv.chk\n"
    file_contents += f"# B3LYP/def2TZVP em=GD3BJ scrf=(pcm,solvent=generic,read) geom=allcheck\n\n"

    file_contents += f"eps={eps}\n"
    file_contents += f"epsinf={epsinf}\n\n"

    out_file = resp_dir + '/' + f'{unit_name}_resp.gjf'
    with open(out_file, 'w') as file:
        file.write(file_contents)

    structure_directory = os.getcwd() + '/' + resp_dir

    slurm = Slurm(J='g16',
                  N=1,
                  n=f'{core}',
                  output=f'{structure_directory}/slurm.{Slurm.JOB_ARRAY_MASTER_ID}.out'
                  )

    job_id = slurm.sbatch(f'g16 {structure_directory}/{unit_name}_resp.gjf')
    time.sleep(10)

    while True:
        status = PEMD_lib.get_slurm_job_status(job_id)
        if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
            print("RESP calculation finish, executing the resp fit with Multiwfn...")
            df = prop.RESP_fit_Multiwfn(unit_name, length, out_dir, method='resp',)
            break
        else:
            print("RESP calculation not finish, waiting...")
            time.sleep(10)
    return df


def apply_chg_to_gmx(unit_name, out_dir, length, resp_chg_df, repeating_unit, end_repeating, target_total_charge=0, correction_factor=1.0):

    (end_ave_chg_noH_df, mid_ave_chg_noH_df, end_ave_chg_H_df, mid_ave_chg_H_df) \
        = PEMD_lib.ave_chg_to_df(resp_chg_df, repeating_unit, end_repeating)

    relax_polymer_lmp_dir = os.path.join(out_dir, 'relax_polymer_lmp')

    xyz_file_path = os.path.join(relax_polymer_lmp_dir, f'{unit_name}_N{length}_gmx.xyz')
    atoms_chg_df = PEMD_lib.xyz_to_df(xyz_file_path)

    # deal with the head non-H atoms
    top_noH_df = end_ave_chg_noH_df
    tail_noH_df = top_noH_df.iloc[::-1].reset_index(drop=True)

    # deal with the mid non-H atoms
    atoms_chg_noH_df = atoms_chg_df[atoms_chg_df['atom'] != 'H']

    cleaned_smiles = repeating_unit.replace('[*]', '')
    molecule = Chem.MolFromSmiles(cleaned_smiles)
    atom_count = molecule.GetNumAtoms()
    N = atom_count * end_repeating

    mid_atoms_chg_noH_df = atoms_chg_noH_df.drop(
        atoms_chg_noH_df.head(N).index.union(atoms_chg_noH_df.tail(N).index)).reset_index(drop=True)

    # traverse the DataFrame of mid atoms
    for idx, row in mid_atoms_chg_noH_df.iterrows():
        # calculate the position of the current atom in the repeating unit
        position_in_cycle = idx % atom_count
        # find the average charge value of the atom at the corresponding position
        ave_chg_noH = mid_ave_chg_noH_df.iloc[position_in_cycle]['charge']
        # update the charge value
        mid_atoms_chg_noH_df.at[idx, 'charge'] = ave_chg_noH

    # deal with the head H atoms
    top_H_df = end_ave_chg_H_df
    tail_H_df = top_H_df.iloc[::-1].reset_index(drop=True)

    # deal with the mid H atoms
    atoms_chg_H_df = atoms_chg_df[atoms_chg_df['atom'] == 'H']

    molecule_with_h = Chem.AddHs(molecule)
    num_H_repeating = molecule_with_h.GetNumAtoms() - molecule.GetNumAtoms() - 2
    N_H = num_H_repeating * end_repeating + 1

    mid_atoms_chg_H_df = atoms_chg_H_df.drop(
        atoms_chg_H_df.head(N_H).index.union(atoms_chg_H_df.tail(N_H).index)).reset_index(drop=True)

    # traverse the DataFrame of mid atoms
    for idx, row in mid_atoms_chg_H_df.iterrows():
        # calculate the position of the current atom in the repeating unit
        position_in_cycle = idx % num_H_repeating
        # find the average charge value of the atom at the corresponding position
        avg_chg_H = mid_ave_chg_H_df.iloc[position_in_cycle]['charge']
        # update the charge value
        mid_atoms_chg_H_df.at[idx, 'charge'] = avg_chg_H

    charge_update_df = pd.concat([top_noH_df, mid_atoms_chg_noH_df, tail_noH_df, top_H_df, mid_atoms_chg_H_df,
                                tail_H_df], ignore_index=True)

    # charge neutralize and scale
    charge_update_df_cor = charge_neutralize_scale(charge_update_df, target_total_charge, correction_factor)

    itp_filepath = os.path.join(out_dir, 'MD_dir', f'{unit_name}_bonded.itp')

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

    # update the charge value in the [ atoms ] section
    charge_index = 0  # 用于跟踪DataFrame中当前的电荷索引
    for i in range(start_index, end_index):
        parts = lines[i].split()
        if charge_index < len(charge_update_df_cor):
            new_charge = charge_update_df_cor.iloc[charge_index]['charge']
            parts[6] = f'{new_charge:.8f}'  # 更新电荷值，假设电荷值在第7个字段
            lines[i] = ' '.join(parts) + '\n'
            charge_index += 1

    # save the updated itp file
    new_itp_filepath = os.path.join(out_dir, 'MD_dir',f'{unit_name}_bonded.itp')
    with open(new_itp_filepath, 'w') as file:
        file.writelines(lines)



def charge_neutralize_scale(df, target_total_charge, correction_factor):
    current_total_charge = df['charge'].sum()  # calculate the total charge of the current system
    charge_difference = target_total_charge - current_total_charge  # calculate the difference between the target and current total charge
    charge_adjustment_per_atom = charge_difference / len(df)  # calculate the charge adjustment per atom

    # update the charge value
    df['charge'] = (df['charge'] + charge_adjustment_per_atom) * correction_factor

    return df



















