"""
PEMD code library.

Developed by: Tan Shendong
Date: 2024.03.15
"""

import os
import time
import shutil
import subprocess
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel as ob
from LigParGenPEMD import Converter
from PEMD.sim_API.gaussian import gaussian
from simple_slurm import Slurm
import PEMD.MD_lib as MDlib
from pysimm import system, lmps, forcefield


# OpenBabel setup
obConversion = ob.OBConversion()
ff = ob.OBForceField.FindForceField('UFF')
mol = ob.OBMol()
np.set_printoptions(precision=20)


# simple function
# This function try to create a directory
def build_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        pass


def is_nan(x):
    return x != x


def get_slurm_job_status(job_id):
    command = f'sacct -j {job_id} --format=State --noheader'
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Split the output by newlines and strip whitespace
    statuses = [line.strip() for line in process.stdout.strip().split('\n')]
    # Check if all statuses indicate the job is completed
    if all(status == 'COMPLETED' for status in statuses):
        return 'COMPLETED'
    elif any(status == 'FAILED' for status in statuses):
        return 'FAILED'
    elif any(status == 'CANCELLED' for status in statuses):
        return 'CANCELLED'
    else:
        return 'RUNNING'


def conformer_search(unit_name, ln, NumConf, working_dir):

    mol_file = unit_name + '_N' + str(ln)

    slurm = Slurm(J='crest',
                  N=1,
                  n=32,
                  output=f'slurm.{Slurm.JOB_ARRAY_MASTER_ID}.out'
                  )

    job_id = slurm.sbatch(f'crest {mol_file}.xyz --gfn2 --T 32 --niceprint')

    # 检查文件是否存在
    while True:
        status = get_slurm_job_status(job_id)
        if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
            print("crest finish, executing the gaussian task...")
            # 保存能量最低的n个结构为列表，并生成gaussian输入文件
            lowest_energy_structures = crest_lowest_energy_str('crest_conformers.xyz', NumConf)
            save_structures(lowest_energy_structures, 'PEO')
            break  # 任务执行完毕后跳出循环
        else:
            print("crest conformer search not finish, waiting...")
            time.sleep(60)  # 等待60秒后再次检查


def parse_xyz_with_energies(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    structures = []
    energies = []
    current_structure = []
    for line in lines:
        if line.strip().isdigit() and current_structure:  # 结构的开始
            energy_line = current_structure[1]  # 第二行是能量
            energy = float(energy_line.split()[-1])
            energies.append(energy)
            structures.append(current_structure)
            current_structure = [line.strip()]  # 开始新的结构
        else:
            current_structure.append(line.strip())

    # 添加文件中的最后一个结构
    if current_structure:
        energy_line = current_structure[1]
        energy = float(energy_line.split()[-1])
        energies.append(energy)
        structures.append(current_structure)

    return structures, energies


def save_structures(structures, base_filename):
    # 获取当前工作目录的路径
    current_directory = os.getcwd()
    job_ids = []

    for i, structure in enumerate(structures):
        # 为每个结构创建一个新目录
        structure_directory = os.path.join(current_directory, f"{base_filename}_{i + 1}")
        os.makedirs(structure_directory, exist_ok=True)

        # 在新创建的目录中保存XYZ文件
        file_path = os.path.join(structure_directory, f"{base_filename}_{i + 1}.xyz")

        with open(file_path, 'w') as file:
            for line in structure:
                file.write(f"{line}\n")

        gaussian(files=file_path,
                 qm_input='opt freq B3LYP/6-311+g(d,p) nosymm em=GD3BJ',
                 suffix='',
                 prefix='',
                 program='gaussian',
                 mem='64GB',
                 nprocs=32,
                 chk=True,
                 chk_path=structure_directory,
                 destination=structure_directory,
                 )

        slurm = Slurm(J='g16',
                      N=1,
                      n=32,
                      output=f'{structure_directory}/slurm.{Slurm.JOB_ARRAY_MASTER_ID}.out'
                      )

        # com_file = os.path.join(structure_directory, f"{base_filename}_{i + 1}_conf_1.com")
        #         print(f'g16 {structure_directory}/{base_filename}_{i+1}_conf_1.com')
        job_id = slurm.sbatch(f'g16 {structure_directory}/{base_filename}_{i + 1}_conf_1.com')
        job_ids.append(job_id)

    # 检查所有任务的状态
    while True:
        all_completed = True
        for job_id in job_ids:
            status = get_slurm_job_status(job_id)
            if status not in ['COMPLETED', 'FAILED', 'CANCELLED']:
                all_completed = False
                break

        if all_completed:
            print("All gaussian tasks finished, find the lowest energy structure...")
            # 执行下一个任务的代码...
            g16_lowest_energy_str(file_path='./')
            break
        else:
            print("g16 conformer search not finish, waiting...")
            time.sleep(60)  # 等待360秒后再次检查


def crest_lowest_energy_str(file_path, NumConf):
    structures, energies = parse_xyz_with_energies(file_path)

    # 获取能量最低的n个结构的索引
    lowest_indices = sorted(range(len(energies)), key=lambda i: energies[i])[:NumConf]

    # 提取这些结构
    lowest_energy_structures = [structures[i] for i in lowest_indices]

    return lowest_energy_structures


def g16_lowest_energy_str(file_path = './'):
    data = []
    for root, dirs, files in os.walk(file_path):
        for dir_name in dirs:
            if dir_name.startswith("PEO"):

                dir_path = os.path.join(root, dir_name)
                # print(dir_path)
                for file in os.listdir(dir_path):
                    if file.endswith(".log"):
                        log_file_path = os.path.join(dir_path, file)
                        energy = read_log_file(log_file_path)
                        if energy is not None:
                            data.append({"Directory": dir_path, "File": file, "Energy": float(
                                energy)})  # Append the directory, file name, and energy value to the data list

    df = pd.DataFrame(data)
    min_str = df.loc[df['Energy'].idxmin()]

    # Renaming the directory with the lowest energy structure
    lowest_energy_dir = min_str['Directory']
    os.rename(lowest_energy_dir, "PEO_lowest_confor")

    # Deleting other directories
    for index, row in df.iterrows():
        dir_path = row['Directory']
        if dir_path != "PEO_conformer":
            shutil.rmtree(dir_path, ignore_errors=True)


def read_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        for line in file:
            if "Sum of electronic and thermal Free Energies=" in line:
                energy = float(line.split()[-1])
                break
    return energy


def rdkitmol2xyz(unit_name, m, dir_xyz, IDNum):
    try:
        Chem.MolToXYZFile(m, dir_xyz + unit_name + '.xyz', confId=IDNum)
    except Exception:
        obConversion.SetInAndOutFormats("mol", "xyz")
        Chem.MolToMolFile(m, dir_xyz + unit_name + '.mol', confId=IDNum)
        mol = ob.OBMol()
        obConversion.ReadFile(mol, dir_xyz + unit_name + '.mol')
        obConversion.WriteFile(mol, dir_xyz + unit_name + '.xyz')


# This function generates a VASP input (polymer) file
# INPUT: name of VASP directory, name of a monomer, XYZ-coordinates, row numbers for dummy and
# connecting atoms , chemical name of dummy atom, Serial number
# OUTPUT: Generates a VASP input file
def gen_vasp(vasp_dir, unit_name, unit, dum1, dum2, atom1, atom2, dum, unit_dis, SN=0, length=0, Inter_Chain_Dis=12, Polymer=False,):

    add_dis = add_dis_func(unit, atom1, atom2)

    unit = trans_origin(unit, atom2)
    unit = alignZ(unit, atom2, dum1)
    unit = unit.sort_values(by=[0])

    if SN == 0 and length == 0:
        file_name = vasp_dir + unit_name.replace('.xyz', '') + '.vasp'
    elif SN == 0 and length != 0:
        file_name = (
            vasp_dir + unit_name.replace('.xyz', '') + '_N' + str(length) + '.vasp'
        )
    elif SN != 0 and length == 0:
        file_name = vasp_dir + unit_name.replace('.xyz', '') + '_C' + str(SN) + '.vasp'
    else:
        file_name = (vasp_dir + unit_name.replace('.xyz', '') + '_N' + str(length) + '_C' + str(SN) + '.vasp')

    file = open(file_name, 'w+')
    file.write('### ' + str(unit_name) + ' ###\n')
    file.write('1\n')

    # Get the size of the box
    a_vec = unit[1].max() - unit[1].min() + Inter_Chain_Dis
    b_vec = unit[2].max() - unit[2].min() + Inter_Chain_Dis

    if Polymer:
        c_vec = unit.loc[dum1][3] + unit_dis + add_dis  #
    else:
        c_vec = unit[3].max() - unit[3].min() + Inter_Chain_Dis

    # move unit to the center of a box
    unit[1] = unit[1] - unit[1].min() + Inter_Chain_Dis / 2
    unit[2] = unit[2] - unit[2].min() + Inter_Chain_Dis / 2

    if Polymer:
        unit[3] = unit[3] + (1.68 + unit_dis + add_dis) / 2
    else:
        unit[3] = unit[3] - unit[3].min() + Inter_Chain_Dis / 2

    unit = unit.drop([dum1, dum2])
    file.write(' ' + str(a_vec) + ' ' + str(0.0) + ' ' + str(0.0) + '\n')
    file.write(' ' + str(0.0) + ' ' + str(b_vec) + ' ' + str(0.0) + '\n')
    file.write(' ' + str(0.0) + ' ' + str(0.0) + ' ' + str(c_vec) + '\n')

    ele_list = []
    count_ele_list = []
    for element in sorted(set(unit[0].values)):
        ele_list.append(element)
        count_ele_list.append(list(unit[0].values).count(element))

    for item in ele_list:
        file.write(str(item) + '  ')

    file.write('\n ')
    for item in count_ele_list:
        file.write(str(item) + ' ')

    file.write('\nCartesian\n')

    file.write(unit[[1, 2, 3]].to_string(header=False, index=False))
    file.close()


# This function create XYZ files from SMILES
# INPUT: ID, SMILES, directory name
# OUTPUT: xyz files in 'work_dir', result = DONE/NOT DONE, mol without Hydrogen atom
def smiles_xyz(unit_name, SMILES, dir_xyz):
    try:
        m1 = Chem.MolFromSmiles(SMILES)    # Get mol(m1) from smiles
        m2 = Chem.AddHs(m1)   # Add H
        AllChem.Compute2DCoords(m2)    # Get 2D coordinates
        AllChem.EmbedMolecule(m2)    # Make 3D mol
        m2.SetProp("_Name", unit_name + '   ' + SMILES)    # Change title
        AllChem.UFFOptimizeMolecule(m2, maxIters=200)    # Optimize 3D str
        rdkitmol2xyz(unit_name, m2, dir_xyz, -1)
        result = 'DONE'
    except Exception:
        result, m1 = 'NOT_DONE', ''
    return result, m1


# This function indentifies row numbers of dummy atoms
# INPUT: SMILES
# OUTPUT: row indices of dummy atoms and nature of bond with connecting atom
def FetchDum(smiles):
    m = Chem.MolFromSmiles(smiles)
    dummy_index = []
    if m is not None:
        for atom in m.GetAtoms():
            if atom.GetSymbol() == '*':
                dummy_index.append(atom.GetIdx())
        for bond in m.GetBonds():
            if (
                bond.GetBeginAtom().GetSymbol() == '*'
                or bond.GetEndAtom().GetSymbol() == '*'
            ):
                bond_type = bond.GetBondType()
                break
    return dummy_index, str(bond_type)


# Connection information obtained by OpenBabel
# INPUT: XYZ file
# OUTPUT: Connectivity information
def connec_info(unit_name):
    obConversion = ob.OBConversion()
    obConversion.SetInFormat("xyz")
    mol = ob.OBMol()
    obConversion.ReadFile(mol, unit_name)
    neigh_atoms_info = []

    for atom in ob.OBMolAtomIter(mol):
        neigh_atoms = []
        bond_orders = []
        for allatom in ob.OBAtomAtomIter(atom):
            neigh_atoms.append(allatom.GetIndex())
            bond_orders.append(atom.GetBond(allatom).GetBondOrder())
        neigh_atoms_info.append([neigh_atoms, bond_orders])
    neigh_atoms_info = pd.DataFrame(neigh_atoms_info, columns=['NeiAtom', 'BO'])
    return neigh_atoms_info



# This function estimate distance between two repeating units
# INPUT: XYZ coordinates, row numbers for connecting atoms
# OUTPUT: Distance
def add_dis_func(unit, atom1, atom2):
    add_dis = 0.0
    if unit.loc[atom1][0] == 'C' and unit.loc[atom2][0] == 'N':
        add_dis = -0.207
    elif unit.loc[atom1][0] == 'N' and unit.loc[atom2][0] == 'N':
        add_dis = -0.4
    elif unit.loc[atom1][0] == 'C' and unit.loc[atom2][0] == 'O':
        add_dis = -0.223
    elif unit.loc[atom1][0] == 'O' and unit.loc[atom2][0] == 'O':
        add_dis = -0.223
    return add_dis


# Align a molecule on Z-axis wrt two atoms
# INPUT: XYZ-coordinates, row numbers of two atoms
# OUTPUT: A new sets of XYZ-coordinates
def alignZ(unit, atom1, atom2):
    dis_zx = np.sqrt(
        (unit.iloc[atom1].values[3] - unit.iloc[atom2].values[3]) ** 2
        + (unit.iloc[atom1].values[1] - unit.iloc[atom2].values[1]) ** 2
    )
    angle_zx = (np.arccos(unit.iloc[atom2].values[3] / dis_zx)) * 180.0 / np.pi
    if unit.iloc[atom2].values[1] > 0.0:  # or angle_zx < 90.0: # check and improve
        angle_zx = -angle_zx
    unit = rotateXZ(unit, angle_zx)

    dis_zy = np.sqrt(
        (unit.iloc[atom1].values[3] - unit.iloc[atom2].values[3]) ** 2
        + (unit.iloc[atom1].values[2] - unit.iloc[atom2].values[2]) ** 2
    )
    angle_zy = (np.arccos(unit.iloc[atom2].values[3] / dis_zy)) * 180.0 / np.pi
    if unit.iloc[atom2].values[2] > 0.0:  # or angle_zy < 90.0: # need to improve
        angle_zy = -angle_zy

    unit = rotateYZ(unit, angle_zy)
    return unit


# Rotate on ZY plane
# INPUT: XYZ-coordinates and angle in Degree
# OUTPUT: A new sets of XYZ-coordinates
def rotateYZ(unit, theta):  # XYZ coordinates and angle
    R_z = np.array(
        [
            [np.cos(theta * np.pi / 180.0), -np.sin(theta * np.pi / 180.0)],
            [np.sin(theta * np.pi / 180.0), np.cos(theta * np.pi / 180.0)],
        ]
    )
    oldXYZ = unit[[2, 3]].copy()
    XYZcollect = []
    for eachatom in np.arange(oldXYZ.values.shape[0]):
        rotate_each = oldXYZ.iloc[eachatom].values.dot(R_z)
        XYZcollect.append(rotate_each)
    newXYZ = pd.DataFrame(XYZcollect)
    unit[[2, 3]] = newXYZ[[0, 1]]
    return unit


# Rotate on XZ plane
# INPUT: XYZ-coordinates and angle in Degree
# OUTPUT: A new sets of XYZ-coordinates
def rotateXZ(unit, theta):  # XYZ coordinates and angle
    R_z = np.array(
        [
            [np.cos(theta * np.pi / 180.0), -np.sin(theta * np.pi / 180.0)],
            [np.sin(theta * np.pi / 180.0), np.cos(theta * np.pi / 180.0)],
        ]
    )
    oldXYZ = unit[[1, 3]].copy()
    XYZcollect = []
    for eachatom in np.arange(oldXYZ.values.shape[0]):
        rotate_each = oldXYZ.iloc[eachatom].values.dot(R_z)
        XYZcollect.append(rotate_each)
    newXYZ = pd.DataFrame(XYZcollect)
    unit[[1, 3]] = newXYZ[[0, 1]]
    return unit


# Translation to origin
# INPUT: XYZ-coordinates and row number of an atom which will be moved to the origin.
# OUTPUT: A new sets of XYZ-coordinates
def trans_origin(unit, atom1):  # XYZ coordinates and angle
    unit[1] = unit[1] - (unit.iloc[atom1][1])
    unit[2] = unit[2] - (unit.iloc[atom1][2])
    unit[3] = unit[3] - (unit.iloc[atom1][3])
    return unit


# complex function
def Init_info(unit_name, smiles_each_ori, length):
    # Get index of dummy atoms and bond type associated with it
    try:
        dum_index, bond_type = FetchDum(smiles_each_ori)
        if len(dum_index) == 2:
            dum1 = dum_index[0]
            dum2 = dum_index[1]
        else:
            print(
                unit_name,
                ": There are more or less than two dummy atoms in the SMILES string; "
                "Hint: PSP works only for one-dimensional polymers.",
            )
            return unit_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'REJECT'
    except Exception:
        print(
            unit_name,
            ": Couldn't fetch the position of dummy atoms. Hints: (1) In SMILES strings, use '*' for a dummy atom,"
            "(2) Check RDKit installation.",
        )
        return unit_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'REJECT'

    # Assign dummy atom according to bond type
    if bond_type == 'SINGLE':
        dum, unit_dis = 'Cl', -0.17
        # List of oligomers
        oligo_list = list(set(length) - set(['n']))
    elif bond_type == 'DOUBLE':
        dum, unit_dis = 'O', 0.25
        # List of oligomers
        oligo_list = []
    else:
        print(
            unit_name,
            ": Unusal bond type (Only single or double bonds are acceptable)."
            "Hints: (1) Check bonds between the dummy and connecting atoms in SMILES string"
            "       (2) Check RDKit installation.",
        )
        return unit_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'REJECT'

    # Replace '*' with dummy atom
    smiles_each = smiles_each_ori.replace(r'*', dum)

    # Convert SMILES to XYZ coordinates
    convert_smiles2xyz, m1 = smiles_xyz(unit_name, smiles_each, './')

    # if fails to get XYZ coordinates; STOP
    if convert_smiles2xyz == 'NOT_DONE':
        print(
            unit_name,
            ": Couldn't get XYZ coordinates from SMILES string. Hints: (1) Check SMILES string,"
            "(2) Check RDKit installation.",
        )
        return unit_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'REJECT'

    # Collect valency and connecting information for each atom
    neigh_atoms_info = connec_info('./' + unit_name + '.xyz')

    try:
        # Find connecting atoms associated with dummy atoms.
        # dum1 and dum2 are connected to atom1 and atom2, respectively.
        atom1 = neigh_atoms_info['NeiAtom'][dum1].copy()[0]
        atom2 = neigh_atoms_info['NeiAtom'][dum2].copy()[0]

    except Exception:
        print(
            unit_name,
            ": Couldn't get the position of connecting atoms. Hints: (1) XYZ coordinates are not acceptable,"
            "(2) Check Open Babel installation.",
        )
        return unit_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'REJECT'

    return (unit_name, dum1, dum2, atom1, atom2, m1, neigh_atoms_info, oligo_list, dum, unit_dis, '',)


def gen_oligomer_smiles(unit_name, dum1, dum2, atom1, atom2, smiles_each, ln, smiles_LCap_, LCap_, smiles_RCap_, RCap_,):
    input_mol = Chem.MolFromSmiles(smiles_each)
    edit_m1 = Chem.EditableMol(input_mol)

    edit_m1.RemoveAtom(dum1)

    if dum1 < dum2:
        edit_m1.RemoveAtom(dum2 - 1)

    else:
        edit_m1.RemoveAtom(dum2)

    monomer_mol = edit_m1.GetMol()
    inti_mol = monomer_mol

    if atom1 > atom2:
        atom1, atom2 = atom2, atom1

    if dum1 < atom1 and dum2 < atom1:
        first_atom = atom1 - 2
    elif (dum1 < atom1 and dum2 > atom1) or (dum1 > atom1 and dum2 < atom1):
        first_atom = atom1 - 1
    else:
        first_atom = atom1

    if dum1 < atom2 and dum2 < atom2:
        second_atom = atom2 - 2
    elif (dum1 < atom2 and dum2 > atom2) or (dum1 > atom2 and dum2 < atom2):
        second_atom = atom2 - 1
    else:
        second_atom = atom2

    for i in range(1, ln):
        combo = Chem.CombineMols(inti_mol, monomer_mol)
        edcombo = Chem.EditableMol(combo)
        edcombo.AddBond(
            second_atom + (i - 1) * monomer_mol.GetNumAtoms(),
            first_atom + i * monomer_mol.GetNumAtoms(),
            order=Chem.rdchem.BondType.SINGLE,
        )

        inti_mol = edcombo.GetMol()

    if LCap_ is True or RCap_ is True:
        inti_mol = gen_smiles_with_cap(unit_name,0,0, first_atom, second_atom + i * monomer_mol.GetNumAtoms(),
                                       inti_mol, smiles_LCap_, smiles_RCap_, LCap_, RCap_, WithDum=False,)

        return inti_mol

    return Chem.MolToSmiles(inti_mol)


def gen_smiles_with_cap(unit_name, dum1, dum2, atom1, atom2, smiles_each, smiles_LCap_, smiles_RCap_, LCap_, RCap_, WithDum=True,):
    # Main chain
    # Check if there are dummy atoms in the chain
    if WithDum is True:
        main_mol = Chem.MolFromSmiles(smiles_each)
        main_edit_m1 = Chem.EditableMol(main_mol)

        # Remove dummy atoms
        main_edit_m1.RemoveAtom(dum1)
        if dum1 < dum2:
            main_edit_m1.RemoveAtom(dum2 - 1)
        else:
            main_edit_m1.RemoveAtom(dum2)

        # Mol without dummy atom
        main_mol_noDum = main_edit_m1.GetMol()

        # Get linking atoms
        if atom1 > atom2:
            atom1, atom2 = atom2, atom1

        if dum1 < atom1 and dum2 < atom1:
            first_atom = atom1 - 2
        elif (dum1 < atom1 and dum2 > atom1) or (dum1 > atom1 and dum2 < atom1):
            first_atom = atom1 - 1
        else:
            first_atom = atom1

        if dum1 < atom2 and dum2 < atom2:
            second_atom = atom2 - 2
        elif (dum1 < atom2 and dum2 > atom2) or (dum1 > atom2 and dum2 < atom2):
            second_atom = atom2 - 1
        else:
            second_atom = atom2
    else:
        main_mol_noDum = smiles_each
        first_atom, second_atom = atom1, atom2

    LCap_add = 0
    # Left Cap
    if LCap_ is True:
        (unit_name, dum_L, atom_L, m1L, neigh_atoms_info_L, flag_L) = Init_info_Cap(
            unit_name, smiles_LCap_
        )

        # Reject if SMILES is not correct
        if flag_L == 'REJECT':
            return unit_name, 'REJECT', 0

        # Editable Mol for LeftCap
        LCap_m1 = Chem.MolFromSmiles(smiles_LCap_)
        LCap_edit_m1 = Chem.EditableMol(LCap_m1)

        # Remove dummy atoms
        LCap_edit_m1.RemoveAtom(dum_L)

        # Mol without dummy atom
        LCap_m1 = LCap_edit_m1.GetMol()
        LCap_add = LCap_m1.GetNumAtoms()

        # Linking atom
        if dum_L < atom_L:
            LCap_atom = atom_L - 1
        else:
            LCap_atom = atom_L

        # Join main chain with Left Cap
        combo = Chem.CombineMols(LCap_m1, main_mol_noDum)
        edcombo = Chem.EditableMol(combo)
        edcombo.AddBond(
            LCap_atom, first_atom + LCap_add, order=Chem.rdchem.BondType.SINGLE
        )
        main_mol_noDum = edcombo.GetMol()

    # Right Cap
    if RCap_ is True:
        (unit_name, dum_R, atom_R, m1L, neigh_atoms_info_R, flag_R) = Init_info_Cap(
            unit_name, smiles_RCap_
        )

        # Reject if SMILES is not correct
        if flag_R == 'REJECT':
            return unit_name, 'REJECT', 0

        # Editable Mol for RightCap
        RCap_m1 = Chem.MolFromSmiles(smiles_RCap_)
        RCap_edit_m1 = Chem.EditableMol(RCap_m1)

        # Remove dummy atoms
        RCap_edit_m1.RemoveAtom(dum_R)

        # Mol without dummy atom
        RCap_m1 = RCap_edit_m1.GetMol()

        # Linking atom
        if dum_R < atom_R:
            RCap_atom = atom_R - 1
        else:
            RCap_atom = atom_R

        # Join main chain with Left Cap
        combo = Chem.CombineMols(main_mol_noDum, RCap_m1)
        edcombo = Chem.EditableMol(combo)
        edcombo.AddBond(
            LCap_add + second_atom,
            RCap_atom + main_mol_noDum.GetNumAtoms(),
            order=Chem.rdchem.BondType.SINGLE,
        )
        main_mol_noDum = edcombo.GetMol()
    return Chem.MolToSmiles(main_mol_noDum)


def Init_info_Cap(unit_name, smiles_each_ori):
    # Get index of dummy atoms and bond type associated with it
    try:
        dum_index, bond_type = FetchDum(smiles_each_ori)
        if len(dum_index) == 1:
            dum1 = dum_index[0]
        else:
            print(
                unit_name,
                ": There are more or less than one dummy atoms in the SMILES string; ",
            )
            return unit_name, 0, 0, 0, 0, 'REJECT'
    except Exception:
        print(
            unit_name,
            ": Couldn't fetch the position of dummy atoms. Hints: (1) In SMILES string, use '*' for a dummy atom,"
            "(2) Check RDKit installation.",
        )
        return unit_name, 0, 0, 0, 0, 'REJECT'

    # Replace '*' with dummy atom
    smiles_each = smiles_each_ori.replace(r'*', 'Cl')

    # Convert SMILES to XYZ coordinates
    convert_smiles2xyz, m1 = smiles_xyz(unit_name, smiles_each, './')

    # if fails to get XYZ coordinates; STOP
    if convert_smiles2xyz == 'NOT_DONE':
        print(
            unit_name,
            ": Couldn't get XYZ coordinates from SMILES string. Hints: (1) Check SMILES string,"
            "(2) Check RDKit installation.",
        )
        return unit_name, 0, 0, 0, 0, 'REJECT'

    # Collect valency and connecting information for each atom
    neigh_atoms_info = connec_info('./' + unit_name + '.xyz')

    try:
        # Find connecting atoms associated with dummy atoms.
        # dum1 and dum2 are connected to atom1 and atom2, respectively.
        atom1 = neigh_atoms_info['NeiAtom'][dum1].copy()[0]

    except Exception:
        print(
            unit_name,
            ": Couldn't get the position of connecting atoms. Hints: (1) XYZ coordinates are not acceptable,"
            "(2) Check Open Babel installation.",
        )
        return unit_name, 0, 0, 0, 0, 'REJECT'
    return (
        unit_name,
        dum1,
        atom1,
        m1,
        neigh_atoms_info,
        '',
    )


def gen_conf_xyz_vasp(unit_name, m1, out_dir, ln, OPLS, polymer, atom_typing_):
    m2 = Chem.AddHs(m1)
    NAttempt = 100000

    for i in range(10):
        cids = AllChem.EmbedMultipleConfs(
            m2,
            numConfs=10,
            numThreads=64,
            randomSeed=i,
            maxAttempts=NAttempt,
        )

        if len(cids) > 0:
            break

    cid = cids[0]
    AllChem.UFFOptimizeMolecule(m2, confId=cid)
    # AllChem.MMFFOptimizeMolecule(m2, confId=cid)

    # 使用 os.path.join 来正确地拼接路径和文件名
    file_base = '{}_N{}'.format(unit_name, ln)
    pdb_filename = os.path.join(out_dir, file_base + '.pdb')
    xyz_filename = os.path.join(out_dir, file_base + '.xyz')

    Chem.MolToPDBFile(m2, pdb_filename, confId=cid)  # Generate pdb file
    Chem.MolToXYZFile(m2, xyz_filename, confId=cid)  # Generate xyz file

    Chem.MolToPDBFile(m2, pdb_filename, confId=cid)  # Generate pdb file
    Chem.MolToXYZFile(m2, xyz_filename, confId=cid)  # Generate xyz file

    # outfile_name = out_dir + unit_name + '_N' + str(ln) + '_C' + str(n)

    # if IrrStruc is False:
    # Chem.MolToPDBFile(m2, outfile_name + '.pdb', confId=cid)  # Generate pdb file
    # Chem.MolToXYZFile(m2, outfile_name + '.xyz', confId=cid)  # Generate xyz file

    # Generate OPLS parameter file
    if OPLS is True:
        print(unit_name, ": Generating OPLS parameter file ...")
        if os.path.exists(pdb_filename):
            try:
                Converter.convert(
                    pdb=pdb_filename,
                    resname=unit_name,
                    charge=0,
                    opt=0,
                    outdir= out_dir,
                    ln = ln,
                )
                print(unit_name, ": OPLS parameter file generated.")
            except BaseException:
                print('problem running LigParGen for {}.pdb.'.format(pdb_filename))
    else:
        os.chdir(out_dir)

    if polymer is True:
        print("\n", unit_name, ": Performing a short MD simulation using LAMMPS...\n", )
        get_gaff2(file_base, out_dir, atom_typing=atom_typing_)
        # input_file = file_base + '_gaff2.lmp'
        # output_file = file_base + '_gaff2.data'
        relax_polymer_lmp(file_base)
        print("\n", unit_name, ": MD simulation normally terminated.\n")


def get_gaff2(file_base, mol, atom_typing='pysimm'):
    print("\nGenerating GAFF2 parameter file ...\n")
    # r = MDlib.get_coord_from_pdb(outfile_name + ".pdb")
    # from pysimm import system, forcefield

    obConversion.SetInAndOutFormats("pdb", "cml")
    if os.path.exists(file_base + '.pdb'):
        mol = ob.OBMol()
        obConversion.ReadFile(mol, file_base + '.pdb')
    else:
        try:
            count_atoms = mol.NumAtoms()
            if count_atoms > 0:
                pass
            else:
                print("ERROR: Number of atoms = ", count_atoms)
                print("Couldn't generate GAFF2 parameter file\n")
                return
        except BaseException:
            print("ERROR: pdb file not found; OBMol not provided")
            print("Couldn't generate GAFF2 parameter file\n")
            return

    obConversion.WriteFile(mol, file_base + '.cml')
    data_fname = file_base + '_gaff2.lmp'

    try:
        print("Pysimm working on {}".format(file_base + '.mol2'))
        s = system.read_cml(file_base + '.cml')

        f = forcefield.Gaff2()
        if atom_typing == 'pysimm':
            for b in s.bonds:
                if b.a.bonds.count == 3 and b.b.bonds.count == 3:
                    b.order = 4
            s.apply_forcefield(f, charges='gasteiger')
        elif atom_typing == 'antechamber':
            obConversion.SetOutFormat("mol2")
            obConversion.WriteFile(mol, file_base + '.mol2')
            print("Antechamber working on {}".format(file_base + '.mol2'))
            MDlib.get_type_from_antechamber(s, file_base + '.mol2', 'gaff2', f)
            s.pair_style = 'lj'
            s.apply_forcefield(f, charges='gasteiger', skip_ptypes=True)
        else:
            print('Invalid atom typing option, please select pysimm or antechamber.')
        s.write_lammps(data_fname)
        print("\nGAFF2 parameter file generated.")
    except BaseException:
        print('problem reading {} for Pysimm.'.format(file_base + '.cml'))


def disorder_struc(filename, dir_path, NCores_opt):

    # pdb to cml
    obConversion.SetInAndOutFormats("pdb", "cml")
    obConversion.ReadFile(mol, os.path.join(dir_path, filename + '.pdb'))
    obConversion.WriteFile(mol, os.path.join(dir_path, filename + '.cml'))

    # MD simulation followed by opt
    scml = system.read_cml(os.path.join(dir_path, filename + '.cml'))
    scml.apply_forcefield(forcefield.Gaff2())
    lmps.quick_min(scml, np=NCores_opt, etol=1.0e-5, ftol=1.0e-5)
    lmps.quick_md(scml, np=NCores_opt, ensemble='nvt', timestep=0.5, run=100000)

    # Write files
    scml.write_xyz(os.path.join(dir_path, filename + '.xyz'))
    scml.write_pdb(os.path.join(dir_path, filename + '.pdb'))


def gen_dimer_smiles(dum1, dum2, atom1, atom2, input_smiles):
    input_mol = Chem.MolFromSmiles(input_smiles)
    edit_m1 = Chem.EditableMol(input_mol)
    edit_m2 = Chem.EditableMol(input_mol)

    edit_m1.RemoveAtom(dum1)
    edit_m2.RemoveAtom(dum2)

    edit_m1_mol = edit_m1.GetMol()
    edit_m2_mol = edit_m2.GetMol()

    if dum1 < atom1:
        first_atom = atom1 - 1
    else:
        first_atom = atom1

    if dum2 < atom2:
        second_atom = atom2 - 1
    else:
        second_atom = atom2 + edit_m1_mol.GetNumAtoms()

    combo = Chem.CombineMols(edit_m1_mol, edit_m2_mol)
    edcombo = Chem.EditableMol(combo)
    edcombo.AddBond(first_atom, second_atom, order=Chem.rdchem.BondType.SINGLE)
    combo_mol = edcombo.GetMol()
    return Chem.MolToSmiles(combo_mol)


def find_best_conf(unit_name, m1, dum1, dum2, atom1, atom2, xyz_in_dir):
    m2 = Chem.AddHs(m1)
    cids = AllChem.EmbedMultipleConfs(m2, numConfs=100)
    cid_list = []
    for cid in cids:
        AllChem.UFFOptimizeMolecule(m2, confId=cid)
        conf = m2.GetConformer(cid)
        ffu = AllChem.UFFGetMoleculeForceField(m2, confId=cid)
        cid_list.append(
            [
                cid,
                abs(
                    Chem.rdMolTransforms.GetDihedralDeg(
                        conf, int(dum1), int(atom1), int(atom2), int(dum2)
                    )
                ),
                ffu.CalcEnergy(),
            ]
        )
    cid_list = pd.DataFrame(cid_list, columns=['cid', 'Dang', 'Energy'])
    cid_list = cid_list.sort_values(by=['Dang'], ascending=False)
    cid_list = cid_list[
        cid_list['Dang'] > int(cid_list.head(1)['Dang'].values[0]) - 8.0
    ]
    cid_list = cid_list.sort_values(by=['Energy'], ascending=True)

    rdkitmol2xyz(unit_name, m2, xyz_in_dir, int(cid_list.head(1)['cid'].values[0]))


# This function minimize molecule using UFF forcefield and Steepest Descent method
# INPUT: ID, path and name of XYZ file, row indices of dummy and connecting atoms, name of working directory
# OUTPUT: XYZ coordinates of the optimized molecule
def localopt(unit_name, file_name, dum1, dum2, atom1, atom2, xyz_tmp_dir):
    constraints = ob.OBFFConstraints()
    obConversion.SetInAndOutFormats("xyz", "xyz")
    obConversion.ReadFile(mol, file_name)
    for atom_id in [dum1 + 1, dum2 + 1, atom1 + 1, atom2 + 1]:
        constraints.AddAtomConstraint(atom_id)

    # Set the constraints
    ff.Setup(mol, constraints)
    ff.SteepestDescent(5000)
    ff.UpdateCoordinates(mol)
    obConversion.WriteFile(mol, xyz_tmp_dir + unit_name + '_opt.xyz')

    # Check Connectivity
    neigh_atoms_info_old = connec_info(file_name)
    neigh_atoms_info_new = connec_info(xyz_tmp_dir + unit_name + '_opt.xyz')
    for row in neigh_atoms_info_old.index.tolist():
        if sorted(neigh_atoms_info_old.loc[row]['NeiAtom']) != sorted(
            neigh_atoms_info_new.loc[row]['NeiAtom']
        ):
            unit_opt = pd.read_csv(
                file_name, header=None, skiprows=2, delim_whitespace=True
            )
            return unit_opt
        else:
            # read XYZ file: skip the first two rows
            unit_opt = pd.read_csv(
                xyz_tmp_dir + unit_name + '_opt.xyz',
                header=None,
                skiprows=2,
                delim_whitespace=True,
            )
            return unit_opt


def relax_polymer_lmp(file_base):
    # 创建LAMMPS输入文件字符串
    lmp_commands = """
    units real
    boundary s s s
    dimension 3
    atom_style full
    bond_style harmonic
    angle_style harmonic
    dihedral_style fourier
    improper_style harmonic
    pair_style lj/cut 3.0
    read_data {0}_gaff2.lmp
    thermo 100
    thermo_style custom step temp pxx pyy pzz ebond eangle edihed eimp epair ecoul evdwl pe ke etotal lx ly lz vol density
    fix xwalls all wall/reflect xlo EDGE xhi EDGE
    fix ywalls all wall/reflect ylo EDGE yhi EDGE
    fix zwalls all wall/reflect zlo EDGE zhi EDGE
    velocity all create 300 3243242
    minimize 1e-8 1e-8 10000 10000
    dump 1 all custom 500 soft.lammpstrj id type mass mol x y z
    dump dump_xyz all xyz 1000 {0}_lmp.xyz
    fix 1 all nvt temp 800 800 100 drag 2
    run 50000000
    write_data {0}_gaff2.data
    """.format(file_base)

    # 将LAMMPS输入命令写入临时文件
    with open("lammps_input.in", "w") as f:
        f.write(lmp_commands)

    slurm = Slurm(J='lammps',
                  N=1,
                  n=32,
                  output=f'slurm.{Slurm.JOB_ARRAY_MASTER_ID}.out'
                  )

    slurm.add_cmd('module load LAMMPS')
    job_id = slurm.sbatch('mpirun lmp < lammps_input.in >out.lmp 2>lmp.err')

    # while True:
    #     status = get_slurm_job_status(job_id)
    #     if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
    #         print("lammps finish, executing the other task...")
    #         # 保存能量最低的n个结构为列表，并生成gaussian输入文件
    #
    #         break  # 任务执行完毕后跳出循环
    #     else:
    #         print("crest conformer search not finish, waiting...")
    #         time.sleep(60)  # 等待60秒后再次检查















