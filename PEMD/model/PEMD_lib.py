"""
PEMD code library.

Developed by: Tan Shendong
Date: 2024.03.15
"""


import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel as ob
from simple_slurm import Slurm
import PEMD.model.MD_lib as MDlib
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



def crest_lowest_energy_str(file_path, numconf):
    # Parse XYZ file to obtain structures and their corresponding energies
    structures, energies = parse_xyz_with_energies(file_path)

    # Get indices of the NumConf lowest energy structures
    lowest_indices = sorted(range(len(energies)), key=lambda i: energies[i])[:numconf]

    # Extract the structures with the lowest energies
    lowest_energy_structures = [structures[i] for i in lowest_indices]

    return lowest_energy_structures


def g16_lowest_energy_str(dir_path, unit_name, ln):
    data = []
    # 遍历指定文件夹中的所有文件
    for file in os.listdir(dir_path):
        if file.endswith(".log"):
            log_file_path = os.path.join(dir_path, file)
            energy = read_log_file(log_file_path)
            if energy is not None:
                # 将文件路径、文件名和能量值添加到数据列表中
                data.append({"File_Path": log_file_path, "Energy": float(energy)})

    # 将数据列表转换为DataFrame
    df = pd.DataFrame(data)

    # 找到能量最低的结构对应的行
    if not df.empty:
        min_energy_row = df.loc[df['Energy'].idxmin()]

        # 获取最低能量结构的文件路径
        lowest_energy_file_path = min_energy_row['File_Path']

        # 构造新的文件名，带有 'lowest_energy_' 前缀
        new_file_name = f"{unit_name}_N{ln}_lowest.log"

        # 复制文件到新的文件名
        shutil.copy(lowest_energy_file_path, os.path.join(dir_path, new_file_name))


def read_log_file(log_file_path):
    with open(log_file_path, 'r') as file:
        for line in file:
            if "Sum of electronic and thermal Free Energies=" in line:
                energy = float(line.split()[-1])
                break
    return energy


def rdkitmol2xyz(unit_name, m, out_dir, IDNum):
    try:
        Chem.MolToXYZFile(m, out_dir + '/' + unit_name + '.xyz', confId=IDNum)
    except Exception:
        obConversion.SetInAndOutFormats("mol", "xyz")
        Chem.MolToMolFile(m, out_dir + '/' + unit_name + '.mol', confId=IDNum)
        mol = ob.OBMol()
        obConversion.ReadFile(mol, out_dir + '/' + unit_name + '.mol')
        obConversion.WriteFile(mol, out_dir + '/' + unit_name + '.xyz')


# This function create XYZ files from SMILES
# INPUT: ID, SMILES, directory name
# OUTPUT: xyz files in 'work_dir', result = DONE/NOT DONE, mol without Hydrogen atom
def smiles_xyz(unit_name, SMILES, out_dir):
    try:
        m1 = Chem.MolFromSmiles(SMILES)    # Get mol(m1) from smiles
        m2 = Chem.AddHs(m1)   # Add H
        AllChem.Compute2DCoords(m2)    # Get 2D coordinates
        AllChem.EmbedMolecule(m2)    # Make 3D mol
        m2.SetProp("_Name", unit_name + '   ' + SMILES)    # Change title
        AllChem.UFFOptimizeMolecule(m2, maxIters=200)    # Optimize 3D str
        rdkitmol2xyz(unit_name, m2, out_dir, -1)
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
def Init_info(unit_name, smiles_each_ori, length, out_dir = './'):
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
                "Hint: PEMD works only for one-dimensional polymers.",
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
        oligo_list = [length]
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
    convert_smiles2xyz, m1 = smiles_xyz(unit_name, smiles_each, out_dir)

    # if fails to get XYZ coordinates; STOP
    if convert_smiles2xyz == 'NOT_DONE':
        print(
            unit_name,
            ": Couldn't get XYZ coordinates from SMILES string. Hints: (1) Check SMILES string,"
            "(2) Check RDKit installation.",
        )
        return unit_name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'REJECT'

    # Collect valency and connecting information for each atom
    neigh_atoms_info = connec_info(out_dir + '/' + unit_name + '.xyz')

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


def gen_oligomer_smiles(unit_name, dum1, dum2, atom1, atom2, smiles_each, length, smiles_LCap_, LCap_, smiles_RCap_, RCap_,):
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

    for i in range(1, length):
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


def get_gaff2(unit_name, length, out_dir, mol, atom_typing='pysimm'):
    print("\nGenerating GAFF2 parameter file ...\n")
    # r = MDlib.get_coord_from_pdb(outfile_name + ".pdb")
    # from pysimm import system, forcefield

    file_base = out_dir + "/" + '{}_N{}'.format(unit_name, length)

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


def relax_polymer_lmp(unit_name, length, out_dir):
    origin_dir = os.getcwd()
    os.chdir(out_dir)
    # 创建LAMMPS输入文件字符串
    file_base = '{}_N{}'.format(unit_name, length)
    lmp_commands = """
    units real
    boundary s s s
    dimension 3
    atom_style full
    bond_style harmonic
    angle_style harmonic
    dihedral_style fourier
    improper_style harmonic
    pair_style lj/cut 2.0
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
    run 5000000
    write_data {0}_gaff2.data
    """.format(file_base)

    # 将LAMMPS输入命令写入临时文件
    with open('lammps_input.in', "w") as f:
        f.write(lmp_commands)

    slurm = Slurm(J='lammps',
                  N=1,
                  n=32,
                  output=f'slurm.{Slurm.JOB_ARRAY_MASTER_ID}.out'
                  )

    slurm.add_cmd('module load LAMMPS')
    job_id = slurm.sbatch('mpirun lmp < lammps_input.in >out.lmp 2>lmp.err')
    os.chdir(origin_dir)

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















