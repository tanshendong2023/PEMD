"""
PEMD code library.

Developed by: Tan Shendong
Date: 2024.03.15
"""


import os
import time
import re
import subprocess
import numpy as np
import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel as ob
from simple_slurm import Slurm
import PEMD.model.MD_lib as MDlib
from openbabel import openbabel
from rdkit.Chem import Descriptors
from networkx.algorithms import isomorphism
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


def count_atoms(mol, atom_type, length):
    # Initialize the counter for the specified atom type
    atom_count = 0
    # Iterate through all atoms in the molecule
    for atom in mol.GetAtoms():
        # Check if the atom is of the specified type
        if atom.GetSymbol() == atom_type:
            atom_count += 1
    return round(atom_count / length)


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


def orderxyz_energy_crest(file_path, numconf):
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

    # Get indices of the NumConf lowest energy structures
    lowest_indices = sorted(range(len(energies)), key=lambda i: energies[i])[:numconf]

    # Extract the structures with the lowest energies
    lowest_energy_structures = [structures[i] for i in lowest_indices]

    return lowest_energy_structures


def orderlog_energy_gaussian(dir_path):
    data = []
    # 遍历指定文件夹中的所有文件
    for file in os.listdir(dir_path):
        if file.endswith(".log"):
            log_file_path = os.path.join(dir_path, file)
            energy = read_G_from_gaussian(log_file_path)
            if energy is not None:
                # 将文件路径、文件名和能量值添加到数据列表中
                data.append({"File_Path": log_file_path, "Energy": float(energy)})

    # 将数据列表转换为DataFrame
    df = pd.DataFrame(data)

    # 找到能量最低的结构对应的行
    if not df.empty:
        sorted_df = df.sort_values(by='Energy', ascending=True)
        return sorted_df
    else:
        print(f"No sucessful log files found in {dir_path}")
        return None


def read_G_from_gaussian(log_file_path):
    energy = None
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
def smiles_xyz(unit_name, SMILES, ):
    try:
        m1 = Chem.MolFromSmiles(SMILES)    # Get mol(m1) from smiles
        m2 = Chem.AddHs(m1)   # Add H
        AllChem.Compute2DCoords(m2)    # Get 2D coordinates
        AllChem.EmbedMolecule(m2)    # Make 3D mol
        m2.SetProp("_Name", unit_name + '   ' + SMILES)    # Change title
        AllChem.UFFOptimizeMolecule(m2, maxIters=200)    # Optimize 3D str
        rdkitmol2xyz(unit_name, m2, '.', -1)
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


def Init_info(unit_name, smiles_mid, length, ):
    # Get index of dummy atoms and bond type associated with it
    dum_index, bond_type = FetchDum(smiles_mid)
    dum1 = dum_index[0]
    dum2 = dum_index[1]

    dum = None
    oligo_list = []
    unit_dis = None
    # Assign dummy atom according to bond type
    if bond_type == 'SINGLE':
        dum, unit_dis = 'Cl', -0.17
        # List of oligomers
        oligo_list = [length]
    elif bond_type == 'DOUBLE':
        dum, unit_dis = 'O', 0.25
        # List of oligomers
        oligo_list = []

    # Replace '*' with dummy atom
    smiles_each = smiles_mid.replace(r'*', dum)

    # Convert SMILES to XYZ coordinates
    convert_smiles2xyz, m1 = smiles_xyz(unit_name, smiles_each, )

    # Collect valency and connecting information for each atom
    neigh_atoms_info = connec_info(unit_name + '.xyz')

    # Find connecting atoms associated with dummy atoms.
    # dum1 and dum2 are connected to atom1 and atom2, respectively.
    atom1 = neigh_atoms_info['NeiAtom'][dum1].copy()[0]
    atom2 = neigh_atoms_info['NeiAtom'][dum2].copy()[0]

    return (unit_name, dum1, dum2, atom1, atom2, m1, smiles_each, neigh_atoms_info, oligo_list, dum, unit_dis, '',)


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
    convert_smiles2xyz, m1 = smiles_xyz(unit_name, smiles_each,)

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


def get_gaff2(unit_name, length, relax_polymer_lmp_dir, mol, atom_typing='pysimm'):
    print("\nGenerating GAFF2 parameter file ...\n")
    # r = MDlib.get_coord_from_pdb(outfile_name + ".pdb")
    # from pysimm import system, forcefield

    file_base = relax_polymer_lmp_dir + '/' + f'{unit_name}_N{length}'

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


def relax_polymer_lmp(unit_name, length, relax_polymer_lmp_dir, core):
    origin_dir = os.getcwd()
    os.chdir(relax_polymer_lmp_dir)
    # 创建LAMMPS输入文件字符串
    file_base = f'{unit_name}_N{length}'
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
    fix 1 all nvt temp 800 800 100 drag 2
    run 5000000
    write_data {0}_gaff2.data
    write_dump all xyz {0}_lmp.xyz
    """.format(file_base)

    # 将LAMMPS输入命令写入临时文件
    with open('lammps_input.in', "w") as f:
        f.write(lmp_commands)

    slurm = Slurm(J='lammps',
                  N=1,
                  n=f'{core}',
                  output=f'slurm.{Slurm.JOB_ARRAY_MASTER_ID}.out'
                  )

    slurm.add_cmd('module load LAMMPS')
    job_id = slurm.sbatch('mpirun lmp < lammps_input.in >out.lmp 2>lmp.err')
    while True:
        status = get_slurm_job_status(job_id)
        if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
            print("\n", unit_name, ": MD simulation normally terminated.\n")
            toxyz_lammps(f'{file_base}_lmp.xyz', f'{file_base}_gmx.xyz', f'{file_base}_gaff2.lmp')
            os.chdir(origin_dir)
            break
        else:
            print("polymer relax not finish, waiting...")
            time.sleep(10)


def mol_to_nx(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), element=atom.GetSymbol())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return G


def is_isomorphic(G1, G2):
    GM = isomorphism.GraphMatcher(G1, G2, node_match=lambda x, y: x['element'] == y['element'])
    return GM.is_isomorphic()


def get_closest_element_by_mass(target_mass):
    # 手动定义一个包含元素质量的映射
    element_masses = {
        'H': 1.008,  # 氢
        'C': 12.01,  # 碳
        'N': 14.007,  # 氮
        'O': 15.999,  # 氧
        'P': 30.974,  # 磷
        'S': 32.06,  # 硫
        'F': 18.998,  # 氟
        'Cl': 35.45,  # 氯
        'Br': 79.904,  # 溴
        'I': 126.90,  # 碘
        'Si': 28.085,  # 硅
        'B': 10.81,  # 硼
        'Al': 26.982,  # 铝
        'Se': 78.971,  # 硒
        'Zn': 65.38,  # 锌
        'Cu': 63.546,  # 铜
        'Mn': 54.938,  # 锰
        'Fe': 55.845,  # 铁
        'Co': 58.933,  # 钴
        'Ni': 58.693,  # 镍
        'Mo': 95.95,  # 钼
        'W': 183.84,  # 钨
        'Cr': 51.996,  # 铬
        'Ti': 47.867,  # 钛
        'V': 50.942,  # 钒
        # 可以根据需要添加更多元素
    }

    min_diff = np.inf
    closest_element = None

    for element, mass in element_masses.items():
        diff = abs(mass - target_mass)
        if diff < min_diff:
            min_diff = diff
            closest_element = element

    return closest_element


def parse_masses_from_lammps(data_filename):
    atom_map = {}
    with open(data_filename, 'r') as f:
        lines = f.readlines()

    start = lines.index("Masses\n") + 2
    end = start + lines[start:].index("\n")

    for line in lines[start:end]:
        parts = line.split()
        atom_id = int(parts[0])
        mass = float(parts[1])
        atom_symbol = get_closest_element_by_mass(mass)
        atom_map[atom_id] = atom_symbol

    return atom_map


def toxyz_lammps(input_filename, output_filename, data_filename):
    atom_map = parse_masses_from_lammps(data_filename)

    with open(input_filename, 'r') as fin, open(output_filename, 'w') as fout:
        for i, line in enumerate(fin):
            if i < 2:
                fout.write(line)
            else:
                parts = line.split()
                if parts[0].isdigit() and int(parts[0]) in atom_map:
                    parts[0] = atom_map[int(parts[0])]
                fout.write(' '.join(parts) + '\n')


def convert_xyz_to_pdb(xyz_filename, pdb_filename, molecule_name, resname):
    obConversion = openbabel.OBConversion()
    # 设置输入和输出格式
    obConversion.SetInAndOutFormats("xyz", "pdb")

    mol = openbabel.OBMol()
    # 读取 XYZ 文件
    obConversion.ReadFile(mol, xyz_filename)

    # 设置分子名称
    mol.SetTitle(molecule_name)

    # 遍历所有原子并设置自定义残基名称
    for atom in openbabel.OBMolAtomIter(mol):
        res = atom.GetResidue()
        res.SetName(resname)

    # 写入 PDB 文件，Open Babel 会尝试推断键的信息
    obConversion.WriteFile(mol, pdb_filename)


def convert_xyz_to_mol2(xyz_filename, mol2_filename, molecule_name, resname):
    obConversion = openbabel.OBConversion()
    # 设置输入格式为XYZ，输出格式为MOL2
    obConversion.SetInAndOutFormats("xyz", "mol2")

    mol = openbabel.OBMol()
    # 从XYZ文件中读取分子数据
    obConversion.ReadFile(mol, xyz_filename)

    # 设置分子名称，这会在MOL2文件中作为注释出现
    mol.SetTitle(molecule_name)

    # 遍历所有原子，设置自定义残基名称
    for atom in openbabel.OBMolAtomIter(mol):
        res = atom.GetResidue()
        if res:  # 确保残基信息存在
            res.SetName(resname)

    # 将分子数据写入MOL2文件
    obConversion.WriteFile(mol, mol2_filename)

    # 后处理：去除残基名称后的数字1
    remove_numbers_from_residue_names(mol2_filename, resname)


def remove_numbers_from_residue_names(mol2_filename, resname):
    with open(mol2_filename, 'r') as file:
        content = file.read()

    # 使用正则表达式删除特定残基名称后的数字1（确保只删除末尾的数字1）
    updated_content = re.sub(r'({})1\b'.format(resname), r'\1', content)

    with open(mol2_filename, 'w') as file:
        file.write(updated_content)


def extract_from_top(top_file, out_itp_file, nonbonded=False, bonded=False):
    sections_to_extract = []
    if nonbonded:
        sections_to_extract = ["[ atomtypes ]"]
    elif bonded:
        sections_to_extract = ["[ moleculetype ]", "[ atoms ]", "[ bonds ]", "[ pairs ]", "[ angles ]", "[ dihedrals ]"]

        # 打开 .top 文件进行读取
    with open(top_file, 'r') as file:
        lines = file.readlines()

    # 初始化变量以存储提取的信息
    extracted_lines = []
    current_section = None

    # 遍历所有行，提取相关部分
    for line in lines:
        if line.strip() in sections_to_extract:
            current_section = line.strip()
            extracted_lines.append(line)  # 添加部分标题
        elif current_section and line.strip().startswith(";"):
            extracted_lines.append(line)  # 添加注释行
        elif current_section and line.strip():
            extracted_lines.append(line)  # 添加数据行
        elif line.strip() == "" and current_section:
            extracted_lines.append("\n")  # 添加部分之间的空行
            current_section = None  # 重置当前部分

    # 写入提取的内容到 bonded.itp 文件
    with open(out_itp_file, 'w') as file:
        file.writelines(extracted_lines)

def mol_to_xyz(mol, conf_id, filename):
    """将RDKit分子对象的构象保存为XYZ格式文件"""
    xyz = Chem.MolToXYZBlock(mol, confId=conf_id)
    with open(filename, 'w') as f:
        f.write(xyz)

def read_energy_from_xtb(filename):
    """从xtb的输出文件中读取能量值"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    # 假设能量在输出文件的第二行
    energy_line = lines[1]
    energy = float(energy_line.split()[1])
    return energy

def std_xyzfile(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    structure_count = int(lines[0].strip())  # Assuming the first line contains the number of atoms

    # Process each structure in the file
    for i in range(0, len(lines), structure_count + 2):  # +2 for the atom count and energy lines
        # Copy the atom count line
        modified_lines.append(lines[i])

        # Extract and process the energy value line
        energy_line = lines[i + 1].strip()
        energy_value = energy_line.split()[1]  # Extract the energy value
        modified_lines.append(f" {energy_value}\n")  # Reconstruct the energy line with only the energy value

        # Copy the atom coordinate lines
        for j in range(i + 2, i + structure_count + 2):
            modified_lines.append(lines[j])

    # Overwrite the original file with modified lines
    with open(file_path, 'w') as file:
        file.writelines(modified_lines)


def log_to_xyz(log_file_path, xyz_file_path):
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("g09", "xyz")
    mol = openbabel.OBMol()

    try:
        obConversion.ReadFile(mol, log_file_path)
        obConversion.WriteFile(mol, xyz_file_path)
    except Exception as e:
        print(f"An error occurred during conversion: {e}")


def convert_chk_to_fchk(chk_file_path):

    fchk_file_path = chk_file_path.replace('.chk', '.fchk')

    try:
        subprocess.run(['formchk', chk_file_path, fchk_file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error convert chk to fchk: {e}")


def ave_end_chg(df, N):
    # 处理端部原子的电荷平均值
    top_N_df = df.head(N)
    tail_N_df = df.tail(N).iloc[::-1].reset_index(drop=True)
    average_charge = (top_N_df['charge'].reset_index(drop=True) + tail_N_df['charge']) / 2
    average_df = pd.DataFrame({
        'atom': top_N_df['atom'].reset_index(drop=True),  # 保持原子名称
        'charge': average_charge
    })
    return average_df


def ave_mid_chg(df, atom_count):
    # 处理中间原子的电荷平均值
    average_charges = []
    for i in range(atom_count):
        same_atoms = df[df.index % atom_count == i]
        avg_charge = same_atoms['charge'].mean()
        average_charges.append({'atom': same_atoms['atom'].iloc[0], 'charge': avg_charge})
    return pd.DataFrame(average_charges)


def read_sec_from_gmxitp_to_df(unit_name, out_dir, sec_name):

    itp_filepath = os.path.join(out_dir, f'{unit_name}_bonded.itp')

    with open(itp_filepath, 'r') as file:
        lines = file.readlines()

    data = []  # 用于存储数据行
    in_section = False  # 标记是否处于指定部分
    section_pattern = r'\[\s*.+\s*\]'  # 用于匹配部分标题的正则表达式
    columns = None  # 存储列名

    for line in lines:
        # 检测到指定部分的开始
        if line.strip() == sec_name:
            in_section = True
            continue

        # 使用正则表达式检测是否遇到了其他部分的标题行，标志着指定部分的结束
        if in_section and re.match(section_pattern, line.strip()):
            break

        # 处理列名行
        if in_section and not columns and line.startswith(';'):
            # 通常列名行以 ';' 开头，我们需要去除 ';' 并分割剩余字符串
            columns = re.sub(';', '', line).split()
            continue

        # 在指定部分内，且行不是注释或空行，则视为数据行
        if in_section and not line.startswith(';') and line.strip():
            # 分割数据行并添加到数据列表中
            data.append(line.split())

    # 如果有列名和数据，则创建 DataFrame
    if columns and data:
        df = pd.DataFrame(data, columns=columns)
    else:
        df = pd.DataFrame()

    return df


def xyz_to_df(xyz_file_path):
    # 初始化空列表来存储原子类型
    atoms = []

    # 读取XYZ文件
    with open(xyz_file_path, 'r') as file:
        next(file)  # 跳过第一行（原子总数）
        next(file)  # 跳过第二行（注释行）
        for line in file:
            atom_type = line.split()[0]  # 原子类型是每行的第一个元素
            atoms.append(atom_type)

    # 创建DataFrame
    df = pd.DataFrame(atoms, columns=['atom'])

    # 添加空的'charge'列
    df['charge'] = None  # 初始化为空值

    return df


def ave_chg_to_df(resp_chg_df, repeating_unit, num_repeating):

    # 处理非氢原子
    nonH_df = resp_chg_df[resp_chg_df['atom'] != 'H']

    cleaned_smiles = repeating_unit.replace('[*]', '')
    molecule = Chem.MolFromSmiles(cleaned_smiles)
    atom_count = molecule.GetNumAtoms()
    N = atom_count * num_repeating + 1  # 多一个端集CH3中的C

    # end_ave_chg_noH_df = ave_end_chg(nonH_df, N)
    top_N_noH_df = nonH_df.head(N)
    tail_N_noH_df = nonH_df.tail(N)
    mid_df_noH_df = nonH_df.drop(nonH_df.head(N).index.union(nonH_df.tail(N).index)).reset_index(drop=True)
    mid_ave_chg_noH_df = ave_mid_chg(mid_df_noH_df, atom_count)

    # 处理氢原子
    H_df = resp_chg_df[resp_chg_df['atom'] == 'H']

    molecule_with_h = Chem.AddHs(molecule)
    num_H_repeating = molecule_with_h.GetNumAtoms() - molecule.GetNumAtoms() - 2
    N_H = num_H_repeating * num_repeating + 3   # 多三个端集CH3中的H

    # end_ave_chg_H_df = ave_end_chg(H_df, N_H)
    top_N_H_df = H_df.head(N_H)
    tail_N_H_df = H_df.tail(N_H)
    mid_df_H_df = H_df.drop(H_df.head(N_H).index.union(H_df.tail(N_H).index)).reset_index(drop=True)
    mid_ave_chg_H_df = ave_mid_chg(mid_df_H_df, num_H_repeating)

    return top_N_noH_df, tail_N_noH_df, mid_ave_chg_noH_df, top_N_H_df, tail_N_H_df, mid_ave_chg_H_df


def calc_mol_weight(pdb_file):
    mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
    if mol is not None:
        mol_weight = Descriptors.MolWt(mol)
        return mol_weight
    else:
        raise ValueError(f"Unable to read molecular structure from {pdb_file}")


def smiles_to_pdb(smiles, output_file, molecule_name, resname):
    try:
        # Generate molecule object from SMILES string
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string. Please check the input string.")

        # Add hydrogens to the molecule
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        if AllChem.EmbedMolecule(mol, randomSeed=42) == -1:
            raise ValueError("Cannot embed the molecule into a 3D space.")
        AllChem.UFFOptimizeMolecule(mol)

        # Write molecule to a temporary SDF file
        tmp_sdf = "temp.sdf"
        with Chem.SDWriter(tmp_sdf) as writer:
            writer.write(mol)

        # Convert SDF to PDB using OpenBabel
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("sdf", "pdb")
        obmol = openbabel.OBMol()
        if not obConversion.ReadFile(obmol, tmp_sdf):
            raise IOError("Failed to read from the temporary SDF file.")

        # Set molecule name in OpenBabel
        obmol.SetTitle(molecule_name)

        # Set residue name for all atoms in the molecule in OpenBabel
        for atom in openbabel.OBMolAtomIter(obmol):
            res = atom.GetResidue()
            res.SetName(resname)

        if not obConversion.WriteFile(obmol, output_file):
            raise IOError("Failed to write the PDB file.")

        print(f"PDB file successfully created: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


def print_compounds(info_dict, key_name):
    """
    This function recursively searches for and prints 'compound' entries in a nested dictionary.
    """
    compounds = []
    for key, value in info_dict.items():
        # If the value is a dictionary, we make a recursive call
        if isinstance(value, dict):
            compounds.extend(print_compounds(value,key_name))
        # If the key is 'compound', we print its value
        elif key == key_name:
            compounds.append(value)
    return compounds


def extract_volume(partition, module_soft, edr_file, output_file='volume.xvg', option_id='21'):
    """
    使用GROMACS的gmx_mpi energy工具提取体积数据。此函数加载必要的模块，执行gmx_mpi命令，并处理输出。
    """
    # 构建命令字符串
    if partition == 'gpu':
        command = f"module load {module_soft} && echo {option_id} | gmx energy -f {edr_file} -o {output_file}"
    else:
        command = f"module load {module_soft} && echo {option_id} | gmx_mpi energy -f {edr_file} -o {output_file}"

    # 使用subprocess.run执行命令，由于这里使用bash -c，所以stdin的传递方式需要调整
    try:
        # Capture_output=True来捕获输出，而不是使用PIPE
        process = subprocess.run(['bash', '-c', command], capture_output=True, text=True, check=True)
        # 检查输出，无需单独检查returncode，因为check=True时如果命令失败会抛出异常
        print(f"Output: {process.stdout}")
        return process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.stderr}")
        return None


def read_volume_data(volume_file):
    """
    从 XVG 文件读取体积数据，忽略注释行，并从给定的开始时间 `start` 后收集数据。

    参数:
    - volume_file: 包含体积数据的 XVG 文件名。
    - start: 数据收集开始的时间（单位与 XVG 文件中的时间单位相同）。
    - dt_collection: 数据点之间的时间间隔。

    返回:
    - volumes: 从 `start` 时间开始的体积数据数组。
    """
    volumes = []
    with open(volume_file, 'r') as file:
        for line in file:
            if line.startswith(('@', '#')):
                continue
            parts = line.split()
            volumes.append(float(parts[1]))

    return np.array(volumes)


def analyze_volume(volumes, start, dt_collection):
    """
    计算并返回平均体积及最接近平均体积的帧索引。
    """
    start_time = int(start) / dt_collection
    average_volume = np.mean(volumes[int(start_time):])
    closest_index = np.argmin(np.abs(volumes - average_volume))
    return average_volume, closest_index


def extract_structure(partition, module_soft, tpr_file, xtc_file, save_gro_file, frame_time):
    """
    使用 GROMACS 的 gmx trjconv 工具从轨迹文件中提取特定时间点的结构。

    参数:
    - module_soft: 需要加载的模块名称。
    - tpr_file: TPR 输入文件路径。
    - xtc_file: XTC 轨迹文件路径。
    - save_gro_file: 输出的 GRO 文件路径。
    - frame_time: 需要提取的帧对应的时间（单位ps）。
    """
    # 构建完整的命令字符串
    if partition == 'gpu':
        command = (f"module load {module_soft} && echo 0 | gmx trjconv -s {tpr_file} -f {xtc_file} -o {save_gro_file} "
                   f"-dump {frame_time} -quiet")
    else:
        command = (f"module load {module_soft} && echo 0 | gmx_mpi trjconv -s {tpr_file} -f {xtc_file} -o {save_gro_file} "
                   f"-dump {frame_time} -quiet")

    # 使用 subprocess.run 执行命令，以更安全地处理外部命令
    try:
        # 使用 subprocess.run，避免使用shell=True以增强安全性
        process = subprocess.run(['bash', '-c', command], capture_output=True, text=True, check=True)
        print(f"Output: {process.stdout}")
        return process.stdout
    except subprocess.CalledProcessError as e:
        # 错误处理：打印错误输出并返回None
        print(f"Error executing command: {e.stderr}")
        return None



