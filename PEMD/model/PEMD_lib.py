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
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import openbabel as ob
from simple_slurm import Slurm
import PEMD.model.MD_lib as MDlib
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


def g16_lowest_energy_str(dir_path, unit_name,length):
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
        new_file_name = f"{unit_name}_N{length}_lowest.log"

        # 复制文件到新的文件名
        shutil.copy(lowest_energy_file_path, os.path.join(dir_path, new_file_name))


def read_log_file(log_file_path):
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


def get_gaff2(unit_name, length, out_dir, mol, atom_typing='pysimm'):
    print("\nGenerating GAFF2 parameter file ...\n")
    # r = MDlib.get_coord_from_pdb(outfile_name + ".pdb")
    # from pysimm import system, forcefield

    file_base = out_dir + '{}_N{}'.format(unit_name, length)

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


def relax_polymer_lmp(unit_name, length, out_dir, core):
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
            file_base = '{}_N{}'.format(unit_name, length)
            toxyz_lammps(f'{file_base}_lmp.xyz', f'{file_base}_gmx.xyz', f'{file_base}_gaff2.lmp')
            os.chdir(origin_dir)
            break
        else:
            print("polymer relax not finish, waiting...")
            time.sleep(30)


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
        'H': 1.008,
        'C': 12.01,
        'N': 14.007,
        'O': 15.999,
        'P': 30.974,
        'S': 32.06,
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










