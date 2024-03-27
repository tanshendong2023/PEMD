"""
Polymer model building tools.

Developed by: Tan Shendong
Date: 2024.01.18
"""


import os
import random
import threading
import itertools
import pandas as pd
import datamol as dm
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import Descriptors
from IPython.display import display
from PEMD.model import PEMD_lib
from LigParGenPEMD import Converter


def mol_from_smiles(unit_name, repeating_unit, leftcap, rightcap, length):

    input_data = [[unit_name, repeating_unit, leftcap, rightcap]]
    df_smiles = pd.DataFrame(input_data, columns=['ID', 'SMILES', 'LeftCap', 'RightCap'])

    # Extract cap SMILES if available
    smiles_LCap_ = df_smiles.loc[df_smiles['ID'] == unit_name, 'LeftCap'].values[0]
    LCap_ = not pd.isna(smiles_LCap_)

    smiles_RCap_ = df_smiles.loc[df_smiles['ID'] == unit_name, 'RightCap'].values[0]
    RCap_ = not pd.isna(smiles_RCap_)

    # Get repeating unit SMILES
    smiles_mid = df_smiles.loc[df_smiles['ID'] == unit_name, 'SMILES'].values[0]

    smiles_poly = None
    if length == 1:

        if not LCap_ and not RCap_:
            mol = Chem.MolFromSmiles(smiles_mid)
            mol_new = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts('[#0]'))
            smiles_poly = Chem.MolToSmiles(mol_new)

        else:
            (unit_name, dum1, dum2, atom1, atom2, m1, smiles_each, neigh_atoms_info, oligo_list, dum, unit_dis, flag,) \
                = PEMD_lib.Init_info(unit_name, smiles_mid, length, )
            # Join end caps
            smiles_poly = (
                PEMD_lib.gen_smiles_with_cap(unit_name, dum1, dum2, atom1, atom2, smiles_mid,
                                             smiles_LCap_, smiles_RCap_, LCap_, RCap_, )
            )

    elif length > 1:
        # smiles_each = copy.copy(smiles_each_copy)
        (unit_name, dum1, dum2, atom1, atom2, m1, smiles_each, neigh_atoms_info, oligo_list, dum, unit_dis, flag,) \
            = PEMD_lib.Init_info(unit_name, smiles_mid, length, )

        smiles_poly = PEMD_lib.gen_oligomer_smiles(unit_name, dum1, dum2, atom1, atom2, smiles_mid,
                                                   length, smiles_LCap_, LCap_, smiles_RCap_, RCap_, )

    # Delete intermediate XYZ file if exists
    xyz_file_path = unit_name + '.xyz'
    if os.path.exists(xyz_file_path):
        os.remove(xyz_file_path)

    mol = Chem.MolFromSmiles(smiles_poly)
    if mol is None:
        raise ValueError(f"Invalid SMILES string generated: {smiles_poly}")

    return smiles_poly, mol


def build_polymer(unit_name, smiles_poly, out_dir, length, opls, core = '32', atom_typing_ = 'pysimm', ):

    # get origin dir
    # origin_dir = os.getcwd()

    # build directory
    out_dir = out_dir + '/'
    PEMD_lib.build_dir(out_dir)

    # print(smiles)
    mol = pybel.readstring("smi", smiles_poly)
    mol.addh()
    mol.make3D()
    obmol = mol.OBMol
    angle_range = (0, 0.1)
    for obatom in pybel.ob.OBMolAtomIter(obmol):
        for bond in pybel.ob.OBAtomBondIter(obatom):
            neighbor = bond.GetNbrAtom(obatom)
            if len(list(pybel.ob.OBAtomAtomIter(neighbor))) < 2:
                continue
            angle = random.uniform(*angle_range)
            n1 = next(pybel.ob.OBAtomAtomIter(neighbor))
            n2 = next(pybel.ob.OBAtomAtomIter(n1))
            obmol.SetTorsion(obatom.GetIdx(), neighbor.GetIdx(), n1.GetIdx(), n2.GetIdx(), angle)
    mol.localopt()

    # 写入文件
    # file_base = '{}_N{}'.format(unit_name, length)
    pdb_file = out_dir + f"{unit_name}_N{length}.pdb"
    xyz_file = out_dir + f"{unit_name}_N{length}.xyz"
    mol_file = out_dir + f"{unit_name}_N{length}.mol2"

    mol.write("pdb", pdb_file, overwrite=True)
    mol.write("xyz", xyz_file, overwrite=True)
    mol.write("mol2", mol_file, overwrite=True)

    # Generate OPLS parameter file
    if opls is True:
        print(unit_name, ": Generating OPLS parameter file ...")

        if os.path.exists(f"{unit_name}_N{length}.xyz"):
            try:
                Converter.convert(
                    pdb=pdb_file,
                    resname=unit_name,
                    charge=0,
                    opt=0,
                    outdir= out_dir,
                    ln = length,
                )
                print(unit_name, ": OPLS parameter file generated.")
            except BaseException:
                print('problem running LigParGen for {}.pdb.'.format(pdb_file))

    # os.chdir(out_dir)

    print("\n", unit_name, ": Performing a short MD simulation using LAMMPS...\n", )

    PEMD_lib.get_gaff2(unit_name, length, out_dir, mol, atom_typing=atom_typing_)
    # input_file = file_base + '_gaff2.lmp'
    # output_file = file_base + '_gaff2.data'
    PEMD_lib.relax_polymer_lmp(unit_name, length, out_dir, core)


def F_poly_gen(unit_name, repeating_unit, leftcap, rightcap, length, ):

    input_data = [[unit_name, repeating_unit, leftcap, rightcap]]
    df_smiles = pd.DataFrame(input_data, columns=['ID', 'SMILES', 'LeftCap', 'RightCap'])
    smiles_mid = df_smiles.loc[df_smiles['ID'] == unit_name, 'SMILES'].values[0]

    # Extract cap SMILES if available
    smiles_LCap_ = df_smiles.loc[df_smiles['ID'] == unit_name, 'RightCap'].values[0]
    LCap_ = not pd.isna(smiles_LCap_)

    smiles_RCap_ = df_smiles.loc[df_smiles['ID'] == unit_name, 'LeftCap'].values[0]
    RCap_ = not pd.isna(smiles_RCap_)

    # 初始分子定义
    mol = Chem.MolFromSmiles(smiles_mid)
    mol = Chem.AddHs(mol)

    # 获取所有氢原子的索引
    hydrogen_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == 'H']

    # 存储所有生成的分子及其图表示
    graphs = []
    mol_poly_list = []
    smiles_poly_list = []
    smiles_poly = None

    # 逐步替换氢原子为氟原子
    for num_replacements in range(1, len(hydrogen_atoms) + 1):
        for hydrogen_idxs in itertools.combinations(hydrogen_atoms, num_replacements):
            modified_mol = Chem.RWMol(mol)
            for idx in hydrogen_idxs:
                modified_mol.ReplaceAtom(idx, Chem.Atom('F'))
            # Chem.SanitizeMol(modified_mol)

            # 检查是否有同构的分子已经存在
            modified_mol_graph = PEMD_lib.mol_to_nx(modified_mol)
            if not any(PEMD_lib.is_isomorphic(modified_mol_graph, graph) for graph in graphs):
                # molecules.append(modified_mol)
                graphs.append(modified_mol_graph)
                # remove the H atoms
                # mol_noHs = Chem.RemoveHs(modified_mol)

                smiles = Chem.MolToSmiles(modified_mol)
                smiles_noH = smiles.replace('([H])', '')

                # replace Cl with [*]
                # smiles_noCl = smiles_noH.replace('Cl', '[*]')

                (unit_name, dum1, dum2, atom1, atom2, m1, smiles_each, neigh_atoms_info, oligo_list, dum, unit_dis, flag,) \
                    = PEMD_lib.Init_info(unit_name, smiles_noH, length, )
                # print(dum1, dum2, atom1, atom2, dum)

                if length == 1:
                    # Join end caps
                    smiles_poly = PEMD_lib.gen_smiles_with_cap(unit_name, dum1, dum2, atom1, atom2, smiles_noH,
                                                     smiles_LCap_, smiles_RCap_, LCap_, RCap_, )

                elif length > 1:
                    smiles_poly = PEMD_lib.gen_oligomer_smiles(unit_name, dum1, dum2, atom1, atom2, smiles_noH,
                                                           length, smiles_LCap_, LCap_, smiles_RCap_, RCap_, )

                # print(smiles_poly)
                smiles_poly_list.append(smiles_poly)
                mol2 = Chem.MolFromSmiles(smiles_poly)
                mol_poly_list.append(mol2)

    # Delete intermediate XYZ file if exists
    xyz_file_path = unit_name + '.xyz'
    if os.path.exists(xyz_file_path):
        os.remove(xyz_file_path)

    return smiles_poly_list, mol_poly_list


def generate_polymer_smiles(leftcap, repeating_unit, rightcap, length):
    repeating_cleaned = repeating_unit.replace('[*]', '')
    full_sequence = repeating_cleaned * length
    leftcap_cleaned = leftcap.replace('[*]', '')
    rightcap_cleaned = rightcap.replace('[*]', '')
    smiles = leftcap_cleaned + full_sequence + rightcap_cleaned
    return smiles


def smiles_to_files(smiles, angle_range=(0, 0), apply_torsion=False, xyz=False, pdb=False, mol2=False,
                    file_prefix=None):
    if file_prefix is None:
        file_prefix = smiles
    mol = pybel.readstring("smi", smiles)
    mol.addh()
    mol.make3D()
    obmol = mol.OBMol
    if apply_torsion:
        for obatom in pybel.ob.OBMolAtomIter(obmol):
            for bond in pybel.ob.OBAtomBondIter(obatom):
                neighbor = bond.GetNbrAtom(obatom)
                if len(list(pybel.ob.OBAtomAtomIter(neighbor))) < 2:
                    continue
                angle = random.uniform(*angle_range)
                n1 = next(pybel.ob.OBAtomAtomIter(neighbor))
                n2 = next(pybel.ob.OBAtomAtomIter(n1))
                obmol.SetTorsion(obatom.GetIdx(), neighbor.GetIdx(), n1.GetIdx(), n2.GetIdx(), angle)
    mol.localopt()
    if xyz:
        mol.write("xyz", f"{file_prefix}.xyz", overwrite=True)
    if pdb:
        mol.write("pdb", f"{file_prefix}.pdb", overwrite=True)
    if mol2:
        mol.write("mol2", f"{file_prefix}.mol2", overwrite=True)
    # return mol


def pdbtoxyz(pdb_file, xyz_file):
    """
    Convert a PDB file to an XYZ file.

    Args:
    pdb_file (str): Path to the input PDB file.
    xyz_file (str): Path to the output XYZ file.
    """
    with open(pdb_file, 'r') as file:
        lines = file.readlines()

    atoms = []
    for line in lines:
        if line.startswith("HETATM"):
            elements = line.split()
            atom_type = elements[2]  # 原子类型在第三列
            # 只保留元素符号，忽略后缀（如“N1-”中的“1-”）
            atom_type = ''.join(filter(str.isalpha, atom_type))
            x, y, z = elements[5], elements[6], elements[7]  # 坐标在第6、7、8列
            atoms.append(f"{atom_type} {x} {y} {z}")

    with open(xyz_file, 'w') as file:
        file.write(f"{len(atoms)}\n\n")  # 文件头部，包含原子总数和空白标题行
        for atom in atoms:
            file.write(atom + "\n")


def to_resp_gjf(xyz_content, out_file, charge=0, multiplicity=1):
    formatted_coordinates = ""
    lines = xyz_content.split('\n')
    for line in lines[2:]:  # Skip the first two lines (atom count and comment)
        elements = line.split()
        if len(elements) >= 4:
            atom_type, x, y, z = elements[0], elements[1], elements[2], elements[3]
            formatted_coordinates += f"  {atom_type}  {x:>12}{y:>12}{z:>12}\n"

    # RESP template
    template = f"""%nprocshared=32
%mem=64GB
%chk=resp.chk
# B3LYP/6-311G** em=GD3BJ opt 

opt

{charge} {multiplicity}
[GEOMETRY]

ligand_ini.gesp
ligand.gesp    
\n\n"""
    gaussian_input = template.replace("[GEOMETRY]", formatted_coordinates)

    with open(out_file, 'w') as file:
        file.write(gaussian_input)


def run_function_in_background(func):
    def wrapper():
        # 封装的函数，用于在后台执行
        func()

    # 创建一个线程对象，目标函数是wrapper
    thread = threading.Thread(target=wrapper)
    # 启动线程
    thread.start()


def read_and_merge_data(topology_path, directory='./', charge_file='RESP2.chg'):
    # 读取拓扑数据
    atoms_df = read_topology_atoms(topology_path)
    # 读取RESP电荷数据并计算平均电荷
    average_charge = read_resp_charges(directory, charge_file)
    # 合并数据
    merged_data = atoms_df.join(average_charge['Average_Charge'])

    return merged_data


def read_topology_atoms(path):
    with open(path, 'r') as file:
        atom_section_started = False
        atoms_data = []

        for line in file:
            if '[ atoms ]' in line:
                atom_section_started = True
                continue

            if atom_section_started and line.startswith('['):
                break

            if atom_section_started and not line.startswith(';'):
                parts = line.split()
                if len(parts) > 4:
                    atom_name = parts[4]
                    atom_type = parts[1]
                    atoms_data.append((atom_name, atom_type))

    atoms_df = pd.DataFrame(atoms_data, columns=['Atom', 'opls_type'])
    return atoms_df


def read_resp_charges(directory='./', charge_file='RESP2.chg'):
    # Find all directories with "RESP" prefix in the given directory
    resp_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d)) and d.startswith('RESP')]

    all_charge = []

    # Iterate through each RESP directory and add charges and atom names to the list
    for dir_name in resp_dirs:
        charge_file_path = os.path.join(directory, dir_name, charge_file)
        charge_data = pd.read_csv(charge_file_path, delim_whitespace=True, header=None, usecols=[0, 4],
                                  names=['Atom', 'Charge'])

        if charge_data is not None:
            all_charge.append(charge_data)

    # If there are no charge data, return an empty DataFrame
    if not all_charge:
        ave_charge = pd.DataFrame(columns=['Atom', 'Average_Charge'])
    else:
        # Concatenate all charges and keep the Atom names as a column
        ave_charge = pd.concat([df.set_index('Atom') for df in all_charge], axis=1)
        ave_charge['Average_Charge'] = ave_charge.mean(axis=1, numeric_only=True)
        ave_charge.reset_index(inplace=True)  # Resetting index to make 'Atom' a column

        # Select only Atom and Average_Charge columns
        ave_charge = ave_charge[['Atom', 'Average_Charge']]

    return ave_charge


def new_charges(df, charge_counts, scale=1):
    # Create a dictionary to store counters for each atom type
    atom_counters = {atom: 0 for atom in df['Atom'].unique()}

    # List to store results
    results = []

    # Iterate through DataFrame
    for index, row in df.iterrows():
        atom = row['Atom']
        current_charge = row['Average_Charge']
        opls_type = row['opls_type']

        # Update atom counter
        atom_counters[atom] += 1

        # Get the specified number of charge values
        if atom_counters[atom] <= charge_counts.get(atom, 0):
            # print(atom_counters[atom])
            # Find the charge of the last occurrence of this atom type
            last_index = -atom_counters[atom]
            last_charge = df[df['Atom'] == atom].iloc[last_index]['Average_Charge']

            # Calculate the average charge
            new_charge = (current_charge + last_charge) / 2
            results.append({'Atom': atom, 'opls_type': opls_type, 'Index': index, 'Average_Charge': new_charge})
        elif atom_counters[atom] > len(df[df['Atom'] == atom]) - charge_counts.get(atom, 0):
            # 当前原子是后几个原子之一
            last_index = -atom_counters[atom]
            last_charge = df[df['Atom'] == atom].iloc[last_index]['Average_Charge']

            new_charge = (current_charge + last_charge) / 2
            results.append({'Atom': atom, 'opls_type': opls_type, 'Index': index, 'Average_Charge': new_charge})
        else:
            # if atom_counters[atom] == charge_counts.get(atom, 0) + 1:
            new_charge = df[df['Atom'] == atom].iloc[charge_counts.get(atom, 0):-charge_counts.get(atom, 0)][
                'Average_Charge'].mean()
            results.append({'Atom': atom, 'opls_type': opls_type, 'Index': index, 'Average_Charge': new_charge})

    # Create new DataFrame
    new_charges = pd.DataFrame(results)

    new_charges['Average_Charge'] *= scale
    return new_charges


def insert_charge_top(input_file, insert_charge, output_file):
    # Function to update the charge values in the file
    def update_charge_values(peo10_content, dataframe, start, end):
        updated_content = peo10_content.copy()
        df_index = 0

        for i in range(start, end):
            line = peo10_content[i]
            if not line.strip().startswith(';') and len(line.split()) > 6:
                opls_type = line.split()[1]
                if df_index < len(dataframe) and opls_type == dataframe.iloc[df_index]['opls_type']:
                    line_parts = line.split()
                    line_parts[6] = f'{dataframe.iloc[df_index]["Average_Charge"]:.8f}'
                    updated_line = ' '.join(line_parts) + '\n'
                    updated_content[i] = updated_line
                    df_index += 1
                if df_index >= len(dataframe):
                    break
        return updated_content

    # Read the file
    with open(input_file, 'r') as file:
        peo10_top_contents = file.readlines()
    # Searching for the atom section in the file
    atom_section_start = None
    atom_section_end = None
    for i, line in enumerate(peo10_top_contents):
        if line.startswith(';   nr'):
            atom_section_start = i
        elif atom_section_start and line.startswith('['):
            atom_section_end = i
            break
    # Update the file with new charge values
    updated_peo10_top = update_charge_values(peo10_top_contents, insert_charge, atom_section_start, atom_section_end)
    # Saving the updated file
    with open(output_file, 'w') as updated_file:
        updated_file.writelines(updated_peo10_top)
    print(f"Updated file saved at: {output_file}")

    return output_file


def vis_2Dsmiles(smiles, mol_size=(350, 150)):
    img = dm.to_image(smiles, mol_size=mol_size)
    display(img)


# 计算分子的相对分子质量
def calculate_molecular_weight(pdb_file):
    mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)
    if mol is not None:
        molecular_weight = Descriptors.MolWt(mol)
        return molecular_weight
    else:
        raise ValueError(f"Unable to read molecular structure from {pdb_file}")


# 根据密度和分子数量计算盒子大小
def calculate_box_size(numbers, pdb_files, density):
    """
    Calculate the edge length of the box needed based on the quantity of multiple molecules,
    their corresponding PDB files, and the density of the entire system.

    :param numbers: List of quantities of each type of molecule
    :param pdb_files: List of PDB files corresponding to each molecule
    :param density: Density of the entire system in g/cm^3
    :return: Edge length of the box in Angstroms
    """
    total_mass = 0
    for number, pdb_file in zip(numbers, pdb_files):
        molecular_weight = calculate_molecular_weight(pdb_file)  # in g/mol
        total_mass += molecular_weight * number / 6.022e23  # accumulate mass of each molecule in grams

    total_volume = total_mass / density  # volume in cm^3
    length = (total_volume * 1e24) ** (1 / 3)  # convert to Angstroms
    return length


# 定义生成Packmol输入文件的函数
def generate_packmol_input(density, numbers, pdb_files, packmol_input='packmol.inp', packmol_out='packmol.pdb'):
    # Calculate the box size for the given molecules and density
    box_length = calculate_box_size(numbers, pdb_files, density) + 4

    # Initialize the input content with general settings
    input_content = [
        "tolerance 2.5",
        f"output {packmol_out}",
        "filetype pdb"
    ]

    # Add the structure information for each molecule type
    for number, pdb_file in zip(numbers, pdb_files):
        input_content.extend([
            f"\nstructure {pdb_file}",
            f"  number {number}",
            f"  inside box 0.0 0.0 0.0 {box_length:.2f} {box_length:.2f} {box_length:.2f}",
            "end structure"
        ])

    # Write the content to the specified Packmol input file
    with open(packmol_input, 'w') as file:
        file.write('\n'.join(input_content))

    return packmol_input
