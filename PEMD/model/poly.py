"""
Polymer model building tools.

Developed by: Tan Shendong
Date: 2024.01.18
"""


import os
import random
import itertools
import subprocess
import pandas as pd
from rdkit import Chem
from pathlib import Path
from shutil import which
from openbabel import pybel
from PEMD.model import PEMD_lib
from rdkit.Chem import Descriptors
from LigParGenPEMD import Converter


def calc_poly_chains(num_Li_salt , conc_Li_salt, mass_per_chain):

    # calculate the mol of LiTFSI salt
    avogadro_number = 6.022e23  # unit 1/mol
    mol_Li_salt = num_Li_salt / avogadro_number # mol

    # calculate the total mass of the polymer
    total_mass_polymer =  mol_Li_salt / (conc_Li_salt / 1000)  # g

    # calculate the number of polymer chains
    num_chains = (total_mass_polymer*avogadro_number) / mass_per_chain  # no unit; mass_per_chain input unit g/mol

    return int(num_chains)


def calc_poly_length(total_mass_polymer, smiles_repeating_unit, smiles_leftcap, smiles_rightcap, ):
    # remove [*] from the repeating unit SMILES, add hydrogens, and calculate the molecular weight
    simplified_smiles_repeating_unit = smiles_repeating_unit.replace('[*]', '')
    molecule_repeating_unit = Chem.MolFromSmiles(simplified_smiles_repeating_unit)
    mol_weight_repeating_unit = Descriptors.MolWt(molecule_repeating_unit) - 2 * 1.008

    # remove [*] from the end group SMILES, add hydrogens, and calculate the molecular weight
    simplified_smiles_rightcap = smiles_rightcap.replace('[*]', '')
    simplified_smiles_leftcap = smiles_leftcap.replace('[*]', '')
    molecule_rightcap = Chem.MolFromSmiles(simplified_smiles_rightcap)
    molecule_leftcap = Chem.MolFromSmiles(simplified_smiles_leftcap)
    mol_weight_end_group = Descriptors.MolWt(molecule_rightcap) + Descriptors.MolWt(molecule_leftcap) - 2 * 1.008

    # calculate the mass of the polymer chain
    mass_polymer_chain = total_mass_polymer - mol_weight_end_group

    # calculate the number of repeating units in the polymer chain
    length = round(mass_polymer_chain / mol_weight_repeating_unit)

    return length


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


def build_polymer(unit_name, smiles_poly, out_dir, length, opls, core, atom_typing_ = 'pysimm', ):

    # build directory
    out_dir = out_dir + '/'
    PEMD_lib.build_dir(out_dir)

    relax_polymer_lmp_dir = os.path.join(out_dir, 'relax_polymer_lmp')
    os.makedirs(relax_polymer_lmp_dir, exist_ok=True)

    # print(smiles)
    mol = pybel.readstring("smi", smiles_poly)
    mol.addh()
    mol.make3D()
    obmol = mol.OBMol
    angle_range = (0, 1)
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

    # 构建文件名基础，这样可以避免重复拼接字符串
    file_base = f"{unit_name}_N{length}"

    # 使用os.path.join构建完整的文件路径，确保路径在不同操作系统上的兼容性
    pdb_file = os.path.join(relax_polymer_lmp_dir, f"{file_base}.pdb")
    xyz_file = os.path.join(relax_polymer_lmp_dir, f"{file_base}.xyz")
    mol_file = os.path.join(relax_polymer_lmp_dir, f"{file_base}.mol2")

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

    print("\n", unit_name, ": Performing a short MD simulation using LAMMPS...\n", )

    PEMD_lib.get_gaff2(unit_name, length, relax_polymer_lmp_dir, mol, atom_typing=atom_typing_)
    PEMD_lib.relax_polymer_lmp(unit_name, length, relax_polymer_lmp_dir, core)


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


# 根据密度和分子数量计算盒子大小
def calculate_box_size(numbers, pdb_files, density):
    total_mass = 0
    for num, file in zip(numbers, pdb_files):

        molecular_weight = PEMD_lib.calc_mol_weight(file)  # in g/mol
        total_mass += molecular_weight * num / 6.022e23  # accumulate mass of each molecule in grams

    total_volume = total_mass / density  # volume in cm^3
    length = (total_volume * 1e24) ** (1 / 3)  # convert to Angstroms
    return length


# 定义生成Packmol输入文件的函数
def gen_packmol_input(out_dir, density, model_info, add_length=30, packinp_name='pack.inp',packout_name='pack_cell.pdb'):

    current_path = os.getcwd()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    MD_dir = os.path.join(current_path, out_dir, 'MD_dir')
    PEMD_lib.build_dir(MD_dir)  # 确保这个函数可以正确创建目录

    packinp_path = os.path.join(MD_dir, packinp_name)

    numbers = PEMD_lib.print_compounds(model_info,'numbers')
    compounds = PEMD_lib.print_compounds(model_info,'compound')

    pdb_files = []
    for com in compounds:
        filepath = os.path.join(MD_dir, f"{com}.pdb")
        pdb_files.append(filepath)

    box_length = calculate_box_size(numbers, pdb_files, density) + add_length  # add 10 Angstroms to each side

    file_contents = "tolerance 2.0\n"
    file_contents += f"add_box_sides 1.2\n"
    file_contents += f"output {packout_name}\n"
    file_contents += "filetype pdb\n\n"

    # 循环遍历每种分子的数量和对应的 PDB 文件
    for num, file in zip(numbers, pdb_files):
        file_contents = file_contents + f"\nstructure {file}.pdb\n"
        file_contents = file_contents + f"  number {num}\n"
        file_contents = file_contents + f"  inside box 0.0 0.0 0.0 {box_length:.2f} {box_length:.2f} {box_length:.2f}\n"
        file_contents = file_contents + "end structure\n\n"

    # write to file
    with open(packinp_path, 'w') as file:
        file.write(file_contents)
    print(f"packmol input file generation successful：{packinp_path}")

    return packinp_path


def run_packmol(out_dir, input_file='pack.inp', output_file='pack.out'):
    current_path = os.getcwd()
    if not which("packmol"):
        raise RuntimeError(
            "Running Packmol requires the executable 'packmol' to be in the path. Please "
            "download packmol from https://github.com/leandromartinez98/packmol and follow the "
            "instructions in the README to compile. Don't forget to add the packmol binary to your path"
        )

    try:
        MD_dir = os.path.join(current_path, out_dir, 'MD_dir')
        os.chdir(MD_dir)
        p = subprocess.run(
            f"packmol < {input_file}",
            check=True,
            shell=True,
            capture_output=True,
        )

        # Check for errors in packmol output
        if "ERROR" in p.stdout.decode():
            if "Could not open file." in p.stdout.decode():
                raise ValueError(
                    "Your packmol might be too old to handle paths with spaces. "
                    "Please try again with a newer version or use paths without spaces."
                )
            msg = p.stdout.decode().split("ERROR")[-1]
            raise ValueError(f"Packmol failed with return code 0 and stdout: {msg}")

        # Write packmol output to the specified output file
        with open(Path(MD_dir, output_file), mode="w") as out:
            out.write(p.stdout.decode())

    except subprocess.CalledProcessError as exc:
        if exc.returncode != 173:  # Only raise the error if it's not the specific 'STOP 173' case
            raise ValueError(f"Packmol failed with error code {exc.returncode} and stderr: {exc.stderr.decode()}") from exc
        else:
            print("Packmol ended with 'STOP 173', but this error is being ignored.")

    finally:
        # Change back to the original working directory
        os.chdir(current_path)



