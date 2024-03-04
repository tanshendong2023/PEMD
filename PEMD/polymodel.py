"""
Polymer model building tools.

Developed by: Tan Shendong
Date: 2023.01.18
"""

import os
import random
import subprocess
import threading
import py3Dmol
import pandas as pd
import datamol as dm
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from IPython.display import display
from PEMD import PEMD_lib

def build_oligomer(unit_name, repeating_unit, leftcap, rightcap, out_dir, Length, OPLS = True, NumConf=1, NCores_opt = 1,):
    # get origin dir
    original_dir = os.getcwd()

    # build a directory
    out_dir = out_dir + '/'
    PEMD_lib.build_dir(out_dir)

    # Dataframe
    input_data = [[unit_name, repeating_unit, leftcap, rightcap]]
    df_smiles = pd.DataFrame(input_data, columns=['ID', 'SMILES', 'LeftCap', 'RightCap'])

    # Obtain Cap smiles
    LCap_ = False
    RCap_ = False
    if 'LeftCap' in df_smiles.columns:
        smiles_LCap_ = df_smiles[df_smiles['ID'] == unit_name]['LeftCap'].values[0]
        if PEMD_lib.is_nan(smiles_LCap_) is False:          # Check if the SMILES string is NaN
            LCap_ = True
    else:
        smiles_LCap_ = ''
    if 'RightCap' in df_smiles.columns:
        smiles_RCap_ = df_smiles[df_smiles['ID'] == unit_name]['RightCap'].values[0]
        if PEMD_lib.is_nan(smiles_RCap_) is False:           # Check if the SMILES string is NaN
            RCap_ = True
    else:
        smiles_RCap_ = ''

    # Get repeating_unit SMILES
    smiles_each = df_smiles[df_smiles['ID'] == unit_name]['SMILES'].values[0]

    # count = 0
    smiles_each_ind = None
    Final_SMILES = []
    for ln in Length:
        # start_1 = time.time()
        if ln == 1:
            if LCap_ is False and RCap_ is False:
                mol = Chem.MolFromSmiles(smiles_each)
                mol_new = Chem.DeleteSubstructs(mol, Chem.MolFromSmarts('[#0]'))
                smiles_each_ind = Chem.MolToSmiles(mol_new)
            else:
                (unit_name, dum1, dum2, atom1, atom2, m1, neigh_atoms_info,oligo_list, dum, unit_dis, flag,) \
                    = PEMD_lib.Init_info(unit_name, smiles_each, Length)

                if flag == 'REJECT' and len(Final_SMILES) == 0 and ln == Length[-1]:
                    return unit_name, 'REJECT', Final_SMILES
                elif flag == 'REJECT' and len(Final_SMILES) >= 1 and ln == Length[-1]:
                    return unit_name, 'PARTIAL SUCCESS', Final_SMILES
                # Join end caps
                smiles_each_ind = (
                    PEMD_lib.gen_smiles_with_cap(unit_name, dum1, dum2, atom1, atom2, smiles_each,
                                                 smiles_LCap_, smiles_RCap_, LCap_, RCap_,)
                )

        elif ln > 1:
            # smiles_each = copy.copy(smiles_each_copy)
            (unit_name, dum1, dum2, atom1, atom2, m1, neigh_atoms_info, oligo_list, dum, unit_dis, flag,) \
                = PEMD_lib.Init_info(unit_name, smiles_each, Length)

            if flag == 'REJECT' and len(Final_SMILES) == 0 and ln == Length[-1]:
                return unit_name, 'REJECT', Final_SMILES
            elif flag == 'REJECT' and len(Final_SMILES) >= 1 and ln == Length[-1]:
                return unit_name, 'PARTIAL SUCCESS', Final_SMILES

            smiles_each_ind = PEMD_lib.gen_oligomer_smiles(unit_name, dum1, dum2, atom1, atom2, smiles_each,
                                                           ln, smiles_LCap_, LCap_, smiles_RCap_, RCap_,)

        m1 = Chem.MolFromSmiles(smiles_each_ind)
        if m1 is None and len(Final_SMILES) == 0 and ln == Length[-1]:
            return unit_name, 'REJECT', Final_SMILES
        elif m1 is None and len(Final_SMILES) >= 1 and ln == Length[-1]:
            return unit_name, 'PARTIAL SUCCESS', Final_SMILES

        Final_SMILES.append(smiles_each_ind)

        PEMD_lib.gen_conf_xyz_vasp(unit_name, m1, out_dir, ln, NumConf, NCores_opt, OPLS, )
        # if NumC == 0 and ln == Length[-1]:
        #     return unit_name, 'FAILURE', Final_SMILES
        # elif ln == Length[-1]:
        #     return unit_name, 'SUCCESS', Final_SMILES

    # go back the origin dir
    os.chdir(original_dir)

def build_polymer(unit_name, repeating_unit, leftcap, rightcap, rot_angles_monomer, rot_angles_dimer,
                  Steps, Substeps, num_conf, length, method, IntraChainCorr, Tol_ChainCorr, Inter_Chain_Dis,):

    input_data = [[unit_name, repeating_unit, leftcap, rightcap]]
    df_smiles = pd.DataFrame(input_data, columns=['ID', 'SMILES', 'LeftCap', 'RightCap'])


    # print(" Chain model building started for", unit_name, "...")
    # vasp_out_dir_indi = vasp_out_dir + unit_name + '/'
    # build_dir(vasp_out_dir_indi)

    # Initial values
    decision = 'FAILURE'
    SN = 0

    # Get SMILES
    smiles_each = df_smiles[df_smiles['ID'] == unit_name]['SMILES'].values[0]

    (unit_name, dum1, dum2, atom1, atom2, m1, neigh_atoms_info, oligo_list, dum, unit_dis,flag,)\
        = PEMD_lib.Init_info(unit_name, smiles_each, length)

    # keep a copy of idx of dummy and linking atoms
    # These idx will be used if there is no rotatable bonds
    dum1_smi, dum2_smi, atom1_smi, atom2_smi = dum1, dum2, atom1, atom2

    if flag == 'REJECT':
        return unit_name, 'REJECT', 0

    if atom1 == atom2:
        smiles_each = PEMD_lib.gen_dimer_smiles(dum1, dum2, atom1, atom2, smiles_each)

        (unit_name, dum1, dum2, atom1, atom2, m1, neigh_atoms_info, oligo_list, dum, unit_dis, flag,) \
            = PEMD_lib.Init_info(unit_name, smiles_each, length)

        if flag == 'REJECT':
            return unit_name, 'REJECT', 0

    # create 100 conformers and select which has the largest dihedral angle (within 8 degree) and lowest energy
    PEMD_lib.find_best_conf(unit_name, m1, dum1, dum2, atom1, atom2)

    # Minimize geometry using steepest descent
    unit = PEMD_lib.localopt(unit_name, unit_name + '.xyz', dum1, dum2, atom1, atom2,)

    # Rearrange rows
    rows = unit.index.tolist()
    for i in [dum1, atom1, atom2, dum2]:
        rows.remove(i)
    new_rows = [dum1, atom1, atom2, dum2] + rows
    unit = unit.loc[new_rows].reset_index(drop=True)
    dum1, atom1, atom2, dum2 = 0, 1, 2, 3

    PEMD_lib.gen_xyz(unit_name + '_rearranged.xyz', unit)

    # update neigh_atoms_info
    neigh_atoms_info = PEMD_lib.connec_info(unit_name + '_rearranged.xyz')

    # Stretch the repeating unit
    if IntraChainCorr == 1:
        unit, unit_init_xyz = PEMD_lib.MakePolymerStraight(
            unit_name,
            unit_name + '_rearranged.xyz',
            unit,
            dum1,
            dum2,
            atom1,
            atom2,
            Tol_ChainCorr,
        )

    check_connectivity_dimer = PEMD_lib.mono2dimer(
        unit_name, unit, 'CORRECT', dum1, dum2, atom1, atom2, unit_dis
    )

    # building unit is copied and may be used in building flexible dimers
    unit_copied = unit.copy()

    if check_connectivity_dimer == 'CORRECT':
        decision = 'SUCCESS'
        SN += 1

        if 'n' in length:
            PEMD_lib.gen_vasp(
                vasp_out_dir_indi,
                unit_name,
                unit,
                dum1,
                dum2,
                atom1,
                atom2,
                dum,
                unit_dis,
                Inter_Chain_Dis=Inter_Chain_Dis,
                Polymer=True,
            )

        if len(oligo_list) > 0:
            for oligo_len in oligo_list:
                (
                    oligomer,
                    dum1_oligo,
                    atom1_oligo,
                    dum2_oligo,
                    atom2_oligo,
                ) = oligomer_build(
                    unit,
                    unit_name,
                    dum1,
                    dum2,
                    atom1,
                    atom2,
                    oligo_len,
                    unit_dis,
                    neigh_atoms_info,
                )
                gen_vasp(
                    vasp_out_dir_indi,
                    unit_name,
                    oligomer,
                    dum1_oligo,
                    dum2_oligo,
                    atom1_oligo,
                    atom2_oligo,
                    dum,
                    unit_dis,
                    length=oligo_len,
                    Inter_Chain_Dis=Inter_Chain_Dis,
                )

                # Remove dummy atoms
                oligomer = oligomer.drop([dum1_oligo, dum2_oligo])

                if SN > 1:
                    xyz_file_name = (
                        vasp_out_dir_indi
                        + unit_name
                        + '_'
                        + str(SN)
                        + '_N'
                        + str(oligo_len)
                        + '.xyz'
                    )
                else:
                    xyz_file_name = (
                        vasp_out_dir_indi + unit_name + '_N' + str(oligo_len) + '.xyz'
                    )

                gen_xyz(xyz_file_name, oligomer)

        if SN >= num_conf:
            print(" Chain model building completed for", unit_name, ".")
            return unit_name, 'SUCCESS', SN

    # Simulated Annealing
    if method == 'SA' and SN < num_conf:
        print(" Entering simulated annealing steps", unit_name, "...")
        #  Find single bonds and rotate
        single_bond = single_bonds(unit_name, unit, xyz_tmp_dir)

        isempty = single_bond.empty
        if isempty is True and SN < num_conf:
            print(unit_name, "No rotatable single bonds, building a dimer ...")

            smiles_each = gen_dimer_smiles(
                dum1_smi, dum2_smi, atom1_smi, atom2_smi, smiles_each
            )
            (
                unit_name,
                dum1,
                dum2,
                atom1,
                atom2,
                m1,
                neigh_atoms_info,
                oligo_list,
                dum,
                unit_dis,
                flag,
            ) = Init_info(unit_name, smiles_each, xyz_in_dir, length)

            if flag == 'REJECT':
                return unit_name, 'REJECT', 0

            # create 100 conformers and select which has the largest dihedral angle (within 8 degree) and lowest energy
            find_best_conf(unit_name, m1, dum1, dum2, atom1, atom2, xyz_in_dir)

            # Minimize geometry using steepest descent
            unit = localopt(
                unit_name,
                xyz_in_dir + unit_name + '.xyz',
                dum1,
                dum2,
                atom1,
                atom2,
                xyz_tmp_dir,
            )

            # Rearrange rows
            rows = unit.index.tolist()
            for i in [dum1, atom1, atom2, dum2]:
                rows.remove(i)
            new_rows = [dum1, atom1, atom2, dum2] + rows
            unit = unit.loc[new_rows].reset_index(drop=True)
            dum1, atom1, atom2, dum2 = 0, 1, 2, 3

            gen_xyz(xyz_tmp_dir + unit_name + '_rearranged.xyz', unit)

            # update neigh_atoms_info
            neigh_atoms_info = connec_info(xyz_tmp_dir + unit_name + '_rearranged.xyz')

            # Stretch the repeating unit
            if IntraChainCorr == 1:
                unit, unit_init_xyz = MakePolymerStraight(
                    unit_name,
                    xyz_tmp_dir + unit_name + '_rearranged.xyz',
                    unit,
                    dum1,
                    dum2,
                    atom1,
                    atom2,
                    xyz_tmp_dir,
                    Tol_ChainCorr,
                )

            check_connectivity_dimer = mono2dimer(
                unit_name, unit, 'CORRECT', dum1, dum2, atom1, atom2, unit_dis
            )

            # building unit is copied and may be used in building flexible dimers
            unit_copied = unit.copy()

            if check_connectivity_dimer == 'CORRECT':
                decision = 'SUCCESS'
                SN += 1

                if 'n' in length:
                    gen_vasp(
                        vasp_out_dir_indi,
                        unit_name,
                        unit,
                        dum1,
                        dum2,
                        atom1,
                        atom2,
                        dum,
                        unit_dis,
                        Inter_Chain_Dis=Inter_Chain_Dis,
                        Polymer=True,
                    )

                if len(oligo_list) > 0:
                    for oligo_len in oligo_list:
                        (
                            oligomer,
                            dum1_oligo,
                            atom1_oligo,
                            dum2_oligo,
                            atom2_oligo,
                        ) = oligomer_build(
                            unit,
                            unit_name,
                            dum1,
                            dum2,
                            atom1,
                            atom2,
                            oligo_len,
                            unit_dis,
                            neigh_atoms_info,
                        )
                        gen_vasp(
                            vasp_out_dir_indi,
                            unit_name,
                            oligomer,
                            dum1_oligo,
                            dum2_oligo,
                            atom1_oligo,
                            atom2_oligo,
                            dum,
                            unit_dis,
                            length=oligo_len,
                            Inter_Chain_Dis=Inter_Chain_Dis,
                        )

                        # Remove dummy atoms
                        oligomer = oligomer.drop([dum1_oligo, dum2_oligo])

                        if SN > 1:
                            xyz_file_name = (
                                vasp_out_dir_indi
                                + unit_name
                                + '_'
                                + str(SN)
                                + '_N'
                                + str(oligo_len)
                                + '.xyz'
                            )
                        else:
                            xyz_file_name = (
                                vasp_out_dir_indi
                                + unit_name
                                + '_N'
                                + str(oligo_len)
                                + '.xyz'
                            )

                        gen_xyz(xyz_file_name, oligomer)

                if SN >= num_conf:
                    print(" Chain model building completed for", unit_name, ".")
                    return unit_name, 'SUCCESS', SN
                else:
                    #  Find single bonds and rotate
                    single_bond = single_bonds(unit_name, unit, xyz_tmp_dir)

        if isempty is True and SN > 0:
            print(" Chain model building completed for", unit_name, ".")
            return unit_name, 'SUCCESS', SN
        if isempty is True and SN == 0:
            return unit_name, 'FAILURE', 0

        results = an.SA(
            unit_name,
            unit,
            single_bond,
            rot_angles_monomer,
            neigh_atoms_info,
            xyz_tmp_dir,
            dum1,
            dum2,
            atom1,
            atom2,
            Steps,
            Substeps,
        )

        results = results.sort_index(ascending=False)

        # Keep num_conf+10 rows to reduce computational costs
        results = results.head(num_conf + 10)
        TotalSN, first_conf_saved = 0, 0
        for index, row in results.iterrows():
            TotalSN += 1

            # Minimize geometry using steepest descent
            final_unit = localopt(
                unit_name, row['xyzFile'], dum1, dum2, atom1, atom2, xyz_tmp_dir
            )

            # Stretch the repeating unit
            if IntraChainCorr == 1:
                final_unit, final_unit_xyz = MakePolymerStraight(
                    unit_name,
                    xyz_tmp_dir + unit_name + '_rearranged.xyz',
                    final_unit,
                    dum1,
                    dum2,
                    atom1,
                    atom2,
                    xyz_tmp_dir,
                    Tol_ChainCorr,
                )
            else:
                final_unit_xyz = row['xyzFile']

            # Check connectivity
            check_connectivity_monomer = 'CORRECT'

            neigh_atoms_info_new = connec_info(final_unit_xyz)
            for row in neigh_atoms_info.index.tolist():
                if sorted(neigh_atoms_info.loc[row]['NeiAtom']) != sorted(
                    neigh_atoms_info_new.loc[row]['NeiAtom']
                ):
                    check_connectivity_monomer = 'WRONG'

            if check_connectivity_monomer == 'CORRECT' and first_conf_saved == 0:
                # building unit is copied and may be used in building flexible dimers
                unit_copied = final_unit.copy()
                first_conf_saved = 1

            check_connectivity_dimer = mono2dimer(
                unit_name,
                final_unit,
                check_connectivity_monomer,
                dum1,
                dum2,
                atom1,
                atom2,
                unit_dis,
            )

            if check_connectivity_dimer == 'CORRECT':
                decision = 'SUCCESS'
                SN += 1
                if 'n' in length:
                    gen_vasp(
                        vasp_out_dir_indi,
                        unit_name,
                        final_unit,
                        dum1,
                        dum2,
                        atom1,
                        atom2,
                        dum,
                        unit_dis,
                        Inter_Chain_Dis=Inter_Chain_Dis,
                        Polymer=True,
                    )

                if len(oligo_list) > 0:
                    for oligo_len in oligo_list:
                        (
                            oligomer,
                            dum1_oligo,
                            atom1_oligo,
                            dum2_oligo,
                            atom2_oligo,
                        ) = oligomer_build(
                            final_unit,
                            unit_name,
                            dum1,
                            dum2,
                            atom1,
                            atom2,
                            oligo_len,
                            unit_dis,
                            neigh_atoms_info_new,
                        )
                        gen_vasp(
                            vasp_out_dir_indi,
                            unit_name,
                            oligomer,
                            dum1_oligo,
                            dum2_oligo,
                            atom1_oligo,
                            atom2_oligo,
                            dum,
                            unit_dis,
                            length=oligo_len,
                            Inter_Chain_Dis=Inter_Chain_Dis,
                        )

                        # Remove dummy atoms
                        oligomer = oligomer.drop([dum1_oligo, dum2_oligo])

                        if SN > 1:
                            xyz_file_name = (
                                vasp_out_dir_indi
                                + unit_name
                                + '_'
                                + str(SN)
                                + '_N'
                                + str(oligo_len)
                                + '.xyz'
                            )
                        else:
                            xyz_file_name = (
                                vasp_out_dir_indi
                                + unit_name
                                + '_N'
                                + str(oligo_len)
                                + '.xyz'
                            )

                        gen_xyz(xyz_file_name, oligomer)

                if SN == num_conf:
                    break

            # If we do not get a proper monomer unit, then consider a dimer as a monomer unit and
            # build a dimer of the same
            if SN < num_conf and TotalSN == results.index.size:
                unit = unit_copied.copy()
                unit = trans_origin(unit, atom1)
                unit = alignZ(unit, atom1, atom2)
                (
                    unit_dimer,
                    neigh_atoms_info_dimer,
                    dum1_2nd,
                    dum2_2nd,
                    atom1_2nd,
                    atom2_2nd,
                ) = build_dimer_rotate(
                    unit_name,
                    rot_angles_dimer,
                    unit,
                    unit,
                    dum,
                    dum1,
                    dum2,
                    atom1,
                    atom2,
                    unit_dis,
                )
                isempty = unit_dimer.empty

                if isempty is True and SN == 0:
                    print(unit_name, "Couldn't find an acceptable dimer.")
                    return unit_name, 'FAILURE', 0
                elif isempty is True and SN > 0:
                    print(" Chain model building completed for", unit_name, ".")
                    return unit_name, 'SUCCESS', SN

                # Generate XYZ file
                gen_xyz(xyz_tmp_dir + unit_name + '_dimer.xyz', unit_dimer)

                # Minimize geometry using steepest descent
                unit_dimer = localopt(
                    unit_name,
                    xyz_tmp_dir + unit_name + '_dimer.xyz',
                    dum1_2nd,
                    dum2_2nd,
                    atom1_2nd,
                    atom2_2nd,
                    xyz_tmp_dir,
                )

                # Stretch the repeating unit
                if IntraChainCorr == 1:
                    unit_dimer, unit_dimer_xyz = MakePolymerStraight(
                        unit_name,
                        xyz_tmp_dir + unit_name + '_dimer.xyz',
                        unit_dimer,
                        dum1_2nd,
                        dum2_2nd,
                        atom1_2nd,
                        atom2_2nd,
                        xyz_tmp_dir,
                        Tol_ChainCorr,
                    )

                check_connectivity_dimer = mono2dimer(
                    unit_name,
                    unit_dimer,
                    'CORRECT',
                    dum1_2nd,
                    dum2_2nd,
                    atom1_2nd,
                    atom2_2nd,
                    unit_dis,
                )

                if check_connectivity_dimer == 'CORRECT':
                    decision = 'SUCCESS'
                    SN += 1
                    if 'n' in length:
                        gen_vasp(
                            vasp_out_dir_indi,
                            unit_name,
                            unit_dimer,
                            dum1_2nd,
                            dum2_2nd,
                            atom1_2nd,
                            atom2_2nd,
                            dum,
                            unit_dis,
                            Inter_Chain_Dis=Inter_Chain_Dis,
                            Polymer=True,
                        )

                    if len(oligo_list) > 0:
                        for oligo_len in oligo_list:
                            (
                                oligomer,
                                dum1_oligo,
                                atom1_oligo,
                                dum2_oligo,
                                atom2_oligo,
                            ) = oligomer_build(
                                unit_dimer,
                                unit_name,
                                dum1_2nd,
                                dum2_2nd,
                                atom1_2nd,
                                atom2_2nd,
                                oligo_len,
                                unit_dis,
                                neigh_atoms_info_dimer,
                            )
                            gen_vasp(
                                vasp_out_dir_indi,
                                unit_name,
                                oligomer,
                                dum1_oligo,
                                dum2_oligo,
                                atom1_oligo,
                                atom2_oligo,
                                dum,
                                unit_dis,
                                length=oligo_len,
                                Inter_Chain_Dis=Inter_Chain_Dis,
                            )

                            # Remove dummy atoms
                            oligomer = oligomer.drop([dum1_oligo, dum2_oligo])
                            if SN > 1:
                                xyz_file_name = (
                                    vasp_out_dir_indi
                                    + unit_name
                                    + '_'
                                    + str(SN)
                                    + '_N'
                                    + str(oligo_len)
                                    + '.xyz'
                                )
                            else:
                                xyz_file_name = (
                                    vasp_out_dir_indi
                                    + unit_name
                                    + '_N'
                                    + str(oligo_len)
                                    + '.xyz'
                                )

                            gen_xyz(xyz_file_name, oligomer)

                    if SN == num_conf:
                        break

                # Generate XYZ file and find connectivity
                # gen_xyz(xyz_tmp_dir + unit_name + '_dimer.xyz', unit_dimer)
                neigh_atoms_info_dimer = connec_info(
                    xyz_tmp_dir + unit_name + '_dimer.xyz'
                )

                #  Find single bonds and rotate
                single_bond_dimer = single_bonds(unit_name, unit_dimer, xyz_tmp_dir)

                isempty = single_bond_dimer.empty
                if isempty is True and SN == 0:
                    print(unit_name, "No rotatable single bonds in dimer")
                    return unit_name, 'FAILURE', 0
                elif isempty is True and SN > 0:
                    print(" Chain model building completed for", unit_name, ".")
                    return unit_name, 'SUCCESS', SN

                results = an.SA(
                    unit_name,
                    unit_dimer,
                    single_bond_dimer,
                    rot_angles_monomer,
                    neigh_atoms_info_dimer,
                    xyz_tmp_dir,
                    dum1_2nd,
                    dum2_2nd,
                    atom1_2nd,
                    atom2_2nd,
                    Steps,
                    Substeps,
                )
                results = results.sort_index(ascending=False)

                for index, row in results.iterrows():

                    # Minimize geometry using steepest descent
                    final_unit = localopt(
                        unit_name,
                        row['xyzFile'],
                        dum1_2nd,
                        dum2_2nd,
                        atom1_2nd,
                        atom2_2nd,
                        xyz_tmp_dir,
                    )

                    # Stretch the repeating unit
                    if IntraChainCorr == 1:
                        final_unit, final_unit_xyz = MakePolymerStraight(
                            unit_name,
                            xyz_tmp_dir + unit_name + '_dimer.xyz',
                            final_unit,
                            dum1_2nd,
                            dum2_2nd,
                            atom1_2nd,
                            atom2_2nd,
                            xyz_tmp_dir,
                            Tol_ChainCorr,
                        )
                    else:
                        final_unit_xyz = row['xyzFile']

                    # Check Connectivity
                    check_connectivity_monomer = 'CORRECT'

                    neigh_atoms_info_new = connec_info(final_unit_xyz)
                    for row in neigh_atoms_info_dimer.index.tolist():
                        if sorted(neigh_atoms_info_dimer.loc[row]['NeiAtom']) != sorted(
                            neigh_atoms_info_new.loc[row]['NeiAtom']
                        ):
                            check_connectivity_monomer = 'WRONG'

                    check_connectivity_dimer = mono2dimer(
                        unit_name,
                        final_unit,
                        check_connectivity_monomer,
                        dum1_2nd,
                        dum2_2nd,
                        atom1_2nd,
                        atom2_2nd,
                        unit_dis,
                    )
                    if check_connectivity_dimer == 'CORRECT':
                        decision = 'SUCCESS'
                        SN += 1
                        if 'n' in length:
                            gen_vasp(
                                vasp_out_dir_indi,
                                unit_name,
                                final_unit,
                                dum1_2nd,
                                dum2_2nd,
                                atom1_2nd,
                                atom2_2nd,
                                dum,
                                unit_dis,
                                Inter_Chain_Dis=Inter_Chain_Dis,
                                Polymer=True,
                            )

                        if len(oligo_list) > 0:
                            for oligo_len in oligo_list:
                                (
                                    oligomer,
                                    dum1_oligo,
                                    atom1_oligo,
                                    dum2_oligo,
                                    atom2_oligo,
                                ) = oligomer_build(
                                    final_unit,
                                    unit_name,
                                    dum1_2nd,
                                    dum2_2nd,
                                    atom1_2nd,
                                    atom2_2nd,
                                    oligo_len,
                                    unit_dis,
                                    neigh_atoms_info_new,
                                )
                                gen_vasp(
                                    vasp_out_dir_indi,
                                    unit_name,
                                    oligomer,
                                    dum1_oligo,
                                    dum2_oligo,
                                    atom1_oligo,
                                    atom2_oligo,
                                    dum,
                                    unit_dis,
                                    length=oligo_len,
                                    Inter_Chain_Dis=Inter_Chain_Dis,
                                )

                                # Remove dummy atoms
                                oligomer = oligomer.drop([dum1_oligo, dum2_oligo])

                                if SN > 1:
                                    xyz_file_name = (
                                        vasp_out_dir_indi
                                        + unit_name
                                        + '_'
                                        + str(SN)
                                        + '_N'
                                        + str(oligo_len)
                                        + '.xyz'
                                    )
                                else:
                                    xyz_file_name = (
                                        vasp_out_dir_indi
                                        + unit_name
                                        + '_N'
                                        + str(oligo_len)
                                        + '.xyz'
                                    )

                                gen_xyz(xyz_file_name, oligomer)

                        if SN == num_conf:
                            break

    elif method == 'Dimer' and SN < num_conf:
        print(" Generating dimers", unit_name, "...")
        SN = 0
        for angle in rot_angles_dimer:
            unit1 = unit.copy()
            unit2 = unit.copy()
            unit2 = trans_origin(unit2, atom1)
            unit2 = alignZ(unit2, atom1, atom2)
            unit2 = rotateZ(unit2, angle, np.arange(len(unit2[0].values)))

            (
                dimer,
                check_connectivity_monomer,
                dum1_2nd,
                dum2_2nd,
                atom1_2nd,
                atom2_2nd,
            ) = TwoMonomers_Dimer(
                unit_name, unit1, unit2, dum1, dum2, atom1, atom2, dum, unit_dis
            )
            if atom1_2nd != atom2_2nd:
                if check_connectivity_monomer == 'CORRECT':
                    unit = dimer.copy()
                    # Build a dimer
                    unit = trans_origin(unit, dum1_2nd)
                    unit = alignZ(unit, dum1_2nd, dum2_2nd)
                    polymer, check_connectivity_dimer = build(
                        unit_name,
                        2,
                        unit,
                        dum1_2nd,
                        dum2_2nd,
                        atom1_2nd,
                        atom2_2nd,
                        unit_dis,
                    )

                    if check_connectivity_dimer == 'CORRECT':
                        decision = 'SUCCESS'
                        SN += 1
                        if 'n' in length:
                            gen_vasp(
                                vasp_out_dir_indi,
                                unit_name,
                                dimer,
                                dum1_2nd,
                                dum2_2nd,
                                atom1_2nd,
                                atom2_2nd,
                                dum,
                                unit_dis,
                                Inter_Chain_Dis=Inter_Chain_Dis,
                                Polymer=True,
                            )

                        if SN == num_conf:
                            break
    print(" Chain model building completed for", unit_name, ".")
    return unit_name, decision, SN


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


def to_resp2_gjf(xyz_content, out_file, charge=0, multiplicity=1):
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
%chk=opt.chk
# B3LYP/TZVP em=GD3BJ opt

opt

{charge} {multiplicity}
[GEOMETRY]    
--link1--
%nprocshared=32
%mem=64GB
%oldchk=opt.chk
%chk=SP_gas.chk
# B3LYP/def2TZVP em=GD3BJ geom=allcheck

--link1--
%nprocshared=32
%mem=64GB
%oldchk=opt.chk
%chk=SP_solv.chk
# B3LYP/def2TZVP em=GD3BJ scrf=(pcm,solvent=generic,read) geom=allcheck

eps=5.0
epsinf=2.1\n\n"""
    gaussian_input = template.replace("[GEOMETRY]", formatted_coordinates)

    with open(out_file, 'w') as file:
        file.write(gaussian_input)


def Conformers_search(smiles, Num, charge=0, multiplicity=1):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # 使用 RDKit 生成构象
    AllChem.EmbedMultipleConfs(mol, numConfs=Num, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)

    for conf_id in range(Num):
        # Generate xyz file content
        xyz_content = f"{mol.GetNumAtoms()}\n\n"
        conf = mol.GetConformer(conf_id)
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            xyz_content += f"{atom.GetSymbol()} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f}\n"

        # Create a directory for each conformer
        dir_name = f"RESP_{conf_id + 1}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        # Generate RESP file
        resp_file = os.path.join(dir_name, f"RESP_{conf_id + 1}.gjf")
        to_resp_gjf(xyz_content, resp_file, charge, multiplicity)


import os

def write_subscript(partition=None, node=1, core=32, task_type='g16'):
    for folder in os.listdir('.'):
        if os.path.isdir(folder) and folder.startswith('RESP'):
            # Start building the script content with common headers
            script_content = f"""#!/bin/bash
#SBATCH -J {task_type}
"""
            # Add the partition line only if the partition parameter is provided
            if partition:
                script_content += f"#SBATCH -p {partition}\n"

            # Continue adding the rest of the script content
            script_content += f"""#SBATCH -N {node}
#SBATCH -n {core}           
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j\n
"""

            # Add Gaussian module loading and execution command only for 'g16' tasks
            if task_type == 'g16':
                script_content += """module load Gaussian
g16 $1
"""
            script_path = os.path.join(folder, f'sub_{task_type}.sh')
            with open(script_path, 'w') as script_file:
                script_file.write(script_content)
            # Use octal literal for setting permissions (755)
            os.chmod(script_path, 0o755)

def submit_gjf_files():
    current_dir = os.getcwd()  # 保存当前工作目录
    for folder in os.listdir('.'):
        if os.path.isdir(folder) and folder.startswith('RESP'):
            gjf_file = f'{folder}.gjf'
            script_file = 'sub_g16.sh'
            os.chdir(folder)  # 切换到子目录
            if os.path.exists(gjf_file) and os.path.exists(script_file):
                command = f'sbatch {script_file} {gjf_file}'
                try:
                    subprocess.run(command, shell=True, check=True)
                    print(f'Successfully submitted: {gjf_file}')
                except subprocess.CalledProcessError as e:
                    print(f'Error submitting {gjf_file}: {e}')
            else:
                print(f'Missing file in {folder}: {gjf_file} or {script_file}')
            os.chdir(current_dir)  # 切换回原始工作目录
    os.chdir(current_dir)  # 确保函数结束时回到原始目录


# 计算RESP电荷拟合
def run_calcRESP_command():
    current_directory = os.getcwd()
    for item in os.listdir(current_directory):
        if os.path.isdir(item) and item.startswith("RESP"):
            dir_path = os.path.join(current_directory, item)
            # 输出开始信息
            print(f"开始执行 {item} 目录的RESP电荷拟合")
            command = "calcRESP.sh SP_gas.chk SP_solv.chk"
            os.chdir(dir_path)
            try:
                subprocess.run(command, shell=True, check=True)
                # 输出结束信息
                print(f"{item} 目录的RESP电荷拟合完毕")
            except subprocess.CalledProcessError as e:
                print(f"命令在目录 {dir_path} 中执行失败: {e}")
            # 返回到原始目录
            os.chdir(current_directory)


def run_function_in_background(func):
    def wrapper():
        # 封装的函数，用于在后台执行
        func()

    # 创建一个线程对象，目标函数是wrapper
    thread = threading.Thread(target=wrapper)
    # 启动线程
    thread.start()


# 可视化xyz结构
def vis_3Dxyz(xyz_file, width=400, height=400):
    with open(xyz_file, 'r') as file:
        xyz_data = file.read()

    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_data, "xyz")
    view.setStyle({'stick': {}})
    view.zoomTo()
    return view


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
