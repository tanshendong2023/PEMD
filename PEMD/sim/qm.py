"""
Polymer model building tools.

Developed by: Tan Shendong
Date: 2024.01.18
"""


import os
import time
from rdkit import Chem
from rdkit.Chem import AllChem
from simple_slurm import Slurm
from PEMD.model import PEMD_lib
from PEMD.sim_API import gaussian


def conformation_search(mol, unit_name, out_dir, length, numconf=10,charge =0, multiplicity=1, memory='64GB',
                        core='32', chk = True, opt_method='B3LYP', opt_basis='6-311+g(d,p)',
                        dispersion_corr = 'em=GD3BJ', freq = 'freq',
                        solv_model = 'scrf=(pcm,solvent=generic,read)',
                        custom_solv='eps=5.0 \nepsinf=2.1'):

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

    while True:
        status = PEMD_lib.get_slurm_job_status(job_id)
        if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
            print("crest finish, executing the gaussian task...")
            order_structures = PEMD_lib.crest_lowest_energy_str('crest_conformers.xyz', numconf)
            os.chdir(origin_dir)
            save_structures(out_dir, order_structures, unit_name, length, charge, multiplicity, memory, core, chk,
                            opt_method, opt_basis, dispersion_corr, freq, solv_model, custom_solv)
            break
        else:
            print("crest conformer search not finish, waiting...")
            time.sleep(30)


def save_structures(out_dir, structures, unit_name, length, charge, multiplicity, memory, core, chk,
                    opt_method, opt_basis, dispersion_corr, freq, solv_model, custom_solv):
    # 获取当前工作目录的路径
    current_directory = os.getcwd()
    job_ids = []
    structure_directory = current_directory + '/' + out_dir + f'{unit_name}_conf_g16'
    print(structure_directory)
    os.makedirs(structure_directory, exist_ok=True)

    for i, structure in enumerate(structures):

        # 在新创建的目录中保存XYZ文件
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

        # com_file = os.path.join(structure_directory, f"{base_filename}_{i + 1}_conf_1.com")
        #         print(f'g16 {structure_directory}/{base_filename}_{i+1}_conf_1.com')
        job_id = slurm.sbatch(f'g16 {structure_directory}/{unit_name}_{i + 1}_conf_1.com')
        job_ids.append(job_id)

    # 检查所有任务的状态
    while True:
        all_completed = True
        for job_id in job_ids:
            status = PEMD_lib.get_slurm_job_status(job_id)
            if status not in ['COMPLETED', 'FAILED', 'CANCELLED']:
                all_completed = False
                break

        if all_completed:
            print("All gaussian tasks finished, find the lowest energy structure...")
            # 执行下一个任务的代码...
            PEMD_lib.g16_lowest_energy_str(structure_directory, unit_name, length)
            break
        else:
            print("g16 conformer search not finish, waiting...")
            time.sleep(30)  # 等待30秒后再次检查


