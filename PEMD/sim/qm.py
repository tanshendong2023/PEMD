"""
Polymer model building tools.

Developed by: Tan Shendong
Date: 2024.01.18
"""


import os
import time
import glob
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
from simple_slurm import Slurm
from PEMD.model import PEMD_lib
from PEMD.sim_API import gaussian


def unit_conformation_search(mol, unit_name, out_dir, length, numconf=10,charge =0, multiplicity=1, memory='64GB',
                        core= 32, chk = True, opt_method='B3LYP', opt_basis='6-311+g(d,p)',
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
    time.sleep(10)

    while True:
        status = PEMD_lib.get_slurm_job_status(job_id)
        if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
            print("crest finish, executing the gaussian task...")

            order_structures = PEMD_lib.orderxyz_energy_crest('crest_conformers.xyz', numconf)
            os.chdir(origin_dir)
            conformer_search_gaussian(out_dir, order_structures, unit_name, length, charge, multiplicity, memory, core, chk,
                            opt_method, opt_basis, dispersion_corr, freq, solv_model, custom_solv)
            break
        else:
            print("crest conformer search not finish, waiting...")
            time.sleep(30)


def conformer_search_gaussian(out_dir, structures, unit_name, length, charge, multiplicity, memory, core, chk,
                    opt_method, opt_basis, dispersion_corr, freq, solv_model, custom_solv):
    # 获取当前工作目录的路径
    current_directory = os.getcwd()
    job_ids = []
    structure_directory = current_directory + '/' + out_dir + f'{unit_name}_conf_g16'
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

        job_id = slurm.sbatch(f'g16 {structure_directory}/{unit_name}_{i + 1}.gjf')
        time.sleep(10)
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
            print("All gaussian tasks finished, order structure with energy calculated by gaussian...")
            # 执行下一个任务的代码...
            sorted_df = PEMD_lib.orderlog_energy_gaussian(structure_directory)
            break
        else:
            print("g16 conformer search not finish, waiting...")
            time.sleep(30)  # 等待30秒后再次检查
    return sorted_df


def poly_conformer_search(mol, out_dir, unit_name, length, max_conformers=1000, top_n_MMFF=100, top_n_xtb=10,
                          epsilon=30,charge =0, multiplicity=1, memory='64GB', core= 32, chk = True,
                          opt_method='B3LYP', opt_basis='6-311+g(d,p)', dispersion_corr = 'em=GD3BJ', freq = 'freq',
                          solv_model = 'scrf=(pcm,solvent=generic,read)', custom_solv='eps=5.0 \nepsinf=2.1'):

    out_dir = out_dir + '/'
    PEMD_lib.build_dir(out_dir)

    """从分子构象中搜索能量最低的构象"""
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
    conformer_search_gaussian(out_dir, order_structures, unit_name, length, charge, multiplicity, memory,
                      core, chk, opt_method, opt_basis, dispersion_corr, freq, solv_model, custom_solv,)


def calc_resp_gaussian(out_dir, unit_name, log_filename, charge, multiplicity, memory, core, eps, epsinf,):

    resp_dir = os.path.join(out_dir, 'resp_work')
    os.makedirs(resp_dir, exist_ok=True)
    origin_dir = os.getcwd()
    os.chdir(resp_dir)


    xyz_filename = log_filename.replace('.log', '.xyz')

    log_filepath = out_dir + f'{unit_name}_conf_g16' + '/' + log_filename
    xyz_filepath = out_dir + f'{unit_name}_conf_g16' + '/' + xyz_filename
    PEMD_lib.log_to_xyz(log_filepath, xyz_filepath)

    # 从XYZ文件中读取内容
    with open(xyz_filepath, 'r') as file:
        xyz_content = file.read()

    formatted_coordinates = ""
    lines = xyz_content.split('\n')
    for line in lines[2:]:  # Skip the first two lines (atom count and comment)
        elements = line.split()
        if len(elements) >= 4:
            atom_type, x, y, z = elements[0], elements[1], elements[2], elements[3]
            formatted_coordinates += f"  {atom_type}  {x:>12}{y:>12}{z:>12}\n"

    # RESP template
    template = f"""%nprocshared={core}
    %mem={memory}
    %chk=opt.chk
    # B3LYP/TZVP em=GD3BJ opt

    opt

    {charge} {multiplicity}
    [GEOMETRY]    
    --link1--
    %nprocshared={core}
    %mem={memory}
    %oldchk=opt.chk
    %chk=SP_gas.chk
    # B3LYP/def2TZVP em=GD3BJ geom=allcheck

    --link1--
    %nprocshared={core}
    %mem={memory}
    %oldchk=opt.chk
    %chk=SP_solv.chk
    # B3LYP/def2TZVP em=GD3BJ scrf=(pcm,solvent=generic,read) geom=allcheck

    eps={eps}
    epsinf={epsinf}\n\n"""
    gaussian_input = template.replace("[GEOMETRY]", formatted_coordinates)

    out_file = out_dir + f'{unit_name}_conf_g16' + f'{log_filename.replace(".log", "_resp2.com")}'
    with open(out_file, 'w') as file:
        file.write(gaussian_input)

    structure_directory = os.getcwd() + '/' + out_dir + f'{unit_name}_conf_g16'

    slurm = Slurm(J='g16',
                  N=1,
                  n=f'{core}',
                  output=f'{structure_directory}/slurm.{Slurm.JOB_ARRAY_MASTER_ID}.out'
                  )

    # com_file = os.path.join(structure_directory, f"{base_filename}_{i + 1}_conf_1.com")
    #         print(f'g16 {structure_directory}/{base_filename}_{i+1}_conf_1.com')
    job_id = slurm.sbatch(f'g16 {structure_directory}/{log_filename.replace(".log", "_resp2.com")}')
    time.sleep(10)

    while True:
        status = PEMD_lib.get_slurm_job_status(job_id)
        if status in ['COMPLETED', 'FAILED', 'CANCELLED']:
            print("RESP calculation finish, executing the gaussian task...")
            break
        else:
            print("RESP calculation not finish, waiting...")
            time.sleep(30)








