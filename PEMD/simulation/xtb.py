# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# Module Docstring
# ******************************************************************************

import os
import subprocess
from PEMD.simulation.slurm import PEMDSlurm

class PEMDXtb:
    def __init__(
            self,
            work_dir,
            xyz_filename,
            outfile_headname,
            epsilon=80.4,
            chg=0,
            mult=1
    ):

        self.work_dir = work_dir
        self.xyz_filename = xyz_filename
        self.xyz_filepath = os.path.join(work_dir, xyz_filename)
        self.epsilon = epsilon
        self.chg = chg
        self.mult = mult
        self.outfile_headname = outfile_headname

    def run_local(self):

        command = (
            f"xtb {self.xyz_filename} --opt --gbsa={self.epsilon} "
            f"--chrg={self.chg} --uhf={self.mult} --ceasefiles --namespace {self.outfile_headname}"
        )

        try:
            result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
            return result.stdout  # Return the standard output from the XTB command
        except subprocess.CalledProcessError as e:
            print(f"Error executing XTB: {e}")
            return e.stderr  # Return the error output if the command fails

    def gen_slurm(self, ):

        slurm_script = PEMDSlurm(
            self.work_dir,
            script_name="sub.xtb",
            job_name="xtb",
            nodes=1,
            ntasks_per_node=64,
            partition="standard",
        )

        slurm_script.add_command(
            f"xtb {self.xyz_filename} --opt --gbsa={self.epsilon} "
            f"--chrg={self.chg} --uhf={self.mult} --ceasefiles --namespace {self.outfile_headname}"
        )

        # Generate the SLURM script
        slurm_script.generate_script()

        return slurm_script

    def run_slurm(self):

        slurm_script = self.gen_slurm()
        slurm_script.submit_job()
