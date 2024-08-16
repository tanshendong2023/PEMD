# Copyright (c) 2024. PEMD developers. All rights reserved.
# Distributed under the terms of the MIT License.

# ******************************************************************************
# simulation.slurm module
# ******************************************************************************

import os
import subprocess

class PEMDSlurm:
    def __init__(
            self,
            work_dir,
            script_name="sub.script",
            job_name="my_job",
            nodes=1,
            ntasks_per_node=64,
            partition="standard",
    ):

        self.work_dir = work_dir
        self.script_name = script_name
        self.job_name = job_name
        self.partition = partition
        self.nodes = nodes
        self.ntasks_per_node = ntasks_per_node
        self.commands = []

    def add_command(self, command):

        self.commands.append(command)

    def gen_header(self):

        header_lines = [
            "#!/bin/bash",
            f"#SBATCH -J {self.job_name}",
            f"#SBATCH -N {self.nodes}",
            f"#SBATCH -n {self.ntasks_per_node}",
            f"#SBATCH -p {self.partition}",
            f"#SBATCH -o {self.work_dir}/slurm.%A.out",
        ]

        return "\n".join(header_lines)

    def generate_script(self):

        slurm_script_content = self.gen_header()
        slurm_script_content += "\n\n" + "\n".join(self.commands)

        script_path = os.path.join(self.work_dir, self.script_name)
        with open(script_path, "w") as script_file:
            script_file.write(slurm_script_content)

        print(f"SLURM script generated at: {script_path}")
        return script_path

    def submit_job(self):

        script_path = os.path.join(self.work_dir, self.script_name)
        try:
            result = subprocess.run(f"sbatch {script_path}", shell=True, check=True, text=True, capture_output=True)
            print(f"SLURM job submitted: {result.stdout.strip()}")
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error submitting SLURM job: {e}")
            return e.stderr