import json
import pandas as pd
from PEMD.analysis import coordination
from PEMD.analysis import residence_time

work_dir = '/home/tsd/polymer/MD/PEO/LI_EO/0.05/450/1_sample/MD_dir'
tpr_file = 'nvt_prod.tpr'
xtc_file = 'nvt_prod.xtc'
run_start = 0
run_end = 10001  # step

distance_dict = {"polymer": 3.7, "anion": 3.3, 'SN':3.7}        # Li-PEO 3.7; Li-TFSI 3.3; Li-SN 3.7
select_dict = {
    "cation": "resname LIP and name Li",
    "anion": "resname NSC and name OBT",
    "polymer": "resname MOL and name O",
    "SN": 'resname SN and name N'
}

run = coordination.load_md_trajectory(work_dir, tpr_file, xtc_file)
times = residence_time.times_array(run, run_start, run_end, time_step=5)
acf_avg = residence_time.calc_neigh_corr(run, distance_dict, select_dict, run_start, run_end)

# 归一化自相关函数
acf_avg_norm = {}
species_list = list(acf_avg.keys())
for kw in species_list:
    if kw in acf_avg:
        acf_avg_norm[kw] = acf_avg[kw] / acf_avg[kw][0]  # 防止除以零的错误处理

# 准备将时间和归一化后的自相关函数保存在同一个CSV文件中
acf_df = pd.DataFrame({'Time (ps)': times})
for key, value in acf_avg_norm.items():
    acf_df[key + ' ACF'] = pd.Series(value)

# 将数据保存到CSV文件
acf_df.to_csv('residence_time.csv', index=False)

data = residence_time.fit_residence_time(times, acf_avg, 500, 5)

with open('residence_time.json', 'w') as f:
    json.dump(data, f, indent=4)





