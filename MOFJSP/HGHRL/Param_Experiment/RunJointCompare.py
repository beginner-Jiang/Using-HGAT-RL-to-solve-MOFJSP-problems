# -*- coding: utf-8 -*-
"""Run the 3-seed joint-vs-isolated comparison and write results to Param."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKER = os.path.join(HERE, "JointCompare.py")
RESULTS = r"D:\360MoveData\Users\lenovo\Desktop\撰写的论文\MOFJSP(修改后的完整项目)\Figure_And_File\HGHRL\Param_Experiment\joint4000"
os.makedirs(RESULTS, exist_ok=True)

jobs = []
for seed in (42, 43, 44):
    for config in ("C0", "Cjoint"):
        path = os.path.join(RESULTS, "%s_s%d.json" % (config, seed))
        jobs.append((config, seed, path))

procs = []
for config, seed, path in jobs:
    log = open(path + ".log", "w", encoding="utf-8")
    p = subprocess.Popen([sys.executable, "-u", WORKER, "--config", config,
                          "--seed", str(seed), "--epochs", "4000", "--envs", "4",
                          "--out", path], stdout=log, stderr=subprocess.STDOUT)
    procs.append((p, log, path))
for p, log, path in procs:
    code = p.wait()
    log.close()
    print(os.path.basename(path), "exit", code, flush=True)

rows = [json.load(open(path, encoding="utf-8")) for _, _, path in jobs
        if os.path.exists(path)]
json.dump(rows, open(os.path.join(RESULTS, "summary.json"), "w", encoding="utf-8"), indent=2)
print("completed", len(rows), "of", len(jobs))



