#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Pi-Yueh Chuang <pychuang@gwu.edu>
#
# Distributed under terms of the BSD 3-Clause license.

"""Check running jobs on Slurm.
"""
import sys
import re
import time
import pathlib
import subprocess
import datetime
import termcolor

basedir = pathlib.Path(__file__).resolve().parents[1]

jobids = subprocess.run(
    ["squeue", "-u", "x_nvidiaA101", "--states", "R", "--format", "%i"],
    capture_output=True, check=True,
)

jobids = jobids.stdout.decode("utf-8")
jobids = re.findall(r"^(\d+?)(?:_\d+?)$", jobids, re.MULTILINE)

logs = []
names = []
for jobid in jobids:
    info = subprocess.run(["scontrol", "show", "job", jobid], capture_output=True, check=True)

    workdir = re.search(r"^\s*WorkDir=(.*?)$", info.stdout.decode("utf-8"), re.MULTILINE)
    workdir = pathlib.Path(workdir.group(1)).resolve()
    names.append(workdir.relative_to(basedir))

    logs.append(max(
        workdir.joinpath("logs").glob("run-*.log"),
        key=lambda inp: int(inp.name.replace("run-", "").replace(".log", ""))
    ))

try:
    while True:
        msg = str(datetime.datetime.utcnow().replace(microsecond=0)) + "\n"

        for jobid, name, logfile in zip(jobids, names, logs):
            result = subprocess.run(["tail", "-n", "50", str(logfile)], capture_output=True, check=True)

            result = re.findall(
                r"^.*?\[step:\s+?\d+?\]\sloss=.*$",
                result.stdout.decode("utf-8"),
                re.MULTILINE
            )
            result = termcolor.colored("no data", "red") if len(result) == 0 else result[-1]

            msg += f"({termcolor.colored(jobid, 'red')}, {termcolor.colored(name, 'yellow')}): "
            msg += f"{result}\n"

        print(msg)
        sys.stdout.write("\x1b[1A"*(len(jobids)+2)) # cursor back up to the first line
        time.sleep(1)
except KeyboardInterrupt:
    sys.stdout.write("\x1b[1B"*(len(jobids)+2)) # cursor goes to the last line
    print("\rMonitor stopped.")  # \r move the cursor to the beginning of the current line
    sys.exit(0)
