from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import pickle as pkl
import re

import numpy as np
import pandas as pd
from legendmeta import LegendMetadata
from legendmeta.catalog import Props
from pygama.pargen.ecal_th import *  # noqa: F403
from pygama.pargen.ecal_th import apply_cuts, high_stats_fitting
from pygama.pargen.utils import get_tcm_pulser_ids, load_data
from util.FileKey import ChannelProcKey, ProcessingFileKey

log = logging.getLogger(__name__)


def update_cal_dicts(cal_dicts, update_dict):
    if re.match(r"(\d{8})T(\d{6})Z", next(iter(cal_dicts))):
        for tstamp in cal_dicts:
            if tstamp in update_dict:
                cal_dicts[tstamp].update(update_dict[tstamp])
            else:
                cal_dicts[tstamp].update(update_dict)
    else:
        cal_dicts.update(update_dict)
    return cal_dicts


argparser = argparse.ArgumentParser()
argparser.add_argument("--input_files", help="files", type=str, nargs="*", required=True)
argparser.add_argument("--tcm_filelist", help="tcm_filelist", type=str, nargs="*", required=True)
argparser.add_argument("--ecal_file", help="ecal_file", type=str, nargs="*", required=True)
argparser.add_argument("--eres_file", help="eres_file", type=str, nargs="*", required=True)
argparser.add_argument("--inplots", help="eres_file", type=str, nargs="*", required=True)

argparser.add_argument("--configs", help="configs", type=str, required=True)
argparser.add_argument("--timestamp", help="Datatype", type=str, required=True)
argparser.add_argument("--datatype", help="Datatype", type=str, required=True)
argparser.add_argument("--channel", help="Channel", type=str, required=True)

argparser.add_argument("--log", help="log_file", type=str)

argparser.add_argument("--plot_file", help="plot_file", type=str, nargs="*", required=False)
argparser.add_argument("--hit_pars", help="hit_pars", nargs="*", type=str)
argparser.add_argument("--fit_results", help="fit_results", nargs="*", type=str)
args = argparser.parse_args()

logging.basicConfig(level=logging.DEBUG, filename=args.log, filemode="w")
logging.getLogger("numba").setLevel(logging.INFO)
logging.getLogger("parse").setLevel(logging.INFO)
logging.getLogger("lgdo").setLevel(logging.INFO)
logging.getLogger("h5py").setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.INFO)


def run_splitter(files):
    """
    Returns list containing lists of each run
    """

    runs = []
    run_files = []
    for file in files:
        fk = ProcessingFileKey.get_filekey_from_pattern(os.path.basename(file))
        if f"{fk.period}-{fk.run}" not in runs:
            runs.append(f"{fk.period}-{fk.run}")
            run_files.append([])
        for i, run in enumerate(runs):
            if run == f"{fk.period}-{fk.run}":
                run_files[i].append(file)
    return run_files


configs = LegendMetadata(path=args.configs)
channel_dict = configs.on(args.timestamp, system=args.datatype)["snakemake_rules"]
channel_dict = channel_dict["pars_pht_partcal"]["inputs"]["pars_pht_partcal_config"][args.channel]
kwarg_dict = Props.read_from(channel_dict)

cal_dict = {}
results_dicts = {}
results_ecal = {}
if isinstance(args.ecal_file, list):
    for ecal in args.ecal_file:
        with open(ecal) as o:
            cal = json.load(o)

        fk = ChannelProcKey.get_filekey_from_pattern(os.path.basename(ecal))
        cal_dict[fk.timestamp] = cal["pars"]
        results_dicts[fk.timestamp] = cal["results"]
        results_ecal = cal["results"]["ecal"]
else:
    with open(args.ecal_file) as o:
        cal = json.load(o)

    fk = ChannelProcKey.get_filekey_from_pattern(os.path.basename(args.ecal_file))
    cal_dict[fk.timestamp] = cal["pars"]
    results_dicts[fk.timestamp] = cal["results"]
    results_ecal = cal["results"]["ecal"]

object_dict = {}
if isinstance(args.eres_file, list):
    for ecal in args.eres_file:
        with open(ecal, "rb") as o:
            cal = pkl.load(o)
        fk = ChannelProcKey.get_filekey_from_pattern(os.path.basename(ecal))
        object_dict[fk.timestamp] = cal
else:
    with open(args.eres_file, "rb") as o:
        cal = pkl.load(o)
    fk = ChannelProcKey.get_filekey_from_pattern(os.path.basename(args.eres_file))
    object_dict[fk.timestamp] = cal

inplots_dict = {}
if args.inplots:
    if isinstance(args.inplots, list):
        for ecal in args.inplots:
            with open(ecal, "rb") as o:
                cal = pkl.load(o)
            fk = ChannelProcKey.get_filekey_from_pattern(os.path.basename(ecal))
            inplots_dict[fk.timestamp] = cal
    else:
        with open(args.inplots, "rb") as o:
            cal = pkl.load(o)
        fk = ChannelProcKey.get_filekey_from_pattern(os.path.basename(args.inplots))
        inplots_dict[fk.timestamp] = cal


if "plot_options" in kwarg_dict:
    for field, item in kwarg_dict["plot_options"].items():
        kwarg_dict["plot_options"][field]["function"] = eval(item["function"])


# sort files in dictionary where keys are first timestamp from run
if isinstance(args.input_files, list):
    files = []
    for file in args.input_files:
        with open(file) as f:
            files += f.read().splitlines()
else:
    with open(args.input_files) as f:
        files = f.read().splitlines()

files = sorted(
    np.unique(files)
)  # need this as sometimes files get double counted as it somehow puts in the p%-* filelist and individual runs also

final_dict = {}
all_file = run_splitter(sorted(files))
for filelist in all_file:
    fk = ProcessingFileKey.get_filekey_from_pattern(os.path.basename(sorted(filelist)[0]))
    timestamp = fk.timestamp
    final_dict[timestamp] = sorted(filelist)

params = [
    kwarg_dict["final_cut_field"],
    "timestamp",
]
params += kwarg_dict["energy_params"]

energy_params = kwarg_dict.pop("energy_params")
cal_energy_params = kwarg_dict.pop(
    "cal_energy_params", [energy_param + "_cal" for energy_param in energy_params]
)
cut_parameters = kwarg_dict.pop("cut_parameters", {})

# load data in
data, threshold_mask = load_data(
    final_dict,
    f"{args.channel}/dsp",
    cal_dict,
    params=energy_params + list(cut_parameters) + ["timestamp", "trapTmax"],
    threshold=kwarg_dict["threshold"],
    return_selection_mask=True,
    cal_energy_param=kwarg_dict["energy_params"][0],
)

# get pulser mask from tcm files
if isinstance(args.tcm_filelist, list):
    tcm_files = []
    for file in args.tcm_filelist:
        with open(file) as f:
            tcm_files += f.read().splitlines()
else:
    with open(args.tcm_filelist) as f:
        tcm_files = f.read().splitlines()

tcm_files = sorted(np.unique(tcm_files))
ids, mask = get_tcm_pulser_ids(
    tcm_files, args.channel, kwarg_dict.pop("pulser_multiplicity_threshold")
)
data["is_pulser"] = mask[threshold_mask]

for tstamp in cal_dict:
    if tstamp not in np.unique(data["run_timestamp"]):
        row = {key: [False] if data.dtypes[key] == "bool" else [np.nan] for key in data}
        row["run_timestamp"] = tstamp
        row = pd.DataFrame(row)
        data = pd.concat([data, row])

# apply cuts
data, cal_dict = apply_cuts(
    data,
    cal_dict,
    cut_parameters,
    kwarg_dict["final_cut_field"],
)
log.info(f"selected events n. {len(data)}")

# run energy supercal
part_results = {}
plot_dict = {}
full_object_dict = {}
for energy_param, cal_energy_param in zip(energy_params, cal_energy_params):
    pk_cal_pars = results_ecal[cal_energy_param].get("pk_cal_pars", None)
    full_object_dict = high_stats_fitting(
        energy_param,
        cal_energy_param=cal_energy_param,
        selection_string=f"({kwarg_dict.pop('final_cut_field')})&(~is_pulser)",
        pk_cal_pars=pk_cal_pars,
        **kwarg_dict,
    )
    full_object_dict[energy_param].update_calibration(data)
    part_results[energy_param] = full_object_dict[energy_param].get_results_dict(data)
    cal_dict.update(full_object_dict[cal_energy_param].cal_dict)
    if full_object_dict[energy_param].results:
        plot_dict[energy_param] = full_object_dict[energy_param].fill_plot_dict(data).copy()
log.info("Finished all calibrations")


if args.plot_file:
    common_dict = plot_dict.pop("common") if "common" in list(plot_dict) else None

    if isinstance(args.plot_file, list):
        for plot_file in args.plot_file:
            fk = ChannelProcKey.get_filekey_from_pattern(os.path.basename(plot_file))
            if args.inplots:
                out_plot_dict = inplots_dict[fk.timestamp]
                out_plot_dict.update({"partition_ecal": plot_dict})
            else:
                out_plot_dict = {"partition_ecal": plot_dict}

            if "common" in list(out_plot_dict) and common_dict is not None:
                out_plot_dict["common"].update(common_dict)
            elif common_dict is not None:
                out_plot_dict["common"] = common_dict

            pathlib.Path(os.path.dirname(plot_file)).mkdir(parents=True, exist_ok=True)
            with open(plot_file, "wb") as w:
                pkl.dump(out_plot_dict, w, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        if args.inplots:
            fk = ChannelProcKey.get_filekey_from_pattern(os.path.basename(args.plot_file))
            out_plot_dict = inplots_dict[fk.timestamp]
            out_plot_dict.update({"partition_ecal": plot_dict})
        else:
            out_plot_dict = {"partition_ecal": plot_dict}
        if "common" in list(out_plot_dict) and common_dict is not None:
            out_plot_dict["common"].update(common_dict)
        elif common_dict is not None:
            out_plot_dict["common"] = common_dict
        pathlib.Path(os.path.dirname(args.plot_file)).mkdir(parents=True, exist_ok=True)
        with open(args.plot_file, "wb") as w:
            pkl.dump(out_plot_dict, w, protocol=pkl.HIGHEST_PROTOCOL)


for out in sorted(args.hit_pars):
    fk = ChannelProcKey.get_filekey_from_pattern(os.path.basename(out))
    final_hit_dict = {
        "pars": cal_dict[fk.timestamp],
        "results": {
            "ecal": results_dicts[fk.timestamp],
            "partition_ecal": part_results,
        },
    }
    pathlib.Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
    with open(out, "w") as w:
        json.dump(final_hit_dict, w, indent=4)

for out in args.fit_results:
    fk = ChannelProcKey.get_filekey_from_pattern(os.path.basename(out))
    final_object_dict = {
        "ecal": object_dict[fk.timestamp],
        "partition_ecal": full_object_dict,
    }
    pathlib.Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as w:
        pkl.dump(final_object_dict, w, protocol=pkl.HIGHEST_PROTOCOL)
