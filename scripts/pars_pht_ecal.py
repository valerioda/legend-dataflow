from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import pickle as pkl
from datetime import datetime

import lgdo.lh5_store as lh5
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pygama.math.histogram as pgh
from legendmeta import LegendMetadata
from legendmeta.catalog import Props
from matplotlib.colors import LogNorm
from pygama.pargen.ecal_th import *  # noqa: F403
from pygama.pargen.ecal_th import apply_cuts, calibrate_parameter
from pygama.pargen.utils import get_tcm_pulser_ids, load_data
from scipy.stats import binned_statistic

log = logging.getLogger(__name__)
mpl.use("agg")


def plot_baseline_timemap(
    data,
    figsize=(12, 8),
    fontsize=12,
    parameter="bl_mean",
    dx=1,
    n_spread=5,
    time_dx=180,
):
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["font.size"] = fontsize

    time_bins = np.arange(
        (np.amin(data["timestamp"]) // time_dx) * time_dx,
        ((np.amax(data["timestamp"]) // time_dx) + 2) * time_dx,
        time_dx,
    )

    mean = np.nanpercentile(data[parameter], 50)
    spread = mean - np.nanpercentile(data[parameter], 10)
    fig = plt.figure()
    plt.hist2d(
        data["timestamp"],
        data[parameter],
        bins=[
            time_bins,
            np.arange(mean - n_spread * spread, mean + n_spread * spread + dx, dx),
        ],
        norm=LogNorm(),
    )

    ticks, labels = plt.xticks()
    plt.xlabel(f"Time starting : {datetime.utcfromtimestamp(ticks[0]).strftime('%d/%m/%y %H:%M')}")
    plt.ylabel("Baseline Value")
    plt.ylim([mean - n_spread * spread, mean + n_spread * spread])

    plt.xticks(
        ticks,
        [datetime.utcfromtimestamp(tick).strftime("%H:%M") for tick in ticks],
    )
    plt.close()
    return fig


def bin_bl_stability(data, time_slice=180, parameter="bl_mean"):
    utime_array = data["timestamp"]
    select_bls = data[parameter].to_numpy()

    time_bins = np.arange(
        (np.amin(utime_array) // time_slice) * time_slice,
        ((np.amax(utime_array) // time_slice) + 2) * time_slice,
        time_slice,
    )
    # bin time values
    times_average = (time_bins[:-1] + time_bins[1:]) / 2

    def nanmedian(x):
        return np.nanpercentile(x, 50) if len(x) >= 10 else np.nan

    def error(x):
        return np.nanvar(x) / np.sqrt(len(x)) if len(x) >= 10 else np.nan

    par_average, _, _ = binned_statistic(
        data["timestamp"], select_bls, statistic=nanmedian, bins=time_bins
    )
    par_error, _, _ = binned_statistic(
        data["timestamp"], select_bls, statistic=error, bins=time_bins
    )

    return {"time": times_average, "baseline": par_average, "spread": par_error}


def bin_baseline(data, parameter="bl_mean-baseline", dx=1, bl_range=None):
    if bl_range is None:
        bl_range = [-500, 500]
    par_array = data.eval(parameter)
    bins = np.arange(bl_range[0], bl_range[1], dx)
    bl_array, bins, _ = pgh.get_hist(par_array, bins=bins)
    return {"bl_array": bl_array, "bins": (bins[1:] + bins[:-1]) / 2}


def baseline_tracking_plots(files, lh5_path, plot_options=None):
    if plot_options is None:
        plot_options = {}
    plot_dict = {}
    data = lh5.load_dfs(files, ["bl_mean", "baseline", "timestamp"], lh5_path)
    for key, item in plot_options.items():
        if item["options"] is not None:
            plot_dict[key] = item["function"](data, **item["options"])
        else:
            plot_dict[key] = item["function"](data)
    return plot_dict


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--files", help="files", nargs="*", type=str)
    argparser.add_argument("--tcm_filelist", help="tcm_filelist", type=str, required=True)
    argparser.add_argument("--cal_dict", help="cal_dict", nargs="*")

    argparser.add_argument("--configs", help="config", type=str, required=True)
    argparser.add_argument("--datatype", help="Datatype", type=str, required=True)
    argparser.add_argument("--timestamp", help="Timestamp", type=str, required=True)
    argparser.add_argument("--channel", help="Channel", type=str, required=True)

    argparser.add_argument("--log", help="log_file", type=str)

    argparser.add_argument("--plot_path", help="plot_path", type=str, required=False)
    argparser.add_argument("--save_path", help="save_path", type=str)
    argparser.add_argument("--results_path", help="results_path", type=str)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.DEBUG, filename=args.log, filemode="w")
    logging.getLogger("numba").setLevel(logging.INFO)
    logging.getLogger("parse").setLevel(logging.INFO)
    logging.getLogger("lgdo").setLevel(logging.INFO)
    logging.getLogger("h5py").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    database_dic = Props.read_from(args.cal_dict)
    hit_dict = database_dic[args.channel]["pars"]

    # get metadata dictionary
    configs = LegendMetadata(path=args.configs)
    channel_dict = configs.on(args.timestamp, system=args.datatype)["snakemake_rules"]
    channel_dict = channel_dict["pars_pht_ecal"]["inputs"]["ecal_config"][args.channel]
    kwarg_dict = Props.read_from(channel_dict)

    # convert plot functions from strings to functions and split off baseline and common plots
    for field, item in kwarg_dict["plot_options"].items():
        kwarg_dict["plot_options"][field]["function"] = eval(item["function"])

    bl_plots = kwarg_dict.pop("bl_plot_options")
    for field, item in bl_plots.items():
        bl_plots[field]["function"] = eval(item["function"])
    common_plots = kwarg_dict.pop("common_plots")

    energy_params = kwarg_dict.pop("energy_params")
    cal_energy_params = kwarg_dict.pop(
        "cal_energy_params", [energy_param + "_cal" for energy_param in energy_params]
    )
    cut_parameters = kwarg_dict.pop("cut_parameters", {})

    # load data in
    data, threshold_mask = load_data(
        args.files,
        f"{args.channel}/dsp",
        hit_dict,
        params=energy_params + list(cut_parameters) + ["timestamp", "trapTmax"],
        threshold=kwarg_dict["threshold"],
        return_selection_mask=True,
        cal_energy_param="trapTmax",
    )

    # get pulser mask from tcm files
    with open(args.tcm_filelist) as f:
        tcm_files = f.read().splitlines()
    tcm_files = sorted(np.unique(tcm_files))
    ids, mask = get_tcm_pulser_ids(
        tcm_files, args.channel, kwarg_dict.pop("pulser_multiplicity_threshold")
    )
    data["is_pulser"] = mask[threshold_mask]

    # apply cuts
    data, hit_dict = apply_cuts(
        data,
        hit_dict,
        cut_parameters,
        kwarg_dict["final_cut_field"],
    )
    log.info(f"selected events n. {len(data)}")

    # run calibrations
    results_dict = {}
    plot_dict = {}
    full_object_dict = {}
    for energy_param, cal_energy_param in zip(energy_params, cal_energy_params):
        log.info(f"\nCalibration for {energy_param}")
        # cal parameters from hit dict
        cal_pars = hit_dict["operations"][cal_energy_param]["parameters"]
        if len(cal_pars) < kwarg_dict["deg"] + 1:
            cal_params = {"a": 0, "b": cal_pars["a"], "c": cal_pars["b"]}
        else:
            cal_params = cal_pars
        # fix parameters
        fix_params = {}
        fixed = kwarg_dict.pop("fixed")
        for i, par in enumerate(cal_params):
            if i in fixed:
                fix_params[i] = cal_params[par]
        full_object_dict = calibrate_parameter(
            energy_param,
            cal_energy_param=cal_energy_param,
            selection_string=f"({kwarg_dict.pop('final_cut_field')})&(~is_pulser)",
            fix_params=fix_params,
            **kwarg_dict,
        )
        full_object_dict[cal_energy_param].calibrate_parameter(data)
        results_dict[cal_energy_param] = full_object_dict[cal_energy_param].get_results_dict(data)
        hit_dict.update(full_object_dict[cal_energy_param].hit_dict)
        if ~np.isnan(full_object_dict[cal_energy_param].pars).all():
            plot_dict[cal_energy_param] = (
                full_object_dict[cal_energy_param].fill_plot_dict(data).copy()
            )
    log.info("Finished all calibrations")

    # get baseline plots and save all plots to file
    if args.plot_path:
        common_dict = baseline_tracking_plots(
            sorted(args.files), f"{args.channel}/dsp", plot_options=bl_plots
        )

        for plot in list(common_dict):
            if plot not in common_plots:
                plot_item = common_dict.pop(plot)
                plot_dict.update({plot: plot_item})

        pathlib.Path(os.path.dirname(args.plot_path)).mkdir(parents=True, exist_ok=True)

        for key, item in plot_dict.items():
            if isinstance(item, dict) and len(item) > 0:
                param_dict = {}
                for plot in common_plots:
                    if plot in item:
                        param_dict.update({plot: item[plot]})
                common_dict.update({key: param_dict})
        plot_dict["common"] = common_dict

        with open(args.plot_path, "wb") as f:
            pkl.dump(plot_dict, f, protocol=pkl.HIGHEST_PROTOCOL)

    # save output dictionary
    output_dict = {"pars": hit_dict, "results": results_dict}
    with open(args.save_path, "w") as fp:
        pathlib.Path(os.path.dirname(args.save_path)).mkdir(parents=True, exist_ok=True)
        json.dump(output_dict, fp, indent=4)

    # save calibration objects
    with open(args.results_path, "wb") as fp:
        pathlib.Path(os.path.dirname(args.results_path)).mkdir(parents=True, exist_ok=True)
        pkl.dump(full_object_dict, fp, protocol=pkl.HIGHEST_PROTOCOL)
