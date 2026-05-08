from __future__ import annotations

import argparse
import json
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from dbetto import TextDB
from dbetto.catalog import Props
import lh5
from lgdo import Array, Table
from pygama.pargen.dsp_optimize import run_one_dsp

from ..utils import alias_table, build_log

warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def _replace_list_with_array(dic):
    """Recursively replace list values with NumPy ``float32`` arrays.

    Parameters
    ----------
    dic : dict
        Nested dictionary whose list values should be converted.

    Returns
    -------
    dict
        *dic* modified in-place, with all list values replaced by
        :class:`numpy.ndarray` of dtype ``float32``.
    """
    for key, value in dic.items():
        if isinstance(value, dict):
            dic[key] = _replace_list_with_array(value)
        elif isinstance(value, list):
            dic[key] = np.array(value, dtype="float32")
        else:
            pass
    return dic



def build_tier_xtk() -> None:
    """Build the XTK tier from raw LH5 data.

    Notes
    -----
    **Command-line arguments**

    ``--configs`` : str
        Path to the dataflow configuration directory (TextDB-compatible).
    ``--table-map`` : str, optional
        JSON-encoded ``{channel: lh5_table_path}`` mapping.
    ``--log`` : str, optional
        Path to the log file.
    ``--datatype`` : str
        Data-type identifier used to select the active configuration (e.g.
        ``cal``, ``phy``).
    ``--timestamp`` : str
        Run timestamp used to select the active configuration.
    ``--tier`` : str
        Tier label (e.g. ``xtk``).
    ``--pars-file`` : list of str
        Database parameter files (``.json`` / ``.yaml``) containing
        per-channel DSP parameters.
    ``--input`` : str
        Path to the input raw LH5 file.
    ``--output`` : str
        Path for the output DSP LH5 file.
    """
    # CLI config
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--configs", help="path to dataflow config files", required=True
    )
    argparser.add_argument(
        "--table-map",
        help="mapping from channel to table name",
        required=False,
        type=str,
    )
    argparser.add_argument("--log", help="log file name")

    argparser.add_argument("--datatype", help="datatype", type=str, default="phy")
    argparser.add_argument("--timestamp", help="timestamp", required=True)
    argparser.add_argument("--tier", help="tier", type=str, default="xtk")

    argparser.add_argument(
        "--pars-file", help="database file for HPGes", nargs="*", default=[]
    )
    argparser.add_argument("--input", help="input file")

    argparser.add_argument("--output", help="output file")
    args = argparser.parse_args()

    table_map = json.loads(args.table_map) if args.table_map is not None else None

    df_configs = TextDB(args.configs, lazy=True)
    config_dict = df_configs.on(args.timestamp, system=args.datatype).snakemake_rules
    config_dict = config_dict[f"tier_{args.tier}"]

    log = build_log(config_dict, args.log, fallback=__name__)

    settings_dict = config_dict.options.get("settings", {})
    if isinstance(settings_dict, str):
        settings_dict = Props.read_from(settings_dict)

    chan_cfg_map = config_dict.inputs.processing_chain

    # now construct the dictionary of DSP configs for build_dsp()
    xtk_cfg_tbl_dict = {}
    for chan, file in chan_cfg_map.items():
        if chan in table_map:
            input_tbl_name = table_map[chan] if table_map is not None else chan + "/raw"
        else:
            continue

        # check if the raw tables are all existing
        if len(lh5.ls(args.input, input_tbl_name)) > 0:
            xtk_cfg_tbl_dict[input_tbl_name] = Props.read_from(file)
            msg = f"found table {input_tbl_name} in {args.input}"
            log.debug(msg)
        else:
            msg = f"table {input_tbl_name} not found in {args.input} skipping"
            log.info(msg)

    if len(xtk_cfg_tbl_dict) == 0:
        msg = f"could not find any of the requested channels in {args.input}"
        raise RuntimeError(msg)

    # par files
    db_files = [
        par_file
        for par_file in args.pars_file
        if Path(par_file).suffix in (".json", ".yaml", ".yml")
    ]

    database_dict = _replace_list_with_array(
        Props.read_from(db_files, subst_pathvar=True)
    )
    database_dict = {
        (table_map[chan].split("/")[0] if chan in table_map else chan): dic
        for chan, dic in database_dict.items()
    }
    log.info("loaded database files")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    start = time.time()

    # --- Processing Loop ---
    log.info("Starting XTK processing")
    current_spms = []
    processed_any = False
    corr_size = None

    for ged, spm_map in database_dict.items():
        ged_group = f"{ged}/raw"
        
        # Verify the Ge table exists before trying to read
        if len(lh5.ls(args.input, ged_group)) == 0:
            log.warning(f"Table {ged_group} not found in {args.input}. Skipping Ge detector {ged}.")
            continue

        raw_data = lh5.read(ged_group, args.input)
        processed_any = True

        for spm, spm_db in spm_map.items():
            log.debug(f"Processing Ge-SiPM pair: {ged} - {spm}")

            # Run DSP
            xtk_data = run_one_dsp(raw_data, chan_cfg_map[ged], db_dict=spm_db)
            
            # Capture correction size for zero-filling later
            if corr_size is None:
                corr_size = xtk_data["xtalk_correction"].nda.shape[1]

            if spm in current_spms:
                log.info(f"Channel {spm} already exists; summing corrections.")
                xtalk_corr_pre = lh5.read(f"{spm}/xtk/xtalk_correction", args.output).nda
                xtalk_corr_sum = xtk_data["xtalk_correction"].nda + xtalk_corr_pre
                xtk_data = Table(col_dict={
                    "xtalk_correction": Array(xtalk_corr_sum)
                })
            else:
                current_spms.append(spm)

            lh5.write(xtk_data, group=spm, name="xtk", lh5_file=args.output, wo_mode="overwrite")

    # --- Optional: Zero-fill remaining channels ---
    if processed_any and corr_size is not None:
        log.info("Writing zero-correction for remaining channels")
        # You would iterate over your full SiPM list here
        # no_corr = Table(col_dict={"xtalk_correction": Array(np.zeros((len(raw_data), corr_size)))})

    msg = f"Finished building XTK in {time.time() - start:.2f} seconds"
    log.info(msg)
