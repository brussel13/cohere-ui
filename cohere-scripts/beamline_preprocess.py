#!/usr/bin/env python

# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script reads raw data, applies correction related to instrument, and saves prepared data.
This script is written for a specific APS beamline. It reads multiple raw data files in each scan directory, applies
darkfield and whitefield correction if applicable, creates 3D stack for each scan, then alignes and combines with
other scans.
"""

__author__ = "Barbara Frosik"
__docformat__ = 'restructuredtext en'
__all__ = ['handle_prep',
           'main']

import argparse
import os
import re
import sys
import importlib
import convertconfig as conv
import cohere_core as cohere
import cohere_core.utilities as ut
import numpy as np
import cupy as cp
import shutil
from multipeak import MultPeakPreparer
from prep_helper import SepPreparer, SinglePreparer


def prep_data(prep_obj, **kwargs):
    """
    Creates prep_data.tif file in <experiment_dir>/preprocessed_data directory or multiple prep_data.tif in <experiment_dir>/<scan_<scan_no>>/preprocessed_data directories.
    Parameters
    ----------
    none
    Returns
    -------
    nothingcreated mp
    """
    if hasattr(prep_obj, 'multipeak') and prep_obj.multipeak:
        preparer = MultPeakPreparer(prep_obj)
    elif prep_obj.separate_scan_ranges or prep_obj.separate_scans:
        preparer = SepPreparer(prep_obj)
    else:
        preparer = SinglePreparer(prep_obj)

    batches = preparer.get_batches()
    if len(batches) == 0:
        return 'no scans to process'
    preparer.prepare(batches)

    return ''


def get_correlation_err(data_dir, refarr):
    """
    author: Paul Frosik
    It is assumed that the reference array and the array in data_dir are scans of the same experiment
    sample. This function aligns the two arrays and finds a correlation error between them.
    The error finding method is based on "Invariant error metrics for image reconstruction"
    by J. R. Fienup.

    :param data_dir: str
    :param refarr: ndarray
    :return: float
    """
    i = 0
    err = 0
    fft_refarr = cp.fft.fftn(refarr, norm='forward')
    for scan_dir in os.listdir(data_dir):
        if scan_dir.startswith('scan'):
            subdir = data_dir + '/' + scan_dir
            datafile = subdir + '/preprocessed_data/prep_data.tif'
            arr = ut.read_tif(datafile)
            # load on GPU
            arr = cp.array(arr)
            # align
            aligned = cp.abs(ut.shift_to_ref_array_cp(fft_refarr, arr))
            err = err + ut.pixel_shift_err(refarr, aligned)
            i = i + 1
    return err / i


def find_outlier_scans(experiment_dir, main_conf_map):
    """
    Author: Paul Frosik
    Added for auto-data. This function is called after experiment data has been read for
    separate_scans. Each scan is aligned with other scans and correlation error is calculated
    for each pair. The errors are summed for each scan. Summed errors are averaged, and standard
    deviation is found. The scans that summed error exceeds standard deviation are considered
    outliers, and they are not counted in the data set.

    :param experiment_dir: str
    :param main_conf_map: dict
    :return:
    """
    print('finding outlier scans now')
    err_dir = []
    outlier_scans = []
    for scan_dir in os.listdir(experiment_dir):
        if scan_dir.startswith('scan'):
            subdir = experiment_dir + '/' + scan_dir
            datafile = subdir + '/preprocessed_data/prep_data.tif'
            refarr = ut.read_tif(datafile)
            # load on GPU
            refarr = cp.array(refarr)
            err_dir.append((get_correlation_err(experiment_dir, refarr), scan_dir))
    err = [el[0].item() for el in err_dir]
    mean = np.average(err)
    stdev = np.std(err)
    for (err_value, dir) in err_dir:
        if err_value > (mean + stdev):
            scan_nr = re.findall(r'\d+', dir)
            outlier_scans.append(int(scan_nr[0]))
    print('outliers scans', outlier_scans)

    # read config_prep and add parameter outliers_scans with outliers
    # then change main config to set the separate scans to false
    prep_conf_file = experiment_dir + '/conf/config_prep'
    prep_conf_map = ut.read_config(prep_conf_file)
    prep_conf_map['outliers_scans'] = outlier_scans
    ut.write_config(prep_conf_map, prep_conf_file)

    main_conf_map['separate_scans'] = False
    ut.write_config(main_conf_map, experiment_dir + '/conf/config')

    # remove individual scan directories
    for scan_dir in os.listdir(experiment_dir):
        if scan_dir.startswith('scan'):
            shutil.rmtree(experiment_dir + '/' + scan_dir)


def handle_prep(experiment_dir, *args, **kwargs):
    print('preparing data')
    experiment_dir = experiment_dir.replace(os.sep, '/')
    # check configuration
    main_conf_file = experiment_dir + '/conf/config'
    main_conf_map = ut.read_config(main_conf_file)
    if main_conf_map is None:
        print('cannot read configuration file ' + main_conf_file)
        return 'cannot read configuration file ' + main_conf_file
    # convert configuration files if needed
    if 'converter_ver' not in main_conf_map or conv.get_version() is None or conv.get_version() > main_conf_map[
        'converter_ver']:
        conv.convert(experiment_dir + '/conf')
        # re-parse config
        main_conf_map = ut.read_config(main_conf_file)

   # main_conf_map = get_config_map(experiment_dir)

    er_msg = cohere.verify('config', main_conf_map)
    if len(er_msg) > 0:
        # the error message is printed in verifier
        return er_msg

    auto_data = 'auto_data' in main_conf_map and main_conf_map['auto_data'] == True
    # check main config if doing auto
    # if not auto run handle_prep_case
    if not auto_data:
        return handle_prep_case(experiment_dir, main_conf_file, args, kwargs)

    from multiprocessing import Process

    # if auto, choose option to set separate scans in main config and use ut.write_config to save
    # run handle_prep to create scans subdirectories
    main_conf_map['separate_scans'] = True
    ut.write_config(main_conf_map, main_conf_file)
    handle_prep_case(experiment_dir, main_conf_file, args, kwargs)

    # find outliers scans by finding correlation error between each two scans after aligning them
    # and finding scans with biggest summed error
    p = Process(target=find_outlier_scans, args=(experiment_dir, main_conf_map))
    p.start()
    p.join()

    # run handle_prep_case again
    return handle_prep_case(experiment_dir, main_conf_file, args, kwargs)


def handle_prep_case(experiment_dir, main_conf_file, *args, **kwargs):
    """
    Reads the configuration files and accrdingly creates prep_data.tif file in <experiment_dir>/prep directory or multiple
    prep_data.tif in <experiment_dir>/<scan_<scan_no>>/prep directories.
    Parameters
    ----------
    experimnent_dir : str
        directory with experiment files
    Returns
    -------
    experimnent_dir : str
        directory with experiment files
    """
    main_conf_map = ut.read_config(main_conf_file)
    if 'beamline' in main_conf_map:
        beamline = main_conf_map['beamline']
        try:
            beam_prep = importlib.import_module('beamlines.' + beamline + '.prep')
        except Exception as e:
            print(e)
            print('cannot import beamlines.' + beamline + '.prep module.')
            return 'cannot import beamlines.' + beamline + '.prep module.'
    else:
        print('Beamline must be configured in configuration file ' + main_conf_file)
        return 'Beamline must be configured in configuration file ' + main_conf_file

    prep_conf_file = experiment_dir + '/conf/config_prep'
    prep_conf_map = ut.read_config(prep_conf_file)
    if prep_conf_map is None:
        return None
    er_msg = cohere.verify('config_prep', prep_conf_map)
    if len(er_msg) > 0:
        # the error message is printed in verifier
        return er_msg

    data_dir = prep_conf_map['data_dir'].replace(os.sep, '/')
    if not os.path.isdir(data_dir):
        print('data directory ' + data_dir + ' is not a valid directory')
        return 'data directory ' + data_dir + ' is not a valid directory'

    instr_config_map = ut.read_config(experiment_dir + '/conf/config_instr')
    # create BeamPrepData object defined for the configured beamline
    conf_map = main_conf_map
    conf_map.update(prep_conf_map)
    conf_map.update(instr_config_map)
    if 'multipeak' in main_conf_map and main_conf_map['multipeak']:
        conf_map.update(ut.read_config(experiment_dir + '/conf/config_mp'))
    prep_obj = beam_prep.BeamPrepData()
    msg = prep_obj.initialize(experiment_dir, conf_map)
    if len(msg) > 0:
        print(msg)
        return msg

    msg = prep_data(prep_obj)
    if len(msg) > 0:
        print(msg)
        return msg

    print('finished beamline preprocessing')
    return ''


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="directory where the configuration files are located")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    handle_prep(experiment_dir)


if __name__ == "__main__":
    exit(main(sys.argv[1:]))
