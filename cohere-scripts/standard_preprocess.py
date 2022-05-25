# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################


"""
This script formats data for reconstruction according to configuration.
"""

import sys
import argparse
import os
import numpy as np
import alien_tools as at
import convertconfig as conv
import cohere


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['prep',
           'format_data',
           'main']

def prep(fname, conf_info):
    """
    This function formats data for reconstruction. It uses configured parameters. The preparation consists of the following steps:
    1. removing the "aliens" - aliens are areas that are effect of interference. The area is manually set in a configuration file after inspecting the data. It could be also a mask file of the same dimensions that data.
    2. clearing the noise - the values below an amplitude threshold are set to zero
    3. amplitudes are set to sqrt
    4. cropping and padding. If the adjust_dimention is negative in any dimension, the array is cropped in this dimension.
    The cropping is followed by padding in the dimensions that have positive adjust dimension. After adjusting, the dimensions
    are adjusted further to find the smallest dimension that is supported by opencl library (multiplier of 2, 3, and 5).
    5. centering - finding the greatest amplitude and locating it at a center of new array. If shift center is defined, the center will be shifted accordingly.
    6. binning - adding amplitudes of several consecutive points. Binning can be done in any dimension.
    The modified data is then saved in data directory as data.tif.
    Parameters
    ----------
    fname : str
        tif file containing raw data
    conf_info : str
        experiment directory or configuration file. If it is directory, the "conf/config_data" will be
        appended to determine configuration file
    Returns
    -------
    nothing
    """
    
    # The data has been transposed when saved in tif format for the ImageJ to show the right orientation
    data = cohere.read_tif(fname)

    if os.path.isdir(conf_info):
        experiment_dir = conf_info
        conf = os.path.join(experiment_dir, 'conf', 'config_data')
        # if the experiment contains separate scan directories
        if not os.path.isfile(conf):
            base_dir = os.path.abspath(os.path.join(experiment_dir, os.pardir))
            conf = os.path.join(base_dir, 'conf', 'config_data')
    else:
        #assuming it's a file
        conf = conf_info
        experiment_dir = None

    config_map = cohere.read_config(conf)
    if config_map is None:
        return

    er_msg = cohere.verify('config_data', config_map)
    if len(er_msg) > 0:
        # the error message is printed in verifier
        return None

    if 'data_dir' in config_map:
        data_dir = config_map['data_dir']
    else:
        data_dir = 'phasing_data'
        if experiment_dir is not None:
            data_dir = os.path.join(experiment_dir, data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    try:
        data = at.remove_aliens(data, config_map, data_dir)
    except AttributeError:
        pass
    except Exception as e:
        print ('exiting, error in aliens removal ', str(e))
        return

    if 'intensity_threshold' in config_map:
        intensity_threshold = config_map['intensity_threshold']
    else:
        print ('define amplitude threshold. Exiting')
        return

    # zero out the noise
    prep_data = np.where(data <= intensity_threshold, 0, data)

    # square root data
    prep_data = np.sqrt(prep_data)

    if 'adjust_dimensions' in config_map:
        crops_pads = config_map['adjust_dimensions']
        # the adjust_dimension parameter list holds adjustment in each direction. Append 0s, if shorter
        if len(crops_pads) < 6:
            for _ in range (6 - len(crops_pads)):
                crops_pads.append(0)
    else:
        # the size still has to be adjusted to the opencl supported dimension
        crops_pads = (0, 0, 0, 0, 0, 0)
    # adjust the size, either pad with 0s or crop array
    pairs = []
    for i in range(int(len(crops_pads)/2)):
        pair = crops_pads[2*i:2*i+2]
        pairs.append(pair)

    prep_data = cohere.adjust_dimensions(prep_data, pairs)
    if prep_data is None:
        print('check "adjust_dimensions" configuration')
        return

    if 'center_shift' in config_map:
        center_shift = config_map['center_shift']
        prep_data = cohere.get_centered(prep_data, center_shift)
    else:
        prep_data = cohere.get_centered(prep_data, [0,0,0])

    if 'binning' in config_map:
        binsizes = config_map['binning']
        try:
            bins = []
            for binsize in binsizes:
                bins.append(binsize)
            filler = len(prep_data.shape) - len(bins)
            for _ in range(filler):
                bins.append(1)
            prep_data = cohere.binning(prep_data, bins)
        except:
            print ('check "binning" configuration')

    # save data
    data_file = os.path.join(data_dir, 'data.tif')
    cohere.save_tif(prep_data, data_file)
    print ('data ready for reconstruction, data dims:', prep_data.shape)
    
    
def format_data(experiment_dir):
    """
    For each prepared data in an experiment directory structure formats the data according to configured parameters and saves in the experiment space.

    Parameters
    ----------
    experiment_dir : str
        directory where the experiment processing files are saved

    Returns
    -------
    nothing
    """
    # convert configuration files if needed
    main_conf = os.path.join(experiment_dir, *("conf", "config"))
    if os.path.isfile(main_conf):
        config_map = cohere.read_config(main_conf)
        if config_map is None:
            print ("info: can't read " + main_conf + " configuration file")
            return None
    else:
        print("info: missing " + main_conf + " configuration file")
        return None

    if 'converter_ver' not in config_map or conv.get_version() is None or conv.get_version() < config_map['converter_ver']:
        conv.convert(os.path.join(experiment_dir, 'conf'))
        # re-parse config
        config_map = cohere.read_config(main_conf)

    er_msg = cohere.verify('config', config_map)
    if len(er_msg) > 0:
        # the error message is printed in verifier
        return None

    print ('formating data')
    prep_file = os.path.join(experiment_dir, 'preprocessed_data', 'prep_data.tif')
    if os.path.isfile(prep_file):
        prep(prep_file, experiment_dir)

    dirs = os.listdir(experiment_dir)
    for dir in dirs:
        if dir.startswith('scan'):
            scan_dir = os.path.join(experiment_dir, dir)
            prep_file = os.path.join(scan_dir, 'preprocessed_data', 'prep_data.tif')
            prep(prep_file, scan_dir)


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="experiment directory")
    args = parser.parse_args()
    format_data(args.experiment_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
