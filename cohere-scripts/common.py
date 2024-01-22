import sys
import os
import cohere_core as cohere
import convertconfig as conv
import cohere_core.utilities as ut


def get_config_maps(experiment_dir, configs, debug=False, config_id=None):
    maps = {}
    # always get main config
    conf_dir = ut.join(experiment_dir, 'conf')
    main_conf = ut.join(conf_dir, 'config')
    if not os.path.isfile(main_conf):
        err_msg = f'info: missing {main_conf} configuration file'
        return err_msg, maps

    main_config_map = cohere.read_config(main_conf)
    # convert configuration files if different converter version
    if 'converter_ver' not in main_config_map or conv.get_version() is None or conv.get_version() < main_config_map[
        'converter_ver']:
        main_config_map = conv.convert(conf_dir, 'config')
    # verify main config file
    err_msg = cohere.verify('config', main_config_map)
    if len(err_msg) > 0:
        # the error message is printed in verifier
        if not debug:
            return err_msg, maps

    maps['config'] = main_config_map

    for conf in configs:
        # special case for rec_id
        if config_id is not None and (conf == 'config_rec' or conf == 'config_disp'):
            conf_file = ut.join(experiment_dir, 'conf', f'{conf}_{config_id}')
        else:
            conf_file = ut.join(experiment_dir, 'conf', conf)

        # special case for multipeak
        if conf == 'config_mp':
            if not ('multipeak' in main_config_map and main_config_map['multipeak']):
                continue

        if not os.path.isfile(conf_file):
            err_msg = f'info: missing {conf_file} configuration file'
            return err_msg, maps

        config_map = cohere.read_config(conf_file)

        # verify configuration
        err_msg = cohere.verify(conf, config_map)
        if len(err_msg) > 0:
            # the error message is printed in verifier
            if not debug:
                return err_msg, maps

        maps[conf] = config_map

    return err_msg, maps


def get_lib(proc):
    lib = 'np'
    err_msg = ''

    # find which library to run it on, default is numpy ('np')
    if sys.platform == 'darwin':
        return err_msg, lib

    if proc == 'auto':
        try:
            import cupy
            lib = 'cp'
        except:
            try:
                import torch
                lib = 'torch'
            except:
                pass
    elif proc == 'cp':
        try:
            import cupy
            lib = 'cp'
        except:
            err_msg = 'cupy is not installed, select different library (proc)'
    elif proc == 'torch':
        try:
            import torch
            lib = 'torch'
        except:
            err_msg = 'pytorch is not installed, select different library (proc)'
    elif proc == 'np':
        pass  # lib set to 'np'
    else:
        err_msg = f'invalid "proc" value, {proc} is not supported'

    return err_msg, lib