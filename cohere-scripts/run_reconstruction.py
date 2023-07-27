# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script manages reconstruction(s).
Depending on configuration it starts either single reconstruction, GA, or multiple reconstructions. In multiple reconstruction scenario or split scans the script runs concurrent reconstructions.
"""

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['rec_process',
           'get_gpu_use',
           'manage_reconstruction',
           'main']

import sys
import signal
import os
import argparse
from multiprocessing import Process, Queue
import cohere_core as cohere
import util.util as ut
import convertconfig as conv
from functools import reduce
import subprocess
import ast


class Memory_broker:
    MEM_FACTOR = 170
    GA_MEM_FACTOR = 250
    GA_FAST_MEM_FACTOR = 184

    def __init__(self, dev):
        self.dev = dev
        if type(self.dev) == dict:
            self.cluster = True # a cluster with multiple hosts
        else:
            self.cluster = False


    def get_rec_mem(self, data_shape, ga_method, pc_in_use):
        # find size of data array
        data_size = reduce((lambda x, y: x * y), data_shape) / 1000000.
        mem_factor = Memory_broker.MEM_FACTOR
        c = 0
        if ga_method is not None:
            if ga_method == 'fast':
                mem_factor = Memory_broker.GA_FAST_MEM_FACTOR
                c = 430
            else:
                mem_factor = Memory_broker.GA_MEM_FACTOR
        rec_mem_size = data_size * mem_factor + c
        if pc_in_use:
            rec_mem_size = rec_mem_size * 2

        return rec_mem_size


    def get_mem_map(self, rec_mem_size):
        if self.cluster: # a cluster with multiple hosts
            hosts = ','.join(self.dev.keys())
            script = 'util.mpi_cmd.py'
            command = ['mpiexec', '-n', str(len(self.dev)), '--host', hosts, 'python', script, str(rec_mem_size), self.dev]
            result = subprocess.run(command, stdout=subprocess.PIPE)
            mem = result.stdout.decode("utf-8").strip()
            mem_map = {}
            for line in mem.splitlines():
                entry = line.split(" ", 1)
                # print('entry', entry)
                mem_map[entry[0]] = ast.literal_eval(entry[1])
            # memory map contains dict with hosts keys, and value a dict of
            # gpu Id/available runs
            print(mem_map)
        else:
            mem_map = ut.get_gpu_load(rec_mem_size, self.dev)

        self.mem_map = mem_map


    def get_balanced_distr(self, map, no):
        total_avail = reduce((lambda x, y: x + y), map.values())
        if total_avail <= no:
            return map, total_avail
        else:
            factor = no * 1.0 / total_avail
            allocated_total = 0
            allocated = {}
            for key, value in map.items():
                allocated[key] = int(factor * value)
                allocated_total += allocated[key]
            # need allocate more to account for the fraction
            need_allocate = no - allocated_total
            for key in map.keys():
                if need_allocate == 0:
                    break
                if allocated[key] < map[key]:
                    allocated[key] += 1
                    need_allocate -= 1
            return allocated, no


    def balance_devs(self, no_rec, no_scan_ranges):
        no_all_recs = no_rec * no_scan_ranges
        if self.cluster:
            host_avail = {}
            for host in self.mem_map.keys():
                host_avail[host] = reduce((lambda x, y: x + y), self.mem_map[host].values())
            # find balanced distribution among hosts
            host_distr, no_total_avail = self.get_balanced_distr(host_avail, no_all_recs)
            if no_total_avail <= no_all_recs:
                allocated = self.mem_map
            host_allocated = {}
            for host, map in host_distr.items():


        available_recs = self.get_avail_recs()
        if available_recs > no_rec * no_scan_ranges:
            # distribute devices evenly in one list or list of lists for multiple scans
            pass
        else:
            # not enough resources, first serialize the scans reconstructions then
            # limit number of reconstructions in one scan
            if no_scan_ranges == 1:
                # return all devices in one list
                pass
            else:
                # return list of lists
                aval_scans = available_recs // no_rec
                # distribute into aval_scans number of buckets with no_rec

    def get_assigned_devs(self, no_rec, no_scan_ranges, data_shape, ga_method, pc_in_use):
        # if there is only one reconstruction find a device in a simplest way
        if no_rec * no_scan_ranges == 1:
            if type(self.dev) == list:
                assigned = self.dev[0]
            elif self.dev == 'all' or self.cluster:
                # dah, have to find a gpu and not allow it on darwin
                # assuming here that if cluster configuration, it includes the  local host
                if sys.platform == 'darwin':
                    print('currently not supporting "all" configuration on Darwin')
                    return [-1]
                else:
                    assigned = ut.get_best_gpu()
            return assigned

        if sys.platform == 'darwin':
            if self.cluster:
                print('currently not supporting cluster configuration on Darwin')
                return [-1]
            # the gpu library is not working on OSX, so run one reconstruction on each GPU
            try:
                self.mem_map = {dev: [1] for dev in self.dev}
            except:
                print('on Darwin platform the available devices must be entered as list')
                return [-1]
        else:
            rec_mem_size = self.get_rec_mem(self, data_shape, ga_method, pc_in_use)
            self.get_mem_map(self, rec_mem_size)
            assigned = self.balance_devs(no_rec, no_scan_ranges)


def rec_process(proc, conf_file, datafile, dir, gpus, r, q):
    """
    Calls the reconstruction function in a module identified by parameter. After the reconstruction is finished, it enqueues th eprocess id wit associated list of gpus.
    Parameters
    ----------
    proc : str
        processing library, chices are cpu, cuda, opencl
    conf_file : str
        configuration file with reconstruction parameters
    datafile : str
        name of file containing data
    dir : str
        parent directory to the <prefix>/results, or results directory
    gpus : list
       a list of gpus that will be used for reconstruction
    r : str
       a string indentifying the module to use for reconstruction
    q : Queue
       a queue that returns tuple of procees id and associated gpu list after the reconstruction process is done
    Returns
    -------
    nothing
    """
    if r == 'g':
        cohere.reconstruction_GA.reconstruction(proc, conf_file, datafile, dir, gpus)
    elif r == 'm':
        cohere.reconstruction_multi.reconstruction(proc, conf_file, datafile, dir, gpus)
    elif r == 's':
        cohere.reconstruction_single.reconstruction(proc, conf_file, datafile, dir, gpus)
    q.put((os.getpid(), gpus))


def get_gpu_use(devices, no_dir, no_rec, data_shape, pc_in_use, ga_method):
    """
    Determines the use case, available GPUs that match configured devices, and selects the optimal distribution of reconstructions on available devices.
    Parameters
    ----------
    devices : list
        list of configured GPU ids to use for reconstructions. If -1, operating system is assigning it
    no_dir : int
        number of directories to run independent reconstructions
    no_rec : int
        configured number of reconstructions to run in each directory
    data_shape : tuple
        shape of data array, needed to estimate how many reconstructions will fit into available memory
    pc_in_use : boolean
        True if partial coherence is configured
    Returns
    -------
    gpu_use : list
        a list of int indicating number of runs per consecuitive GPUs
    """
    from functools import reduce

    no_runs = no_dir * no_rec
    if sys.platform == 'darwin':
        # the gpu library is not working on OSX, so run one reconstruction on each GPU
        try:
            gpu_load = {dev : [1] for dev in devices}
        except:
            print('on Darwin platform the available devices must be entered as list')
            return [-1]
    else:
        # find size of data array
        data_size = reduce((lambda x, y: x * y), data_shape) / 1000000.
        mem_factor = MEM_FACTOR
        c = 0
        if ga_method is not None:
            if ga_method == 'fast':
                mem_factor = GA_FAST_MEM_FACTOR
                c = 430
            else:
                mem_factor = GA_MEM_FACTOR
        rec_mem_size = data_size * mem_factor + c
        if pc_in_use:
            rec_mem_size = rec_mem_size * 2
        if type(devices) == dict: # a cluster with multiple hosts
            hosts = ','.join(devices.keys())
            # memory map returns dict with hosts keys, and value a dict of
            # gpu Id/available runs
            mem_map = ut.run_with_mpi(hosts, len(devices), devices, rec_mem_size)
            # calculate number available runs on each host and distribute the no_runs
            # proportionally
            host_available = {}
            for host in mem_map.keys():
                host_available[host] = reduce((lambda x, y: x + y), mem_map[host].values())
            total_hosts_available = reduce((lambda x, y: x + y), host_available.values())
            if total_hosts_available > no_rec:
                factor = no_rec * 1.0 / total_hosts_available
            if factor < 1:
                accounted = 0
                host_allocated = {}
                for host in host_available:
                    host_allocated[host] = int(factor * host_available[host])
                    accounted += host_allocated[host]
                # need allocate more to account for the fraction
                need_allocate = no_rec - accounted
                for host in host_available:
                    if host_allocated[host] < host_available[host]:
                        host_allocated[host] += 1
                        need_allocate -= 1
                        if need_allocate == 0:
                            break
            else:
                host_allocated = host_available
            # get distribution between GPUs on each host
            gpu_use = {}
            for host in host_allocated:
                gpu_distribution = ut.get_gpu_distribution(no_runs, host_allocated[host])
                gpu_use[host] = sum([[k]*v for k,v in gpu_distribution.items()],[])
            return gpu_use
        else: # the processing runs on one machine
            gpu_load = ut.get_gpu_load(rec_mem_size, devices)
            gpu_distribution = ut.get_gpu_distribution(no_runs, gpu_load)
            print('distr', gpu_distribution)
            ids = list(gpu_distribution.keys())
            gpu_use = []
            while len(ids) > 0:
                to_remove = [id for id in ids if gpu_distribution[id] == 0]
                ids = [id for id in ids if id not in to_remove]
                gpu_use.extend(ids)
                for id in ids:
                    gpu_distribution[id] -=1

            if len(gpu_use) > no_runs:
                gpu_use = gpu_use[:no_runs]

            return gpu_use


def manage_reconstruction(experiment_dir, rec_id=None):
    """
    This function starts the interruption discovery process and continues the recontruction processing.
    It reads configuration file defined as <experiment_dir>/conf/config_rec.
    If multiple generations are configured, or separate scans are discovered, it will start concurrent reconstructions.
    It creates image.npy file for each successful reconstruction.
    Parameters
    ----------
    experiment_dir : str
        directory where the experiment files are loacted
    rec_id : str
        optional, if given, alternate configuration file will be used for reconstruction, (i.e. <rec_id>_config_rec)
    Returns
    -------
    nothing
    """
    def manage_scan_range(generations, rec_config_map, reconstructions, lib, conf_file, datafile, dir, device_use):
        if generations > 1:
            if 'ga_fast' in rec_config_map and rec_config_map['ga_fast']:
                cohere.mpi_cmd.run_with_mpi(lib, conf_file, datafile, dir, device_use)
            else:
                cohere.reconstruction_populous_GA.reconstruction(lib, conf_file, datafile, dir, device_use)
        elif reconstructions > 1:
            cohere.mpi_cmd.run_with_mpi(lib, conf_file, datafile, dir, device_use)
        else:
            cohere.reconstruction_single.reconstruction(lib, conf_file, datafile, dir, device_use)

    print('starting reconstruction')
    experiment_dir = experiment_dir.replace(os.sep, '/')
    # the rec_id is a postfix added to config_rec configuration file. If defined, use this configuration.
    conf_dir = experiment_dir + '/conf'
    # convert configuration files if needed
    main_conf = conf_dir + '/config'
    if os.path.isfile(main_conf):
        main_config_map = ut.read_config(main_conf)
        if main_config_map is None:
            print ("info: can't read " + main_conf + " configuration file")
            return None
    else:
        print("info: missing " + main_conf + " configuration file")
        return None

    if 'converter_ver' not in main_config_map or conv.get_version() is None or conv.get_version() < main_config_map['converter_ver']:
        main_config_map = conv.convert(conf_dir, 'config')
    # verify main config file
    er_msg = cohere.verify('config', main_config_map)
    if len(er_msg) > 0:
        # the error message is printed in verifier
        return None

    if rec_id is None:
        conf_file = conf_dir + '/config_rec'
    else:
        conf_file = conf_dir + '/config_rec_' + rec_id

    rec_config_map = ut.read_config(conf_file)
    if rec_config_map is None:
        return

    # verify configuration
    er_msg = cohere.verify('config_rec', rec_config_map)
    if len(er_msg) > 0:
        # the error message is printed in verifier
        return None

    # find which library to run it on, default is numpy ('np')
    if 'processing' in rec_config_map:
        proc = rec_config_map['processing']
    else:
        proc = 'auto'

    lib = 'np'
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
            print('cupy is not installed, select different library (proc)')
            return
    elif proc == 'torch':
        try:
            import torch
            lib = 'torch'
        except:
            print('pytorch is not installed, select different library (proc)')
            return
    elif proc == 'np':
        pass  # lib set to 'np'
    else:
        print('invalid "proc" value', proc, 'is not supported')
        return

    separate = False
    if 'separate_scans' in main_config_map and main_config_map['separate_scans']:
        separate = True
    if 'separate_scan_ranges' in main_config_map and main_config_map['separate_scan_ranges']:
        separate = True

    # for multipeak reconstruction divert here
    if 'multipeak' in main_config_map and main_config_map['multipeak']:
        config_map = ut.read_config(experiment_dir + "/conf/config_mp")
        config_map.update(main_config_map)
        config_map.update(rec_config_map)
        if 'device' in config_map:
            dev = config_map['device']
        else:
            dev = [-1]
        peak_dirs = []
        for dir in os.listdir(experiment_dir):
            if dir.startswith('mp'):
                peak_dirs.append(experiment_dir + '/' + dir)
        cohere.reconstruction_coupled.reconstruction(lib, config_map, peak_dirs, dev)
    else:
        # exp_dirs_data list hold pairs of data and directory, where the directory is the root of data/data.tif file, and
        # data is the data.tif file in this directory.
        exp_dirs_data = []
        # experiment may be multi-scan in which case reconstruction will run for each scan or scan range
        if separate:
            for dir in os.listdir(experiment_dir):
                if dir.startswith('scan'):
                    datafile = experiment_dir + '/' + dir + '/phasing_data/data.tif'
                    if os.path.isfile(datafile):
                        exp_dirs_data.append((datafile, experiment_dir + '/' + dir))
        else:
            # in typical scenario data_dir is not configured, and it is defaulted to <experiment_dir>/data
            # the data_dir is ignored in multi-scan scenario
            if 'data_dir' in rec_config_map:
                data_dir = rec_config_map['data_dir'].replace(os.sep, '/')
            else:
                data_dir = experiment_dir + '/phasing_data'
            datafile = data_dir + '/data.tif'
            if os.path.isfile(datafile):
                exp_dirs_data.append((datafile, experiment_dir))
        no_scan_ranges = len(exp_dirs_data)
        if no_scan_ranges == 0:
            print('did not find data.tif file(s). ')
            return
        if 'ga_generations' in rec_config_map:
            generations = rec_config_map['ga_generations']
        else:
            generations = 0
        if 'reconstructions' in rec_config_map:
            reconstructions = rec_config_map['reconstructions']
        else:
            reconstructions = 1
        device_use = []
        if lib == 'np':
            cpu_use = [-1] * reconstructions
            if no_scan_ranges > 1:
                for _ in range(no_scan_ranges):
                    device_use.append(cpu_use)
            else:
                device_use = cpu_use
        else:
            if 'device' in rec_config_map:
                devices = rec_config_map['device']
            else:
                devices = 'all'
            if no_scan_ranges * reconstructions > 1:
                data_shape = cohere.read_tif(exp_dirs_data[0][0]).shape
                if generations > 1:
                    if 'ga_fast' in rec_config_map and rec_config_map['ga_fast']:
                        ga_method = 'fast'
                    else:
                        ga_method = 'populous'
                else:
                    ga_method = None
                available_dev = ut.get_gpu_load()
                device_use = get_gpu_use(devices, no_scan_ranges, reconstructions, data_shape, 'pc' in rec_config_map['algorithm_sequence'], ga_method)
                # check if cluster configuration
                if type(device_use) == dict:
                    host_file = open(experiment_dir + 'hosts.txt', mode='w+')
                    temp_dev_list = []
                    for host, devices in device_use.items():
                        host_file.write(host + ':' + str(len(devices)))
                        temp_dev_list.extend(devices)
                    host_file.close()
                    device_use = temp_dev_list
            else:
                device_use = devices
        print('device use', device_use)

        if no_scan_ranges == 1:
            if len(device_use) == 0:
                device_use = [-1]
            dir_data = exp_dirs_data[0]
            datafile = dir_data[0]
            dir = dir_data[1]
            manage_scan_range(generations, rec_config_map, reconstructions, lib, conf_file, datafile, dir, device_use)
            # if generations > 1:
            #     if 'ga_fast' in rec_config_map and rec_config_map['ga_fast']:
            #         cohere.mpi_cmd.run_with_mpi(lib, conf_file, datafile, dir, device_use)
            #     else:
            #         cohere.reconstruction_populous_GA.reconstruction(lib, conf_file, datafile, dir, device_use)
            # elif reconstructions > 1:
            #     cohere.mpi_cmd.run_with_mpi(lib, conf_file, datafile, dir, device_use)
            # else:
            #     cohere.reconstruction_single.reconstruction(lib, conf_file, datafile, dir, device_use)
        else: # multiple scans or scan ranges
            if len(device_use) == 0:
                device_use = [[-1]]
            else:
                # check if is it worth to use last chunk
                if lib != 'np' and len(device_use[0]) > len(device_use[-1]) * 2:
                    device_use = device_use[0:-1]
            if generations > 1:
                r = 'g'
            elif reconstructions > 1:
                r = 'm'
            else:
                r = 's'
            q = Queue()
            for gpus in device_use:
                q.put((None, gpus))
            # index keeps track of the multiple directories
            index = 0
            processes = {}
            pr = []
            while index < no_scan_ranges:
                pid, gpus = q.get()
                if pid is not None:
                    os.kill(pid, signal.SIGKILL)
                    del processes[pid]
                datafile = exp_dirs_data[index][0]
                dir = exp_dirs_data[index][1]
                p = Process(target=rec_process, args=(lib, conf_file, datafile, dir, gpus, r, q))
                p.start()
                pr.append(p)
                processes[p.pid] = index
                index += 1

            for p in pr:
                p.join()

            # close the queue
            q.close()

        print('finished reconstruction')


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="experiment directory.")
    parser.add_argument("--rec_id", help="reconstruction id, a postfix to 'results_phasing_' directory")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir

    if args.rec_id:
        manage_reconstruction(experiment_dir, args.rec_id)
    else:
        manage_reconstruction(experiment_dir)


if __name__ == "__main__":
    main(sys.argv[1:])

