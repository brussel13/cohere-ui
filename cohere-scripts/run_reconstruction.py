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
import convertconfig as conv
from functools import reduce
import subprocess
import ast
import datetime

def write_log(rank: int, msg: str) -> None:
    """
    Use this to force writes for debugging. PBS sometimes doesn't flush
    std* outputs. MPI faults clobber greedy flushing of default python
    logs.
    """
    with open(f'{rank}.log', 'a') as log_f:
        log_f.write(f'{datetime.datetime.now()} | {msg}\n')




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
        print('self dev constr', self.dev)


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


    def get_best_gpu(self):
        import GPUtil

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        gpus = GPUtil.getGPUs()
        best_id = -1
        best_mem = 0

        for gpu in gpus:
            free_mem = gpu.memoryFree
            if free_mem > best_mem:
                best_mem = free_mem
                best_id = gpu.id

        return best_id


    def get_gpu_load(self, mem_size, ids):
        """
        This function is only used when running on Linux OS. The GPUtil module is not supported on Mac.
        This function finds available GPU memory in each GPU that id is included in ids list. It calculates
        how many reconstruction can fit in each GPU available memory.
        Parameters
        ----------
        mem_size : int
            array size
        ids : list
            list of GPU ids user configured for use
        Returns
        -------
        list
            list of available runs aligned with the GPU id list
        """
        import GPUtil

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        gpus = GPUtil.getGPUs()
        available = {}

        for gpu in gpus:
            if dev == 'all' or gpu.id in dev:
                free_mem = gpu.memoryFree
                avail_runs = int(free_mem / mem_size)
                if avail_runs > 0:
                    available[gpu.id] = avail_runs

        print(available)
        return available


    def get_mem_map(self, rec_mem_size):
        if self.cluster: # a cluster with multiple hosts
            hosts = ','.join(self.dev.keys())
            script = 'cohere-scripts/util/mpi_cmd.py'
            command = ['mpiexec', '-n', str(len(self.dev)), '--host', hosts, 'python', script, str(rec_mem_size), str(self.dev)]
            result = subprocess.run(command, stdout=subprocess.PIPE)
            mem = result.stdout.decode("utf-8").strip()
            mem_map = {}
            for line in mem.splitlines():
                entry = line.split(" ", 1)
                mem_map[entry[0]] = ast.literal_eval(entry[1])
            # memory map contains dict with hosts keys, and value a dict of
            # gpu Id/available runs
            print('hosts mem map after parsing', mem_map)
        else:
            print('self dev', self.dev)
            mem_map = self.get_gpu_load(rec_mem_size, self.dev)

        self.mem_map = mem_map


    def get_balanced_distr(self, map, no, no_batches):
        total_avail = reduce((lambda x, y: x + y), map.values())
        if total_avail <= no:
            if no_batches == 1:
                return map, total_avail
            else:
                no_requested = no - no % no_batches
        else:
            no_requested = no
        factor = no_requested * 1.0 / total_avail
        allocated_total = 0
        allocated = {}
        for key, value in map.items():
            allocated[key] = int(factor * value)
            allocated_total += allocated[key]
        # need allocate more to account for the fraction
        need_allocate = no_requested - allocated_total
        for key in map.keys():
            if need_allocate == 0:
                break
            if allocated[key] < map[key]:
                allocated[key] += 1
                need_allocate -= 1
        return allocated, no_requested


    def balance_devs(self, no_rec, no_scan_ranges):
        no_all_recs = no_rec * no_scan_ranges
        if self.cluster:
            host_avail = {}
            for host, host_dev_map in self.mem_map.items():
                print('host, dev map', host, host_dev_map)
                host_avail[host] = reduce((lambda x, y: x + y), host_dev_map.values())
            # find balanced distribution among hosts
            host_distr, no_total_avail = self.get_balanced_distr(host_avail, no_all_recs, no_scan_ranges)
            if no_total_avail <= no_all_recs:
                allocated = self.mem_map
                host_totals = host_avail
            else:
                allocated = {}
                host_totals = {}
                for host, gpu_map in host_distr.items():
                    balanced_gpu_map, host_total = self.get_balanced_distr(self.mem_map[host], host_distr[host], 1)
                    allocated[host] = balanced_gpu_map
                    host_totals[host] = host_total
            return allocated, host_totals
        else:
            return self.get_balanced_distr(self.mem_map, no_all_recs, no_scan_ranges)


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
                    assigned = self.get_best_gpu()
            # returns one single gpu id
            return assigned, 1

        if sys.platform == 'darwin':
            if self.cluster:
                print('currently not supporting cluster configuration on Darwin')
                return [-1], 0
            # the gpu library is not working on OSX, so run one reconstruction on each GPU
            try:
                self.mem_map = {dev: [1] for dev in self.dev}
            except:
                print('on Darwin platform the available devices must be entered as list')
                return [-1], 0

        rec_mem_size = self.get_rec_mem(data_shape, ga_method, pc_in_use)
        self.get_mem_map(rec_mem_size)
        print('mem map', self.mem_map)
        # assigned below is a map of available devices and total
        # in case of cluster it is a map of maps and a map of totals
        assigned = self.balance_devs(no_rec, no_scan_ranges)
        print('assigned',assigned)
        return assigned


def find_lib(proc):
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
            return None
    elif proc == 'torch':
        try:
            import torch
            lib = 'torch'
        except:
            print('pytorch is not installed, select different library (proc)')
            return None
    elif proc == 'np':
        pass  # lib set to 'np'
    else:
        print('invalid "proc" value', proc, 'is not supported')
        return None
    return lib


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
    import util.util as ut

    def manage_scan_range(generations, rec_config_map, reconstructions, lib, conf_file, datafile, dir, device_use, q=None):
        if generations > 1:
            if 'ga_fast' in rec_config_map and rec_config_map['ga_fast']:
                cohere.mpi_cmd.run_with_mpi(lib, conf_file, datafile, dir, device_use)
            else:
                cohere.reconstruction_populous_GA.reconstruction(lib, conf_file, datafile, dir, device_use)
        elif reconstructions > 1:
            cohere.mpi_cmd.run_with_mpi(lib, conf_file, datafile, dir, device_use)
        else:
            cohere.reconstruction_single.reconstruction(lib, conf_file, datafile, dir, device_use)

        if q is not None:
            q.put((os.getpid(), gpus))

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
    lib = find_lib(proc)
    if lib is None:
        return

    separate = False
    if 'separate_scans' in main_config_map and main_config_map['separate_scans']:
        separate = True
    if 'separate_scan_ranges' in main_config_map and main_config_map['separate_scan_ranges']:
        separate = True

    dev = [-1]
    if 'device' in rec_config_map:
        dev = rec_config_map['device']

    print('parsed dev',dev)

    # for multipeak reconstruction divert here
    if 'multipeak' in main_config_map and main_config_map['multipeak']:
        config_map = ut.read_config(experiment_dir + "/conf/config_mp")
        config_map.update(main_config_map)
        config_map.update(rec_config_map)
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

        ga_method = None
        if 'ga_generations' in rec_config_map:
            generations = rec_config_map['ga_generations']
            if 'ga_fast' in rec_config_map and rec_config_map['ga_fast']:
                ga_method = 'fast'
            else:
                ga_method = 'populous'
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
            print('dev creating', dev)
            mb = Memory_broker(dev)
            data_shape = cohere.read_tif(exp_dirs_data[0][0]).shape
            assigned, assig_no = mb.get_assigned_devs(reconstructions, no_scan_ranges, data_shape, ga_method, 'pc' in rec_config_map['algorithm_sequence'])
            temp_dev_list = []
            # check if cluster configuration
            if type(dev) == dict:
                host_file = open(experiment_dir + 'hosts.txt', mode='w+')
                for host, devices_map in assigned.items():
                    host_file.write(host + ':' + str(assig_no[host]))
                    temp_dev_list += sum([[dev_id] * no for dev_id, no in devices_map.items()], [])
                host_file.close()
            else:
                if assig_no == 0:
                    return
                elif assig_no == 1:
                    temp_dev_list = assigned
                else:
                    temp_dev_list = sum([[dev_id] * no for dev_id, no in assigned.items()], [])
        print('temp_dev_list', temp_dev_list)

        if no_scan_ranges == 1:
            device_use = temp_dev_list
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
            elif len(device_use) <= reconstructions * no_scan_ranges:
                device_use = [device_use]
                no_chunks = 1
            else:
                # divide the list into sub-lists of number of reconstructions length
                device_use = [device_use[x : x+reconstructions] for x in range(0, len(device_use), reconstructions)]

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
                # p = Process(target=rec_process, args=(lib, conf_file, datafile, dir, gpus, r, q))
                p = Process(target=manage_scan_range, args=(generations, rec_config_map, reconstructions, lib, conf_file, datafile, dir, device_use, q))
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
