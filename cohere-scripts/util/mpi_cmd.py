import sys
import socket
import argparse
import ast
import datetime
import util as ut

def write_log(rank: int, msg: str) -> None:
    """
    Use this to force writes for debugging. PBS sometimes doesn't flush
    std* outputs. MPI faults clobber greedy flushing of default python
    logs.
    """
    with open(f'{rank}.log', 'a') as log_f:
        log_f.write(f'{datetime.datetime.now()} | {msg}\n')


def run_with_mpi(mem_size, devs):
    host = socket.gethostname()
    dev = devs[host]
    mem = ut.get_gpu_load(float(mem_size), dev)
    print(socket.gethostname(), mem)
    return mem

def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("mem_size", help="mem_size")
    parser.add_argument("devs", help="devs")

    args = parser.parse_args()
    write_log(0,'arg '+ args.devs)
    devs = ast.literal_eval(args.devs)
    return run_with_mpi(args.mem_size, devs)


if __name__ == "__main__":
    exit(main(sys.argv[1:]))
