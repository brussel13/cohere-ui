import sys
import socket
import argparse
import ast
import util.util as ut

def run_with_mpi(mem_size, devs):
    dev = devs[socket.gethostname()]
    mem = ut.get_gpu_load(mem_size, dev)
    print(socket.gethostname(), mem)


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("devs", help="devs")
    parser.add_argument("mem_size", help="mem_size")

    args = parser.parse_args()
    devs = ast.literal_eval(args.devs)
    run_with_mpi(args.mem_size, devs)


if __name__ == "__main__":
    print('args', sys.argv)
    exit(main(sys.argv[1:]))
