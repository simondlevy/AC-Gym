#!/usr/bin/env python3
'''
Script for plotting results of DRL run

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', metavar='CSVFILE', help='input .csv file')
    args = parser.parse_args()

    try:
        data = np.genfromtxt(args.csvfile, delimiter=',', skip_header=1)
    except Exception:
        print('Unable to open file %s' % args.csvfile)
        exit(1)

    if len(data.shape) < 2:
        print('Just one point: Mean = %+3.3f  Std = %+3.3f  Max = %+3.3f' %
              (data[1], data[2], data[3]))
        exit(0)

    ep = data[:, 0]
    # t = data[:, 1]
    r = data[:, 2]

    plt.plot(ep, r)

    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.title(args.csvfile)

    plt.show()


main()
