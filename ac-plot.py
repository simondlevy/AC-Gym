#!/usr/bin/env python3
'''
Script for plotting results of DRL run

Copyright (C) 2021 Simon D. Levy

MIT License
'''

import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot(data, beg, s1, s2, lbl):

    if len(data.shape) < 2:
        print('Just one point: Mean = %+3.3f  Std = %+3.3f  Max = %+3.3f' %
              (data[1], data[2], data[3]))
        exit(0)

    g = data[:, 0]

    mn = data[:, beg]
    sd = data[:, beg+1]
    mx = data[:, beg+2]

    plt.plot(g, mn, s1)
    plt.plot(g, mx, s2)
    plt.plot(g, mn + sd, 'g--')
    plt.plot(g, mn - sd, 'g--')

    plt.xlabel('Generation')
    plt.legend(['Mean ' + lbl, 'Max ' + lbl, '+/-1 StdDev'])


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', metavar='CSVFILE', help='input .csv file')
    args = parser.parse_args()

    try:
        data = np.genfromtxt(args.csvfile, delimiter=',', skip_header=1)
    except Exception:
        print('Unable to open file %s' % args.csvfile)
        exit(1)

    exit(0)

    plot(data, 1, 'r', 'm', 'Fitness')

    plt.title(args.csvfile)

    plt.show()


main()
