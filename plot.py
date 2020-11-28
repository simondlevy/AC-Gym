#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='FILENAME', help='input file')
    args = parser.parse_args()

    plt.plot(np.load(args.filename))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(args.filename)
    plt.show()

if __name__ == '__main__':
    main()





