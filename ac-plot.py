#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='FILENAME', help='.npy input file')
    args = parser.parse_args()
    
    a = np.load(args.filename)

    plt.plot(a[:,0], a[:,1])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(args.filename)
    plt.show()

if __name__ == '__main__':
    main()





