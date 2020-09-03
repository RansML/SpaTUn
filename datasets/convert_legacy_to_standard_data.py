import argparse
import os
import pandas as pd
import numpy as np

# ==============================================================================
# Conversions
# ==============================================================================

def convert_from_legacy(header_names=None, framei=0):
    """
    Converts a dataset from (x,y,z,occupancy) ->
        (t,x,y,z,occupancy,sigx,sigy,sigz)

    Accepts a header argument to be the schema of the old dataset
    framei describes which timestep the dataset corresponds to
    """
    if not header_names:
        header_names = ['X', 'Y', 'Z', 'occupancy']
    df = pd.read_csv(args.path, sep=',', names=header_names)
    n = len(df.index)
    df.insert(0, "t", framei*np.ones(n),True)
    df['sig_x'] = np.zeros(n)
    df['sig_y'] = np.zeros(n)
    df['sig_z'] = np.zeros(n)
    df.to_csv(args.new_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, default='', help='Path to dataset')
    parser.add_argument('--new_path', type=str, default='', help='Path to save')
    args = parser.parse_args()
    if args.path == '' or args.new_path == '':
        print('Must provide a dataset path and new path under --path and --new_path flags')
    convert_from_legacy()
    print('Successfully converted dataset at path %s from legacy to current saved at %s' % (args.path, args.new_path))
