from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd
import torch as pt
from random import sample
import sys

def make_csv(input_file_path, output_file_path):
    """
    @params: input_file_path
    @params: output_file_path
    Given input file path for a ply file, saves a csv file of inner, hit locations, and outer vertices and values at the output file path
    Also saves a normalized csv file of inner, hit locations, and outer vertices and values at output file path + "_normalized"
    """
    plydata = PlyData.read(input_file_path)

    #How much to perturb to get inner and outer data
    epsilon = 0.1

    #Get arrays of data in format (x, y, z, nx, ny, nz)
    sample_data = [list(x) for x in sample(list(plydata.elements[0].data), 1000)]

    #Create arrays of data for hit locations, points perturbed inwards, and points perturbed outwards
    hit_data = []
    in_data = []
    out_data = []
    for point in sample_data:
        hit_data.append([point[0], point[1], point[2]])
        in_data.append([point[0] - epsilon * point[3], point[1] - epsilon * point[4], point[2] - epsilon * point[5]])
        out_data.append([point[0] + epsilon * point[3], point[1] + epsilon * point[4], point[2] + epsilon * point[5]])

    hit_data = np.vstack(hit_data)
    in_data = np.vstack(in_data)
    out_data = np.vstack(out_data)

    #Hit locations have value 0, inner points have value -1, outer points have value 1
    hit_data = np.hstack((np.zeros((hit_data.shape[0],1)), hit_data, np.zeros((hit_data.shape[0],4))))
    in_data = np.hstack((np.zeros((in_data.shape[0],1)), in_data, -1+np.zeros((in_data.shape[0],1)), np.zeros((in_data.shape[0],3))))
    out_data = np.hstack((np.zeros((out_data.shape[0],1)), out_data, 1+np.zeros((out_data.shape[0],1)), np.zeros((out_data.shape[0],3))))

    data = np.vstack((hit_data, in_data, out_data))

    #Output csv is in format index,t,X,Y,Z,occupancy,sig_x,sig_y,sig_z
    df = pd.DataFrame(data,
                   columns=['t','X','Y','Z','occupancy','sig_x','sig_y','sig_z'])
    df.to_csv(output_file_path + '.csv')

    formatted_data = data

    # Normalize data
    for col in range(1, formatted_data.shape[1]):
        min_val = np.amin(formatted_data[:, col])
        max_val = np.amax(formatted_data[:, col])
        print("\nmin_val:", min_val)
        print("max_val:", max_val)

        if min_val != 0 or max_val != 0: # Leave columns with only 0s alone
            formatted_data[:, col] -= min_val
            formatted_data[:, col] /= (max_val - min_val)
            formatted_data[:, col] *= 2
            formatted_data[:, col] -= 1

    for col in range(1, formatted_data.shape[1]):
        min_val = np.amin(formatted_data[:, col])
        max_val = np.amax(formatted_data[:, col])
        print("\nmin_val:", min_val)
        print("max_val:", max_val)

    df_normalized = pd.DataFrame(formatted_data,
                   columns=['t','X','Y','Z','occupancy','sig_x','sig_y','sig_z'])
    df_normalized.to_csv(output_file_path + '_normalized.csv')


if __name__ == '__main__':
    if(len(sys.argv) != 3):
        print("Needs two arguments, input and output path")
    make_csv(sys.argv[1], sys.argv[2])
