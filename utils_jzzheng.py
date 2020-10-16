#
# Jason Zheng (jzzheng)
# 10/23/19
#

import argparse
import numpy as np
import sys
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.graph_objs as go

AIRSIM_DATASET = 'datasets/airsim/depth_data.csv'  # this only contains one time frame
CARLA_DATASET = 'datasets/carla/carla_3d_town1.npz'

# ==============================================================================
# (LEGACY) Dataset Building Functions
# ==============================================================================
def translate_airsim_dataset(depth_csv):
    """
    @params: depth_csv (str)
    @returns: None

    Translates the .csv file given by depth_csv into csv frames
    The schema for the depth csv is
    old: (pixel_x, pixel_y, depth, x, y, z pitch, roll, yaw)
    new: (t, pixel_x, pixel_y, depth, x, y, z, pitch, roll, yaw) where there is
        one t value corresponding to each unique combination of x, y, z, p, r, y
    This is the x,y,z, location for the observer
    pixel_x and pixel_y represent the vector from the observer
    depth represents the distance to the nearest object from this vector
    pitch, roll, yaw represent ?
    """
    data = pd.read_csv(depth_csv, sep=',', header=None).values
    num_hits = len(data)  # since no LIDAR misses are reported, TODO: remove when 0's added in

    train_loader = {
        'poses': np.append(np.expand_dims(data[:,0], axis=-1),data[:,4:], axis=1),  # t, col 4, 5, 6, 7, 8, 9
        'x': data[:,:4],  # t, pixel_x, pixel_y, depth
        'y': np.ones(num_hits)  # 1 by default for hits
    }
    test_loader = {}
    return train_loader, test_loader

def unpack_dataset(npz_file):
    """
    @params: [npz_file (str)]
    @returns: None

    Unpacks .npz file given by npz_file into dataset
        The currently supported carla dataset has no testing data
    """
    with np.load(npz_file) as data:
        train_loader = {
            'poses': data['train_poses'],
            'x': data['X_train'],
            'y': data['Y_train']
        }
        test_loader = {
            'poses': data['test_poses'],
            'x': data['X_test'],
            'y': data['Y_test']
        }
    return train_loader, test_loader

def build_frame_csvs(loader):
    """
    @params: [loader (obj)]
    @returns: None
    Takes data from loader and compiles it into csv frames for training
    """
    fn_train = os.path.abspath('./datasets/carla/carla_3d_town1_frame')
    for framei, row in enumerate(loader['poses']):
        x_pos = row[0]
        y_pos = row[1]
        z_pos = row[2]
        theta_pos = row[3]
        frame_obs = []
        for obsj in range(len(loader['x'])):
            observation = loader['x'][obsj]
            if observation[0] != framei: continue
            object_presence = loader['y'][obsj]
            observation = np.append(observation, object_presence)
            frame_obs.append(observation)
        comp_df = pd.DataFrame(frame_obs).iloc[:,1:]
        comp_df.to_csv(fn_train+'{}.csv'.format(framei), header=False, index=False)

def build_frame_csvs_airsim(loader, save_path):
    """
    @params: [loader (obj)]
    @returns: None

    Takes data from loader and compiles it into csv frames for training
    TODO: CURRENTLY DOES NOT WORK WITH CARLA
    """
    fn_train = os.path.abspath('./datasets/{}'.format(save_path))
    unique_poses = np.unique(loader['poses'], axis=0)  # there should be one frame per unique pose
    for timestep_pose in unique_poses:
        t = int(timestep_pose[0])
        x_pos = timestep_pose[1]
        y_pos = timestep_pose[2]
        z_pos = timestep_pose[3]
        if len(timestep_pose) > 4:
            theta_pos = timestep_pose[4]
        frame_obs = []
        for obsj in range(len(loader['x'])):
            observation = loader['x'][obsj]
            if observation[0] != t: continue  # ensure timestep matches
            object_presence = loader['y'][obsj]
            observation = np.append(observation, object_presence)
            frame_obs.append(observation)
        if len(frame_obs) > 0:
            comp_df = pd.DataFrame(frame_obs).iloc[:,1:]
            comp_df.to_csv(fn_train+'{}.csv'.format(t), header=False, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='carla', help='Name of configuration case')

    args = parser.parse_args()
    if args.dataset == 'carla':
        train_loader, test_loader = unpack_dataset(CARLA_DATASET)
        build_frame_csvs(train_loader, 'carla/carla_3d_town1_frame')

    if args.dataset == 'airsim':
        train_loader, test_loader = translate_airsim_dataset(AIRSIM_DATASET)
        build_frame_csvs_airsim(train_loader, 'airsim/airsim_frame')
