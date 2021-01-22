import argparse
import json
import os
import pandas as pd
import time
import torch
import numpy as np

import train_methods
import query_methods
import utils_filereader
from bhmtorch_cpu import BHM3D_PYTORCH, BHM_REGRESSION_PYTORCH, BHM_VELOCITY_PYTORCH
from plot_methods import BHM_PLOTTER

"""
- features are the kernel distance thing to hinge points?
- output is a scalar value (ie: the velocity long X-Y plane, X-Z plane, or Z-Y plane)?
- How do you determine what the hinge points are?
    - query_dist in config file determines how frequently to space the hinge points?
    - How do you determine how big the total grid is on which to draw the hinge points?
- What are the partitions?

- What exactly is the model learning? How can information about whether or not a space is
occupied, etc. be gleaned from the current features?

- Try making regression grid 3d
    - will this break the plotting??

- train velocity
- query velocity
"""


def load_query_data(path):
    """
    @param path (str): path relative to query_data folder to save data
    """
    filename = './query_data/{}'.format(path)
    print(' Reading queried output from ' + filename)
    return torch.load(filename)

# ==============================================================================
# Train
# ==============================================================================
def train(fn_train, cell_max_min, cell_resolution, args):
    """
    @params: [fn_train, cell_max_min, cell_resolution]
    @returns: []
    Fits the 3D BHM on each frame of the dataset and plots occupancy or regression
    """
    print('\nTraining started---------------')
    alpha = 10**-2
    beta = 10**2
    for framei in range(args.num_frames):
        if args.model_type == "occupancy" or args.model_type == "regression":
            g, X, y_occupancy, sigma, partitions = utils_filereader.read_frame(args, framei, fn_train, cell_max_min)
        elif args.model_type == "velocity":
            X, y_vx, y_vy, y_vz, partitions = utils_filereader.read_frame_velocity(args, framei, fn_train, cell_max_min)
        else:
            raise ValueError("Unknown model type: \"{}\"".format(args.model_type))

        if args.model_type == 'occupancy':
            train_methods.train_occupancy(args, partitions, cell_resolution, X, y_occupancy, sigma, framei)
        elif args.model_type == 'regression':
            train_methods.train_regression(args, alpha, beta, cell_resolution, cell_max_min, X, y_occupancy, g, sigma[:,:2], framei)
            # For regression, we use sigma dimension 2. This is hard coded in the pass to plot_regression for the sigma term above
        elif args.model_type == "velocity": ###===###
            train_methods.train_velocity(args, alpha, beta, X, y_vx, y_vy, y_vz, partitions, cell_resolution, cell_max_min, framei) ###///###

        if args.model_type == "occupancy" or args.model_type == "regression":
            del g, X, y_occupancy, sigma, partitions
        elif args.model_type == "velocity":
            del X, y_vx, y_vy, y_vz, partitions
        else:
            raise ValueError("Unknown model type: \"{}\"".format(args.model_type))
    print('Training completed---------------\n')


# ==============================================================================
# Query
# ==============================================================================
def query(fn_train, cell_max_min, args):
    """
    @params: [fn_train, cell_max_min]
    @returns: []
    Queries the 3D BHM for occupancy or regression on each frame of the dataset
    """
    print('Querying started---------------')
    for framei in range(args.num_frames):
        if args.model_type == "occupancy" or args.model_type == "regression":
            g, X, y_occupancy, sigma, partitions = utils_filereader.read_frame(args, framei, fn_train, cell_max_min)
        elif args.model_type == "velocity":
            X, y_vx, y_vy, y_vz, partitions = utils_filereader.read_frame_velocity(args, framei, fn_train, cell_max_min)
        else:
            raise ValueError("Unknown model type: \"{}\"".format(args.model_type))

        if args.model_type == 'occupancy':
            query_methods.query_occupancy(args, partitions, X, y_occupancy, framei)
        elif args.model_type == 'regression':
            query_methods.query_regression(args, cell_max_min, X, y_occupancy, g, framei)
        elif args.model_type == "velocity": ###===###
            query_methods.query_velocity(args, X, y_vx, y_vy, y_vz, partitions, cell_resolution, cell_max_min, framei) ###///###

        if args.model_type == "occupancy" or args.model_type == "regression":
            del g, X, y_occupancy, sigma, partitions
        elif args.model_type == "velocity":
            del X, y_vx, y_vy, y_vz, partitions
        else:
            raise ValueError("Unknown model type: \"{}\"".format(args.model_type))
    print('Querying completed---------------\n')

# ==============================================================================
# Plot
# ==============================================================================
def plot(args):
    """
    @params: []
    @returns: []
    Plots data loaded from the args.save_query_data_path parameter
    """
    print('Plotting started---------------')
    plotter = BHM_PLOTTER(args, args.plot_title, args.surface_threshold, args.query_dist, occupancy_plot_type=args.occupancy_plot_type)
    for framei in range(args.num_frames):
        if args.model_type == 'occupancy':
            print("\nPlotting occupancy datapoints for frame %d ..." % framei)
            occupancyPlot, X, y, framei = load_query_data('occupancy/{}_f{}'.format(args.save_query_data_path, framei))
            plotter.plot_occupancy_frame(occupancyPlot, X, y, framei)
        elif args.model_type == 'regression':
            print("\nPlotting regression datapoints for frame %d ..." % framei)
            meanVarPlot, filtered, framei, cell_max_min = load_query_data('regression/{}_f{}'.format(args.save_query_data_path, framei))
            plotter.plot_regression_frame(meanVarPlot, filtered, framei, cell_max_min)
        elif args.model_type == "velocity": ###===###
            X, y_vx, y_vy, y_vz, Xq_mv, mean_x, mean_y, mean_z, framei = load_query_data('velocity/{}_f{}'.format(args.save_query_data_path, framei))
            # print("(plot) X.shape:", X.shape)
            # exit()
            plotter.plot_velocity_frame(X, y_vx, y_vy, y_vz, Xq_mv, mean_x, mean_y, mean_z, framei)
    print('Plotting completed---------------\n')


# ==============================================================================
# Interface with Spatun without parser
# ==============================================================================
class Spatun():
    def __init__(self, config_file, save_config, spatun_path="."):
        self.spatun_path = spatun_path
        self.args = argparse.Namespace(config=config_file, save_config=save_config)
        self.load_config()
    
    def save_config(self):
        with open(self.spatun_path + '/configs/' + self.args.save_config, 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)

    def load_config(self):
        config = json.load(open(self.spatun_path + '/configs/' + self.args.config, 'r'))
        defaults = json.load(open(self.spatun_path + '/configs/defaults', 'r'))

        for key in defaults:
            if key == 'save_config': continue
            if key in config and config[key]:
                self.args.__dict__[key] = config[key]
            else:
                self.args.__dict__[key] = defaults[key]
        
        if self.args.save_config:
            self.save_config()
        assert len(self.args.gamma) <= 3, 'Cannot support gamma with greater than dimension 3.'
        self.fn_train, self.cell_max_min, self.cell_resolution = utils_filereader.format_config(self.args)
    
    def update_config(self, key, val):
        self.args.__dict__[key] = val
        if self.args.save_config:
            self.save_config()
        self.fn_train, self.cell_max_min, self.cell_resolution = utils_filereader.format_config(self.args)
    
    def train(self, new_dataset=""):
        if new_dataset:
            self.update_config("dataset_path", new_dataset)
        train(self.fn_train, self.cell_max_min, self.cell_resolution, self.args)
        
    def query(self):
        query(self.fn_train, self.cell_max_min, self.args)
    
    def plot(self):
        plot(self.args)