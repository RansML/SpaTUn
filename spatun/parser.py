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
from spatun import load_query_data, train, query, plot

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


# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Settings Arguments
    parser.add_argument('--mode', type=str, help='tqp: Train Query and Plot, to: Train only, qo: Query only, po: Plot only')
    parser.add_argument('--num_frames', type=int, help='Number of data frames')
    parser.add_argument('--config', type=str, help='Path to the config to load relative to the config folder')
    parser.add_argument('--save_config', type=str, help='Saves the argparse config to path if set relative to the config folder')

    # Train Arguments
    parser.add_argument('--model_type', type=str, help='Model type (occupancy vs regression)')
    parser.add_argument('--likelihood_type', type=str, help='Likelihood type (Gamma, Gaussian)')
    parser.add_argument('--dataset_path', type=str, help='Path to dataset')
    parser.add_argument('--area_min', nargs=3, type=int, help='X Y Z minimum coordinates in bounding box (3 values)')
    parser.add_argument('--area_max', nargs=3, type=int, help='X Y Z maximum coordinates in bounding box (3 values)')
    parser.add_argument('--hinge_dist', nargs=3, type=int, help='X Y Z hinge point resolution (3 values)')
    parser.add_argument('--kernel_type', type=str, help='Type of RBF kernel: Vanilla RBF(), Convolution (conv), Wasserstein (wass)')
    parser.add_argument('--gamma', nargs='+', type=float, help='X Y Z Gamma (1-3 values)')
    parser.add_argument('--num_partitions', nargs=3, type=int, help='X Y Z number of partitions per axis (3 values)')
    parser.add_argument('--partition_bleed', type=float, help='Amount of bleed between partitions for plot stitching')
    parser.add_argument('--save_model_path', type=str, help='Path to save each model \
        (i.e. save_model_path is set to \"toy3_run0\", then the model at partition 1, frame 1 would save to \
        mdls/occupancy/toy3_run0_f1_p1)'
    )

    # Query Arguments
    parser.add_argument('--query_dist', nargs=3, type=float, help='X Y Z Q-resolution (3 values). If any value is\
        negative, a 4th value should be provided to slice the corresponding axis. If all negative, X_query=X_train.')
    parser.add_argument('--query_blocks', type=int, default=None, help='How many blocks to break the query method into')
    parser.add_argument('--variance_only', action="store_true", default=False, help='Only calculate the diagonal of the covariance matrix')
    parser.add_argument('--eval_path', type=str, help='Path of the evaluation dataset')
    parser.add_argument('--eval', type=int, help='1=evaluate metrics, 0, otherwise. Use data in --eval_path, if given.')
    parser.add_argument('--save_query_data_path', type=str, help='Path save each set of queried data \
        (i.e. save_model_path is set to \"toy3_run0\" and the model type is set to occupancy, \
        then the model at frame 1 would save to query_data/occupancy/toy3_run0_f1_p1)'
    )

    # Plot Arguments
    parser.add_argument('--occupancy_plot_type', type=str, help='Plot occupancy as scatter or volumetric plot')
    parser.add_argument('--plot_title', type=str, help='')
    parser.add_argument('--surface_threshold', nargs=2, type=float, help='Minimum threshold to show surface prediction on plot. Min or [Min, Max]')

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Set arguments according to the following Priority (High->Low):
    # 1:CL provided arguments, 2: config provided arguments, 3:default arguments
    if args.config:
        config = json.load(open('../configs/' + args.config, 'r'))
        defaults = json.load(open('../configs/defaults', 'r'))
        for key in vars(args):
            if key == 'save_config': continue
            if getattr(args, key): continue
            if key in config and config[key]:
                args.__dict__[key] = config[key]
            else:
                args.__dict__[key] = defaults[key]
    if args.save_config:
        with open('../configs/' + args.save_config, 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    assert len(args.gamma) <= 3, 'Cannot support gamma with greater than dimension 3.'

    fn_train, cell_max_min, cell_resolution = utils_filereader.format_config(args)
    if args.mode == 'tqp' or args.mode == 't':
        train(fn_train, cell_max_min, cell_resolution, args)
    if args.mode == 'tqp' or args.mode == 'q':
        query(fn_train, cell_max_min, args)
    if args.mode == 'tqp' or args.mode == 'p':
        plot(args)
    if args.mode == 'tq':
        train(fn_train, cell_max_min, cell_resolution, args)
        query(fn_train, cell_max_min, args)
    if args.mode == 'qp':
        query(fn_train, cell_max_min, args)
        plot(args)

    print("Mission complete!\n\n")
