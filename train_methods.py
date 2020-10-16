import argparse
import json
import os
import pandas as pd
import time
import torch
import numpy as np

from bhmtorch_cpu import BHM3D_PYTORCH, BHM_REGRESSION_PYTORCH, BHM_VELOCITY_PYTORCH

def save_mdl(args, model, path):
    """
    @param model: BHM Module to save
    @param path (str): path relative to the mdl folder to save to
    """
    mdl_type = type(model).__name__
    print(" mdl_type:", mdl_type)
    if mdl_type == 'BHM3D_PYTORCH':
        ###===###
        print(" Saving in ./mdls/occupancy/...")
        if not os.path.isdir('./mdls/occupancy'):
            os.makedirs('./mdls/occupancy')

        torch.save({
            'mu': model.mu,
            'sig': model.sig,
            'grid': model.grid,
            'epsilon': model.epsilon,
            }, './mdls/occupancy/{}'.format(path)
        )
    elif mdl_type == 'BHM_REGRESSION_PYTORCH':
        ###===###
        print(" Saving in ./mdls/regression/...")
        if not os.path.isdir("./mdls/regression/"):
            os.makedirs('./mdls/regression/')

        torch.save({
            'mu': model.mu,
            'sig': model.sig,
            'grid': model.grid,
            'alpha': model.alpha,
            'beta': model.beta,
            }, './mdls/regression/{}'.format(path)
        )
    elif mdl_type == "BHM_VELOCITY_PYTORCH": ###===###
        print(" Saving in ./mdls/velocity/...")
        if not os.path.isdir("./mdls/velocity/"):
            os.makedirs('./mdls/velocity/')

        if args.likelihood_type == "gamma":
            torch.save({
                'grid': model.grid,
                "w_hatx":model.w_hatx,
                "w_haty":model.w_haty,
                "w_hatz":model.w_hatz,
                "likelihood_type":model.likelihood_type,
                }, "./mdls/velocity/{}".format(path)
            ) ###///###
        elif args.likelihood_type == "gaussian":
            torch.save({
                'mu_x': model.mu_x,
                'sig_x': model.sig_x,
                'mu_y': model.mu_y,
                'sig_y': model.sig_y,
                'mu_z': model.mu_z,
                'sig_z': model.sig_z,
                'grid': model.grid,
                'alpha': model.alpha,
                'beta': model.beta,
                "likelihood_type":model.likelihood_type,
                }, "./mdls/velocity/{}".format(path)
            )
        else:
            raise ValueError("Unsupported likelihood type: \"{}\"".format(args.likelihood_type))
    else:
        raise ValueError("Unknown model type: \"{}\"".format(mdl_type))

def train_occupancy(args, partitions, cell_resolution, X, y, sigma, framei):
    """
    @params: partitions (array of tuple of ints)
    @params: cell_resolution (tuple of 3 ints)
    @params: X (float32 tensor)
    @params: y (float32 tensor)
    @params: sigma (float32 tensor)
    @params: framei (int)

    @returns: []

    Runs and plots occupancy BHM for a single time frame (framei) given input
    parameters
    """
    totalTime = 0
    num_segments = len(partitions)
    for i, segi in enumerate(partitions):
        print(' Training on segment {} of {}...'.format(i+1, num_segments))
        bhm_mdl = BHM3D_PYTORCH(
            gamma=args.gamma,
            grid=None,
            cell_resolution=cell_resolution,
            cell_max_min=segi,
            X=X,
            nIter=1,
            sigma=sigma,
            kernel_type=args.kernel_type
        )
        t1 = time.time()
        bhm_mdl.fit(X, y)
        totalTime += (time.time()-t1)
        save_mdl(args, bhm_mdl, '{}_f{}_p{}'.format(args.save_model_path, framei, i))
        del bhm_mdl
    print(' Total training time={} s'.format(round(totalTime, 2)))
    del sigma

def train_regression(args, alpha, beta, cell_resolution, cell_max_min, X, y_occupancy, g, sigma, framei):
    """
    @params: alpha (float)
    @params: beta (float)
    @params: cell_resolution (tuple of 3 ints)
    @params: cell_max_min (tuple of 6 ints) - bounding area observed
    @params: X (float32 tensor)
    @params: y_occupancy (float32 tensor)
    @params: g (float32 tensor)
    @params: sigma (float32 tensor)
    @params: framei (int)

    @returns: []

    Runs and plots regression for a single time frame (framei) given input
    parameters
    """
    totalTime = 0
    # filter X,y such that only give the X's where y is 1
    filtered = g[(y_occupancy[:,0]==1),:]
    sigma = sigma[(y_occupancy[:,0]==1),:]
    bhm_regression_mdl = BHM_REGRESSION_PYTORCH(
        gamma=args.gamma,
        alpha=alpha,
        beta=beta,
        grid=None,
        cell_resolution=cell_resolution,
        cell_max_min=cell_max_min,
        X=X,
        nIter=1,
        sigma=sigma,
        kernel_type=args.kernel_type
    )
    time1 = time.time()
    bhm_regression_mdl.fit(filtered[:,:2], filtered[:,2])
    print(' Total training time={} s'.format(round(time.time() - time1, 2)))
    save_mdl(args, bhm_regression_mdl, '{}_f{}'.format(args.save_model_path, framei))
    del bhm_regression_mdl, filtered, sigma

def train_velocity(args, alpha, beta, X, y_vx, y_vy, y_vz, partitions, cell_resolution, cell_max_min, framei):
    totalTime = 0
    # filter X,y such that only give the X's where y is 1

    if args.likelihood_type == "gamma":
        bhm_velocity_mdl = BHM_VELOCITY_PYTORCH(
            gamma=args.gamma,
            grid=None,
            cell_resolution=cell_resolution,
            cell_max_min=cell_max_min,
            X=X,
            nIter=1,
            kernel_type=args.kernel_type,
            likelihood_type=args.likelihood_type
        )
    elif args.likelihood_type == "gaussian":
        bhm_velocity_mdl = BHM_VELOCITY_PYTORCH(
            gamma=args.gamma,
            alpha=alpha,
            beta=beta,
            grid=None,
            cell_resolution=cell_resolution,
            cell_max_min=cell_max_min,
            X=X,
            nIter=1,
            kernel_type=args.kernel_type,
            likelihood_type=args.likelihood_type
        )
    else:
        raise ValueError(" Unsupported likelihood type: \"{}\"".format(args.likelihood_type))

    time1 = time.time()
    bhm_velocity_mdl.fit(X, y_vx, y_vy, y_vz, eps=0) # , y_vy, y_vz
    print(' Total training time={} s'.format(round(time.time() - time1, 2)))
    save_mdl(args, bhm_velocity_mdl, '{}_f{}'.format(args.save_model_path, framei))
    del bhm_velocity_mdl