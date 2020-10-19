import argparse
import json
import os
import pandas as pd
import time
import torch
import numpy as np

from bhmtorch_cpu import BHM3D_PYTORCH, BHM_REGRESSION_PYTORCH, BHM_VELOCITY_PYTORCH


def load_mdl(args, path, type):
    """
    @param path (str): path relative to the mdl folder to load from
    @param type (str): type of model to load ('BHM3D_PYTORCH' / 'BHM_REGRESSION_PYTORCH')
    @returns: model: BHM Module that is loaded
    """
    filename = './mdls/{}'.format(path)
    print(' Loading the trained model from ' + filename)
    model_params = torch.load(filename)
    if type == 'BHM3D_PYTORCH':
        model = BHM3D_PYTORCH(
            gamma=args.gamma,
            grid=model_params['grid'],
            kernel_type=args.kernel_type
        )
        model.updateEpsilon(model_params['epsilon'])
        model.updateMuSig(model_params['mu'], model_params['sig'])
    elif type == 'BHM_REGRESSION_PYTORCH':
        model = BHM_REGRESSION_PYTORCH(
            alpha=model_params['alpha'],
            beta=model_params['beta'],
            gamma=args.gamma,
            grid=model_params['grid'],
            kernel_type=args.kernel_type
        )
        model.updateMuSig(model_params['mu'], model_params['sig'])

    elif type == 'BHM_VELOCITY_PYTORCH':
        #print(" Loading velocity model")
        if args.likelihood_type == "gamma":
            model = BHM_VELOCITY_PYTORCH(
                gamma=args.gamma,
                grid=model_params['grid'],
                w_hatx=model_params["w_hatx"],
                w_haty=model_params["w_haty"],
                w_hatz=model_params["w_hatz"],
                kernel_type=args.kernel_type,
                likelihood_type=model_params["likelihood_type"]
            )
        elif args.likelihood_type == "gaussian":
            model = BHM_VELOCITY_PYTORCH(
                alpha=model_params['alpha'],
                beta=model_params['beta'],
                gamma=args.gamma,
                grid=model_params['grid'],
                kernel_type=args.kernel_type,
                likelihood_type=model_params["likelihood_type"]
            )
            model.updateMuSig(model_params['mu_x'], model_params['sig_x'],
                              model_params['mu_y'], model_params['sig_y'],
                              model_params['mu_z'], model_params['sig_z'])
        else:
            raise ValueError("Unsupported likelihood type: \"{}\"".format(args.likelihood_type))

    else:
        raise ValueError("Unknown model type: \"{}\"".format(mdl_type))
    # model.updateMuSig(model_params['mu'], model_params['sig'])
    return model


def save_query_data(data, path):
    """
    @param data (tuple of elements): datapoints from regression/occupancy query to save
    @param path (str): path relative to query_data folder to save data
    """

    ###===###
    complete_dir = './query_data/{}'.format(path).split("/")
    # print("complete_dir:", complete_dir)
    complete_dir = "/".join(complete_dir[:-1])
    # print("complete_dir:", complete_dir)

    if not os.path.isdir(complete_dir):
        os.makedirs(complete_dir)
    ###///###

    filename = './query_data/{}'.format(path)
    torch.save(data, filename)
    print( ' Saving queried output as ' + filename)

def query_occupancy(args, partitions, X, y, framei):
    """
    @params: partitions (array of tuple of ints)
    @params: X (float32 tensor)
    @params: y (float32 tensor)
    @params: framei (int) - the index of the current frame being read

    @returns: []

    Runs and plots occupancy BHM for a single time frame (framei) given input
    parameters
    """
    totalTime = 0
    occupancyPlot = []
    num_segments = len(partitions)
    for i, segi in enumerate(partitions):
        print(' Querying segment {} of {}...'.format(i+1, num_segments))
        bhm_mdl = load_mdl(args, 'occupancy/{}_f{}_p{}'.format(args.save_model_path, framei, i), 'BHM3D_PYTORCH')
        # query the model
        xx, yy, zz = torch.meshgrid(
            torch.arange(segi[0], segi[1], args.q_resolution[0]),
            torch.arange(segi[2], segi[3], args.q_resolution[1]),
            torch.arange(segi[4], segi[5], args.q_resolution[2])
        )
        Xq = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        time1 = time.time()
        yq, var = bhm_mdl.predict(Xq)
        totalTime += time.time()-time1
        occupancyPlot.append((Xq,yq,var))
    print(' Total querying time={} s'.format(round(totalTime, 2)))
    save_query_data((occupancyPlot, X, y, framei), 'occupancy/{}_f{}'.format(args.save_query_data_path, framei))

def query_regression(args, cell_max_min, X, y_occupancy, g, framei):
    """
    @params: cell_max_min (tuple of 6 ints) - bounding area observed
    @params: X (float32 tensor)
    @params: y_occupancy (float32 tensor)
    @params: g (float32 tensor)
    @params: framei (int) - the index of the current frame being read

    @returns: []

    Runs and plots regression for a single time frame (framei) given input
    parameters
    """
    bhm_regression_mdl = load_mdl(args, 'regression/{}_f{}'.format(args.save_model_path, framei), 'BHM_REGRESSION_PYTORCH')
    # filter X,y such that only give the X's where y is 1
    filtered = g[(y_occupancy[:,0]==1),:]
    print(" Querying regression BHM ...")

    # query the model
    xx, yy = torch.meshgrid(
        torch.arange(
            cell_max_min[0],
            cell_max_min[1]+args.q_resolution[0],
            args.q_resolution[0]
        ),
        torch.arange(
            cell_max_min[2],
            cell_max_min[3]+args.q_resolution[1],
            args.q_resolution[1]
        ),
    )
    Xq_mv = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    time1 = time.time()
    mean, var = bhm_regression_mdl.predict(Xq_mv)
    print(' Total querying time={} s'.format(round(time.time()-time1, 2)))
    meanVarPlot = [Xq_mv, mean, var]
    print("mean.shape:", mean.shape, " var.shape:", var.shape)
    save_query_data((meanVarPlot, filtered[:,:3], framei, cell_max_min), 'regression/{}_f{}'.format(args.save_query_data_path, framei))

def query_velocity(args, X, y_vx, y_vy, y_vz, partitions, cell_resolution, cell_max_min, framei):
    bhm_velocity_mdl = load_mdl(args, 'velocity/{}_f{}'.format(args.save_model_path, framei), 'BHM_VELOCITY_PYTORCH')
    #print(" Querying velocity BHM ...")

    if args.q_resolution[0] <= 0 and args.q_resolution[1] <= 0 and args.q_resolution[2] <= 0:
        #if all q_res are non-positive, then query input = X
        print(" Query data is the same as input data")
        Xq_mv = X
    elif args.q_resolution[0] <= 0 or args.q_resolution[1] <= 0 or args.q_resolution[2] <= 0:
        #if at least one q_res is non-positive, then
        if args.q_resolution[0] <= 0: #x-slice
            print(" Query data is x={} slice ".format(args.q_resolution[3]))
            xx, yy, zz = torch.meshgrid(
                torch.arange(
                    args.q_resolution[3],
                    args.q_resolution[3] + 0.1,
                    1
                ),
                torch.arange(
                    cell_max_min[2],
                    cell_max_min[3] + args.q_resolution[1],
                    args.q_resolution[1]
                ),
                torch.arange(
                    cell_max_min[4],
                    cell_max_min[5] + args.q_resolution[2],
                    args.q_resolution[2]
                )
            )
            Xq_mv = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        elif args.q_resolution[1] <= 0: #y-slice
            print("Query data is y={} slice ".format(args.q_resolution[3]))
            xx, yy, zz = torch.meshgrid(
                torch.arange(
                    cell_max_min[0],
                    cell_max_min[1] + args.q_resolution[0],
                    args.q_resolution[0]
                ),
                torch.arange(
                    args.q_resolution[3],
                    args.q_resolution[3] + 0.1,
                    1
                ),
                torch.arange(
                    cell_max_min[4],
                    cell_max_min[5] + args.q_resolution[2],
                    args.q_resolution[2]
                )
            )
            Xq_mv = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        else: #z-slice
            print("Query data is z={} slice ".format(args.q_resolution[3]))
            xx, yy, zz = torch.meshgrid(
                torch.arange(
                    cell_max_min[0],
                    cell_max_min[1] + args.q_resolution[0],
                    args.q_resolution[0]
                ),
                torch.arange(
                    cell_max_min[2],
                    cell_max_min[3] + args.q_resolution[1],
                    args.q_resolution[1]
                ),
                torch.arange(
                    args.q_resolution[3],
                    args.q_resolution[3] + 0.1,
                    1
                )
            )
            Xq_mv = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
    else:
        #if not use the grid
        print("Query data is a 3D gird.")
        xx, yy, zz = torch.meshgrid(
            torch.arange(
                cell_max_min[0],
                cell_max_min[1]+args.q_resolution[0],
                args.q_resolution[0]
            ),
            torch.arange(
                cell_max_min[2],
                cell_max_min[3]+args.q_resolution[1],
                args.q_resolution[1]
            ),
            torch.arange(
                cell_max_min[4],
                cell_max_min[5]+args.q_resolution[2],
                args.q_resolution[2]
            )
        )
        Xq_mv = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)

    # xx, yy = torch.meshgrid(
    #     torch.arange(
    #         cell_max_min[0],
    #         cell_max_min[1]+args.q_resolution[0],
    #         args.q_resolution[0]
    #     ),
    #     torch.arange(
    #         cell_max_min[2],
    #         cell_max_min[3]+args.q_resolution[1],
    #         args.q_resolution[1]
    #     ),
    # )
    # Xq_mv = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    # xx, yy, zz = torch.meshgrid(
    #     torch.arange(segi[0], segi[1], args.q_resolution[0]),
    #     torch.arange(segi[2], segi[3], args.q_resolution[1]),
    #     torch.arange(segi[4], segi[5], args.q_resolution[2])
    # )
    # Xq = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)


    time1 = time.time()

    if args.likelihood_type == "gamma":
        mean_x, mean_y, mean_z = bhm_velocity_mdl.predict(Xq_mv)
    elif args.likelihood_type == "gaussian":
        mean_x, var_x, mean_y, var_y, mean_z, var_z = bhm_velocity_mdl.predict(Xq_mv)
    else:
        raise ValueError("Unsupported likelihood type: \"{}\"".format(args.likelihood_type))

    print(' Total querying time={} s'.format(round(time.time()-time1, 2)))
    # print("mean.shape:", mean.shape)
    # print("Xq_mv.shape:", Xq_mv.shape)
    # var = np.zeros((mean.shape[0], mean.shape[0]))
    # print("var.shape:", var.shape)
    # meanVarPlot = [Xq_mv, mean_x, mean_y, mean_z]
    # print("Before save query data")
    # save_query_data((meanVarPlot, X, framei, cell_max_min), 'velocity/{}_f{}'.format(args.save_query_data_path, framei))
    save_query_data((X, y_vx, y_vy, y_vz, Xq_mv, mean_x, mean_y, mean_z, framei), 'velocity/{}_f{}'.format(args.save_query_data_path, framei))
    # print("Done querying.")
