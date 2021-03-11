import argparse
import json
import os
import pandas as pd
import time
import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from bhmtorch_cpu import BHM3D_PYTORCH, BHM_REGRESSION_PYTORCH, BHM_VELOCITY_PYTORCH
from utils_filereader import read_frame_velocity
from utils_metrics import calc_scores_velocity


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
        raise ValueError("Unknown model type: \"{}\"".format(type))
    # model.updateMuSig(model_params['mu'], model_params['sig'])
    return model, model_params['train_time']


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

def query_occupancy(args, cell_max_min, partitions, X, y, framei):
    """
    @params: partitions (array of tuple of ints)
    @params: cell_max_min (tuple of 6 ints) - bounding area observed
    @params: X (float32 tensor)
    @params: y (float32 tensor)
    @params: framei (int) - the index of the current frame being read

    @returns: []

    Runs and plots occupancy BHM for a single time frame (framei) given input
    parameters
    """
    totalTime = 0
    occupancyPlot = []
    bhm_mdl, train_time = load_mdl(args, 'occupancy/{}_f{}_p{}'.format(args.save_model_path, framei, 0), 'BHM3D_PYTORCH')
    #If not using grid, associate additional zero-mean large-variance points far away from data points to bhm_mdl to correct variance
    if args.hinge_type != "grid":
        x_x, y_y, z_z = torch.meshgrid(
            torch.arange(cell_max_min[0]-abs(3*args.hinge_dist[0]), cell_max_min[1]+abs(3*args.hinge_dist[0]), abs(args.hinge_dist[0])),
            torch.arange(cell_max_min[2]-abs(3*args.hinge_dist[1]), cell_max_min[3]+abs(3*args.hinge_dist[1]), abs(args.hinge_dist[1])),
            torch.arange(cell_max_min[4]-abs(3*args.hinge_dist[2]), cell_max_min[5]+abs(3*args.hinge_dist[2]), abs(args.hinge_dist[2]))
        )
        add_X = torch.stack([x_x.flatten(), y_y.flatten(), z_z.flatten()], dim=1)
        mask = np.sum(euclidean_distances(add_X, X) <= 0.1, axis=1) <= 1
        add_X = add_X[mask, :]

        add_mu = torch.zeros(add_X.shape[0])
        add_var = torch.ones(add_X.shape[0])

        bhm_mdl.append_values(add_X, add_mu, add_var)

    if args.query_dist[0] <= 0 or args.query_dist[1] <= 0 or args.query_dist[2] <= 0:
        #if at least one q_res is non-positive, then
        if args.query_dist[0] <= 0: #x-slice
            print(" Query data is x={} slice ".format(args.query_dist[3]))
            xx, yy, zz = torch.meshgrid(
                torch.arange(
                    args.query_dist[3],
                    args.query_dist[3] + 0.1,
                    1
                ),
                torch.arange(
                    cell_max_min[2],
                    cell_max_min[3] + args.query_dist[1],
                    args.query_dist[1]
                ),
                torch.arange(
                    cell_max_min[4],
                    cell_max_min[5] + args.query_dist[2],
                    args.query_dist[2]
                )
            )
            Xq = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
            option = 'X slice at '.format(args.query_dist[3])
        elif args.query_dist[1] <= 0: #y-slice
            print("Query data is y={} slice ".format(args.query_dist[3]))
            xx, yy, zz = torch.meshgrid(
                torch.arange(
                    cell_max_min[0],
                    cell_max_min[1] + args.query_dist[0],
                    args.query_dist[0]
                ),
                torch.arange(
                    args.query_dist[3],
                    args.query_dist[3] + 0.1,
                    1
                ),
                torch.arange(
                    cell_max_min[4],
                    cell_max_min[5] + args.query_dist[2],
                    args.query_dist[2]
                )
            )
            Xq = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
            option = 'Y slice at '.format(args.query_dist[3])
        else: #z-slice
            print("Query data is z={} slice ".format(args.query_dist[3]))
            xx, yy, zz = torch.meshgrid(
                torch.arange(
                    cell_max_min[0],
                    cell_max_min[1] + args.query_dist[0],
                    args.query_dist[0]
                ),
                torch.arange(
                    cell_max_min[2],
                    cell_max_min[3] + args.query_dist[1],
                    args.query_dist[1]
                ),
                torch.arange(
                    args.query_dist[3],
                    args.query_dist[3] + 0.1,
                    1
                )
            )
            Xq = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
            option = 'Z slice at '.format(args.query_dist[3])

        time1 = time.time()
        yq, var = bhm_mdl.predictSampling(Xq, nSamples=50)
        totalTime += time.time()-time1
        occupancyPlot.append((Xq,yq,var))
    else:
        num_segments = len(partitions)
        for i, segi in enumerate(partitions):
            print(' Querying segment {} of {}...'.format(i+1, num_segments))
            # query the model
            xx, yy, zz = torch.meshgrid(
                torch.arange(segi[0], segi[1], args.query_dist[0]),
                torch.arange(segi[2], segi[3], args.query_dist[1]),
                torch.arange(segi[4], segi[5], args.query_dist[2])
            )
            Xq = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
            time1 = time.time()
            yq, var = bhm_mdl.predictSampling(Xq, nSamples=50)
            totalTime += time.time()-time1
            occupancyPlot.append((Xq,yq,var))
        option = 'grid'


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
    bhm_regression_mdl, train_time = load_mdl(args, 'regression/{}_f{}'.format(args.save_model_path, framei), 'BHM_REGRESSION_PYTORCH')
    # filter X,y such that only give the X's where y is 1
    filtered = g[(y_occupancy[:,0]==1),:]
    print(" Querying regression BHM ...")

    # query the model
    xx, yy = torch.meshgrid(
        torch.arange(
            cell_max_min[0],
            cell_max_min[1]+args.query_dist[0],
            args.query_dist[0]
        ),
        torch.arange(
            cell_max_min[2],
            cell_max_min[3]+args.query_dist[1],
            args.query_dist[1]
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
    bhm_velocity_mdl, train_time = load_mdl(args, 'velocity/{}_f{}'.format(args.save_model_path, framei), 'BHM_VELOCITY_PYTORCH')

    option = ''
    if args.eval_path != '':
        #if eval is True, test the query
        print(" Query data from the test dataset")
        Xq_mv, y_vx_true, y_vy_true, y_vz_true, _ = read_frame_velocity(args, framei, args.eval_path, cell_max_min)
        option = args.eval_path
    elif args.query_dist[0] <= 0 and args.query_dist[1] <= 0 and args.query_dist[2] <= 0:
        #if all q_res are non-positive, then query input = X
        print(" Query data is the same as input data")
        Xq_mv = X
        option = 'Train data'
    elif args.query_dist[0] <= 0 or args.query_dist[1] <= 0 or args.query_dist[2] <= 0:
        #if at least one q_res is non-positive, then
        if args.query_dist[0] <= 0: #x-slice
            print(" Query data is x={} slice ".format(args.query_dist[3]))
            xx, yy, zz = torch.meshgrid(
                torch.arange(
                    args.query_dist[3],
                    args.query_dist[3] + 0.1,
                    1
                ),
                torch.arange(
                    cell_max_min[2],
                    cell_max_min[3] + args.query_dist[1],
                    args.query_dist[1]
                ),
                torch.arange(
                    cell_max_min[4],
                    cell_max_min[5] + args.query_dist[2],
                    args.query_dist[2]
                )
            )
            Xq_mv = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
            option = 'X slice at '.format(args.query_dist[3])
        elif args.query_dist[1] <= 0: #y-slice
            print("Query data is y={} slice ".format(args.query_dist[3]))
            xx, yy, zz = torch.meshgrid(
                torch.arange(
                    cell_max_min[0],
                    cell_max_min[1] + args.query_dist[0],
                    args.query_dist[0]
                ),
                torch.arange(
                    args.query_dist[3],
                    args.query_dist[3] + 0.1,
                    1
                ),
                torch.arange(
                    cell_max_min[4],
                    cell_max_min[5] + args.query_dist[2],
                    args.query_dist[2]
                )
            )
            Xq_mv = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
            option = 'Y slice at '.format(args.query_dist[3])
        else: #z-slice
            print("Query data is z={} slice ".format(args.query_dist[3]))
            xx, yy, zz = torch.meshgrid(
                torch.arange(
                    cell_max_min[0],
                    cell_max_min[1] + args.query_dist[0],
                    args.query_dist[0]
                ),
                torch.arange(
                    cell_max_min[2],
                    cell_max_min[3] + args.query_dist[1],
                    args.query_dist[1]
                ),
                torch.arange(
                    args.query_dist[3],
                    args.query_dist[3] + 0.1,
                    1
                )
            )
            Xq_mv = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
            option = 'Z slice at '.format(args.query_dist[3])
    else:
        #if not use the grid
        print("Query data is a 3D gird.")
        xx, yy, zz = torch.meshgrid(
            torch.arange(
                cell_max_min[0],
                cell_max_min[1]+args.query_dist[0],
                args.query_dist[0]
            ),
            torch.arange(
                cell_max_min[2],
                cell_max_min[3]+args.query_dist[1],
                args.query_dist[1]
            ),
            torch.arange(
                cell_max_min[4],
                cell_max_min[5]+args.query_dist[2],
                args.query_dist[2]
            )
        )
        Xq_mv = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        option = '3D grid'

    time1 = time.time()

    if args.likelihood_type == "gamma":
        mean_x, mean_y, mean_z = bhm_velocity_mdl.predict(Xq_mv)
    elif args.likelihood_type == "gaussian":
        mean_x, var_x, mean_y, var_y, mean_z, var_z = bhm_velocity_mdl.predict(Xq_mv, args.query_blocks, args.variance_only)
    else:
        raise ValueError("Unsupported likelihood type: \"{}\"".format(args.likelihood_type))

    query_time = time.time() - time1

    print(' Total querying time={} s'.format(round(query_time, 2)))
    save_query_data((X, y_vx, y_vy, y_vz, Xq_mv, mean_x, mean_y, mean_z, framei), \
                    'velocity/{}_f{}'.format(args.save_query_data_path, framei))

    if args.eval == 1:
        if hasattr(args, 'report_notes'):
            notes = args.report_notes
        else:
            notes = ''
        axes = [('x', y_vx_true, mean_x, var_x), ('y', y_vy_true, mean_y, var_y), ('z', y_vz_true, mean_z, var_z)]
        for axis, Xqi, mean, var in axes:
            mdl_name = 'reports/' + args.plot_title + '_' + axis
            calc_scores_velocity(mdl_name, option, Xqi.numpy(), mean.numpy().ravel(), predicted_var=\
                np.diagonal(var.numpy()), train_time=train_time, query_time=query_time, save_report=True, notes=notes)
