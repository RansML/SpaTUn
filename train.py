import argparse
import json
import os
import pandas as pd
import time
import torch
import numpy as np

from bhmtorch_cpu import BHM3D_PYTORCH, BHM_REGRESSION_PYTORCH, BHM_VELOCITY_PYTORCH
from plot import BHM_PLOTTER


"""
- features are the kernel distance thing to hinge points?
- output is a scalar value (ie: the velocity long X-Y plane, X-Z plane, or Z-Y plane)?
- How do you determine what the hinge points are?
    - q_resolution in config file determines how frequently to space the hinge points?
    - How do you determine how big the total grid is on which to draw the hinge points?
- What are the partitions?

- What exactly is the model learning? How can information about whether or not a space is
occupied, etc. be gleaned from the current features?

- Try making regression grid 3d
    - will this break the plotting??

- train velocity
- query velocity

----



"""


# ==============================================================================
# Utility Functions
# ==============================================================================
def format_config():
    """
    Formats default parameters from argparse to be easily digested by module
    """
    # default parameter to reduce min and max bounds of cell_max_min
    delta = (args.area_max[0] - args.area_min[0])*0.03

    fn_train = os.path.abspath(args.dataset_path)
    cell_resolution = (
        args.h_res[0],
        args.h_res[1],
        args.h_res[2]
    )
    cell_max_min = [
        args.area_min[0] + delta,
        args.area_max[0] - delta,
        args.area_min[1] + delta,
        args.area_max[1] - delta,
        args.area_min[2] + delta,
        args.area_max[2] - delta
    ]
    return fn_train, cell_max_min, cell_resolution

def get3DPartitions(cell_max_min, nPartx1, nPartx2, nPartx3):
    """
    @param cell_max_min: The size of the entire area
    @param nPartx1: How many partitions along the longitude
    @param nPartx2: How many partitions along the latitude
    @param nPartx3: How many partition along the altitude
    @return: a list of all partitions
    """
    width = cell_max_min[1] - cell_max_min[0]
    length = cell_max_min[3] - cell_max_min[2]
    height = cell_max_min[5] - cell_max_min[4]

    x_margin = width/2
    y_margin = length/2
    z_margin = height/2

    x_partition_size = width/nPartx1
    y_partition_size = length/nPartx2
    z_partition_size = height/nPartx3
    cell_max_min_segs = []
    for x in range(nPartx1):
        for y in range(nPartx2):
            for z in range(nPartx3):
                seg_i = (
                    cell_max_min[0] + x_partition_size*(x-args.partition_bleed),  # Lower X
                    cell_max_min[0] + x_partition_size*(x+1+args.partition_bleed),  # Upper X
                    cell_max_min[2] + y_partition_size*(y-args.partition_bleed),  # Lower Y
                    cell_max_min[2] + y_partition_size*(y+1+args.partition_bleed),  # Upper Y
                    cell_max_min[4] + z_partition_size*(z-args.partition_bleed),  # Lower Z
                    cell_max_min[4] + z_partition_size*(z+1+args.partition_bleed)  # Upper Z
                )
                cell_max_min_segs.append(seg_i)
    return cell_max_min_segs

def read_frame(framei, fn_train, cell_max_min):
    """
    @params: framei (int) — the index of the current frame being read
    @params: fn_train (str) — the path of the dataset frames being read
    @params: cell_max_min (tuple of 6 ints) — bounding area observed

    @returns: g (float32 tensor)
    @returns: X (float32 tensor)
    @returns: y_occupancy (float32 tensor)
    @returns: sigma (float32 tensor)
    @returns: cell_max_min_segments — partitioned frame data for frame i

    Reads a single frame (framei) of the dataset defined by (fn_train) and
    and returns LIDAR hit data corresponding to that frame and its partitions
    """
    print('\nReading '+fn_train+'.csv...')
    g = pd.read_csv(fn_train+'.csv', delimiter=',')
    g = g.loc[g['t'] == framei].values[:, 2:]

    g = torch.tensor(g, dtype=torch.float32)
    X = g[:, :3]
    y_occupancy = g[:, 3].reshape(-1, 1)
    sigma = g[:, 4:]

    # If there are no defaults, automatically set bounding area.
    if cell_max_min[0] == None:
        cell_max_min[0] = X[:,0].min()
    if cell_max_min[1] == None:
        cell_max_min[1] = X[:,0].max()
    if cell_max_min[2] == None:
        cell_max_min[2] = X[:,1].min()
    if cell_max_min[3] == None:
        cell_max_min[3] = X[:,1].max()
    if cell_max_min[4] == None:
        cell_max_min[4] = X[:,2].min()
    if cell_max_min[5] == None:
        cell_max_min[5] = X[:,2].max()

    cell_max_min_segments = get3DPartitions(tuple(cell_max_min), args.num_partitions[0], args.num_partitions[1], args.num_partitions[2])
    return g, X, y_occupancy, sigma, cell_max_min_segments

def read_frame_velocity(framei, fn_train, cell_max_min):
    print('\nReading '+fn_train+'.csv...')
    g = pd.read_csv(fn_train+'.csv', delimiter=',')
    g = g.loc[g['t'] == framei].values[:, 1:]

    # print("g:", g)

    # g = torch.tensor(g, dtype=torch.double)
    g = torch.tensor(g, dtype=torch.float32)
    X = g[:, :3]

    # print("X:", X)
    # print("g[:, 3]:", g[:, 3])

    y_vx = g[:, 3].reshape(-1, 1)
    y_vy = g[:, 4].reshape(-1, 1)
    y_vz = g[:, 5].reshape(-1, 1)

    # print("y_vx:", y_vx)
    # exit()

    # If there are no defaults, automatically set bounding area.
    if cell_max_min[0] == None:
        cell_max_min[0] = X[:,0].min()
    if cell_max_min[1] == None:
        cell_max_min[1] = X[:,0].max()
    if cell_max_min[2] == None:
        cell_max_min[2] = X[:,1].min()
    if cell_max_min[3] == None:
        cell_max_min[3] = X[:,1].max()
    if cell_max_min[4] == None:
        cell_max_min[4] = X[:,2].min()
    if cell_max_min[5] == None:
        cell_max_min[5] = X[:,2].max()

    # print("(read) X.shape:", X.shape)

    cell_max_min_segments = get3DPartitions(tuple(cell_max_min), args.num_partitions[0], args.num_partitions[1], args.num_partitions[2])
    return X, y_vx, y_vy, y_vz, cell_max_min_segments

def save_mdl(model, path):
    """
    @param model: BHM Module to save
    @param path (str): path relative to the mdl folder to save to
    """
    mdl_type = type(model).__name__
    print("mdl_type:", mdl_type)
    if mdl_type == 'BHM3D_PYTORCH':
        ###===###
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
        print("Saving...")
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
        print("Saved.")
    else:
        raise ValueError("Unknown model type: \"{}\"".format(mdl_type))

def load_mdl(path, type):
    """
    @param path (str): path relative to the mdl folder to load from
    @param type (str): type of model to load ('BHM3D_PYTORCH' / 'BHM_REGRESSION_PYTORCH')
    @returns: model: BHM Module that is loaded
    """
    model_params = torch.load('./mdls/{}'.format(path))
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
        print("Loading velocity model")

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

    print("before torch.save")
    torch.save(data, './query_data/{}'.format(path))
    print("After torch.save")

def load_query_data(path):
    """
    @param path (str): path relative to query_data folder to save data
    """
    return torch.load('./query_data/{}'.format(path))

# ==============================================================================
# Training
# ==============================================================================
def train(fn_train, cell_max_min, cell_resolution):
    """
    @params: [fn_train, cell_max_min, cell_resolution]
    @returns: []
    Fits the 3D BHM on each frame of the dataset and plots occupancy or regression
    """

    print("\nfn_train:", fn_train)

    alpha = 10**-2
    beta = 10**2
    for framei in range(args.num_frames):
        if args.model_type == "occupancy" or args.model_type == "regression":
            g, X, y_occupancy, sigma, partitions = read_frame(framei, fn_train, cell_max_min)
        elif args.model_type == "velocity":
            X, y_vx, y_vy, y_vz, partitions = read_frame_velocity(framei, fn_train, cell_max_min)
        else:
            raise ValueError("Unknown model type: \"{}\"".format(args.model_type))


        if args.model_type == 'occupancy':
            train_occupancy(partitions, cell_resolution, X, y_occupancy, sigma, framei)
        elif args.model_type == 'regression':
            train_regression(alpha, beta, cell_resolution, cell_max_min, X, y_occupancy, g, sigma[:,:2], framei)
            # For regression, we use sigma dimension 2. This is hard coded in the pass to plot_regression for the sigma term above
        elif args.model_type == "velocity": ###===###
            # print("X:", X)
            # print("partitions:", partitions)
            train_velocity(alpha, beta, X, y_vx, y_vy, y_vz, partitions, cell_resolution, cell_max_min, framei) ###///###


        if args.model_type == "occupancy" or args.model_type == "regression":
            del g, X, y_occupancy, sigma, partitions
        elif args.model_type == "velocity":
            del X, y_vx, y_vy, y_vz, partitions
        else:
            raise ValueError("Unknown model type: \"{}\"".format(args.model_type))

def train_occupancy(partitions, cell_resolution, X, y, sigma, framei):
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
        save_mdl(bhm_mdl, '{}_f{}_p{}'.format(args.save_model_path, framei, i))
        del bhm_mdl
    print(' Total training time={} s'.format(round(totalTime, 2)))
    del sigma

def train_regression(alpha, beta, cell_resolution, cell_max_min, X, y_occupancy, g, sigma, framei):
    """
    @params: alpha (float)
    @params: beta (float)
    @params: cell_resolution (tuple of 3 ints)
    @params: cell_max_min (tuple of 6 ints) — bounding area observed
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
    save_mdl(bhm_regression_mdl, '{}_f{}'.format(args.save_model_path, framei))
    del bhm_regression_mdl, filtered, sigma

def train_velocity(alpha, beta, X, y_vx, y_vy, y_vz, partitions, cell_resolution, cell_max_min, framei):
    print("In train velocity")
    # exit()
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
        raise ValueError("Unsupported likelihood type: \"{}\"".format(args.likelihood_type))

    time1 = time.time()
    bhm_velocity_mdl.fit(X, y_vx, y_vy, y_vz, eps=2) # , y_vy, y_vz
    print(' Total training time={} s'.format(round(time.time() - time1, 2)))
    save_mdl(bhm_velocity_mdl, '{}_f{}'.format(args.save_model_path, framei))
    del bhm_velocity_mdl

# ==============================================================================
# Query
# ==============================================================================
def query(fn_train, cell_max_min):
    """
    @params: [fn_train, cell_max_min]
    @returns: []
    Queries the 3D BHM for occupancy or regression on each frame of the dataset
    """
    for framei in range(args.num_frames):
        if args.model_type == "occupancy" or args.model_type == "regression":
            g, X, y_occupancy, sigma, partitions = read_frame(framei, fn_train, cell_max_min)
        elif args.model_type == "velocity":
            X, y_vx, y_vy, y_vz, partitions = read_frame_velocity(framei, fn_train, cell_max_min)
        else:
            raise ValueError("Unknown model type: \"{}\"".format(args.model_type))

        if args.model_type == 'occupancy':
            query_occupancy(partitions, X, y_occupancy, framei)
        elif args.model_type == 'regression':
            query_regression(cell_max_min, X, y_occupancy, g, framei)
        elif args.model_type == "velocity": ###===###
            query_velocity(X, y_vx, y_vy, y_vz, partitions, cell_resolution, cell_max_min, framei) ###///###

        if args.model_type == "occupancy" or args.model_type == "regression":
            del g, X, y_occupancy, sigma, partitions
        elif args.model_type == "velocity":
            del X, y_vx, y_vy, y_vz, partitions
        else:
            raise ValueError("Unknown model type: \"{}\"".format(args.model_type))

def query_occupancy(partitions, X, y, framei):
    """
    @params: partitions (array of tuple of ints)
    @params: X (float32 tensor)
    @params: y (float32 tensor)
    @params: framei (int) — the index of the current frame being read

    @returns: []

    Runs and plots occupancy BHM for a single time frame (framei) given input
    parameters
    """
    totalTime = 0
    occupancyPlot = []
    num_segments = len(partitions)
    for i, segi in enumerate(partitions):
        print(' Querying segment {} of {}...'.format(i+1, num_segments))
        bhm_mdl = load_mdl('occupancy/{}_f{}_p{}'.format(args.save_model_path, framei, i), 'BHM3D_PYTORCH')
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

def query_regression(cell_max_min, X, y_occupancy, g, framei):
    """
    @params: cell_max_min (tuple of 6 ints) — bounding area observed
    @params: X (float32 tensor)
    @params: y_occupancy (float32 tensor)
    @params: g (float32 tensor)
    @params: framei (int) — the index of the current frame being read

    @returns: []

    Runs and plots regression for a single time frame (framei) given input
    parameters
    """
    bhm_regression_mdl = load_mdl('regression/{}_f{}'.format(args.save_model_path, framei), 'BHM_REGRESSION_PYTORCH')
    # filter X,y such that only give the X's where y is 1
    filtered = g[(y_occupancy[:,0]==1),:]
    print("Querying regression BHM ...")

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

def query_velocity(X, y_vx, y_vy, y_vz, partitions, cell_resolution, cell_max_min, framei):
    bhm_velocity_mdl = load_mdl('velocity/{}_f{}'.format(args.save_model_path, framei), 'BHM_VELOCITY_PYTORCH')
    print("Querying velocity BHM ...")
    #
    # print("(query) X.shape:", X.shape)



    # # query the model
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
# ==============================================================================
# Plot
# ==============================================================================
def plot():
    """
    @params: []
    @returns: []
    Plots data loaded from the args.save_query_data_path parameter
    """
    plotter = BHM_PLOTTER(args, args.plot_title, args.surface_threshold, args.q_resolution, occupancy_plot_type=args.occupancy_plot_type)
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
            print("(plot) X.shape:", X.shape)
            # exit()
            plotter.plot_velocity_frame(X, y_vx, y_vy, y_vz, Xq_mv, mean_x, mean_y, mean_z, framei)



# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, help='tqp: Train Query and Plot, to: Train only, qo: Query only, po: Plot only')

    # Training arguments
    parser.add_argument('--dataset_path', type=str, help='Path to dataset')
    parser.add_argument('--num_frames', type=int, help='Number of data frames')
    parser.add_argument('--config', type=str, help='Path to the config to load relative to the config folder')
    parser.add_argument('--save_config', type=str, help='Saves the argparse config to path if set relative to the config folder')
    parser.add_argument('--save_model_path', type=str, help='Path to save each model \
        (i.e. save_model_path is set to \"toy3_run0\", then the model at partition 1, frame 1 would save to \
        mdls/occupancy/toy3_run0_f1_p1)'
    )
    parser.add_argument('--save_query_data_path', type=str, help='Path save each set of queried data \
        (i.e. save_model_path is set to \"toy3_run0\" and the model type is set to occupancy, \
        then the model at frame 1 would save to query_data/occupancy/toy3_run0_f1_p1)'
    )

    # BHM Arguments
    parser.add_argument('--q_resolution', nargs=3, type=float, help='X Y Z Q-resolution (3 values)')
    parser.add_argument('--gamma', nargs='+', type=float, help='X Y Z Gamma (1-3 values)')
    parser.add_argument('--h_res', nargs=3, type=int, help='X Y Z hinge point resolution (3 values)')
    parser.add_argument('--num_partitions', nargs=3, type=int, help='X Y Z number of partitions per axis (3 values)')
    parser.add_argument('--partition_bleed', type=float, help='Amount of bleed between partitions for plot stitching')
    parser.add_argument('--area_min', nargs=3, type=int, help='X Y Z minimum coordinates in bounding box (3 values)')
    parser.add_argument('--area_max', nargs=3, type=int, help='X Y Z maximum coordinates in bounding box (3 values)')
    parser.add_argument('--kernel_type', type=str, help='Type of RBF kernel: Convolution (conv), Wasserstein (wass)')
    parser.add_argument('--occupancy_plot_type', type=str, help='Plot occupancy as scatter or volumetric plot')

    # Plot Arguments
    parser.add_argument('--plot_title', type=str, help='Name to each frame plot')
    parser.add_argument('--surface_threshold', type=float, help='Minimum threshold to show surface prediction on plot')
    parser.add_argument('--model_type', type=str, help='Plot type (occupancy vs regression)')
    parser.add_argument('--likelihood_type', type=str, help='Likelihood type (Gamma, Gaussian)')

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("\n\nargs.save_model_path:", args.save_model_path)

    # Set arguments according to the following Priority (High->Low):
    # 1:CL provided arguments, 2: config provided arguments, 3:default arguments
    if args.config:
        config = json.load(open('./configs/' + args.config, 'r'))
        defaults = json.load(open('./configs/defaults', 'r'))
        for key in vars(args):
            if key == 'save_config': continue
            if getattr(args, key): continue
            if key in config and config[key]:
                args.__dict__[key] = config[key]
            else:
                args.__dict__[key] = defaults[key]
    if args.save_config:
        with open('./configs/' + args.save_config, 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    assert len(args.gamma) <= 3, 'Cannot support gamma with greater than dimension 3.'

    fn_train, cell_max_min, cell_resolution = format_config()
    print("\ncell_max_min:", cell_max_min)
    print("cell_resolution:", cell_resolution, "\n")
    if args.mode == 'tqp' or args.mode == 'to':
        train(fn_train, cell_max_min, cell_resolution)
    if args.mode == 'tqp' or args.mode == 'qo':
        query(fn_train, cell_max_min)
    if args.mode == 'tqp' or args.mode == 'po':
        plot()

    print("Complete.")
