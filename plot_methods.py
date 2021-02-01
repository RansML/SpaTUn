import argparse
import json
import os
import pandas as pd
import plotly
import plotly.graph_objects as go
import time
import torch

#vivian
import plotly.figure_factory as ff
import trimesh
import time
import torch
import numpy as np
from skimage import measure
import utils_filereader

from bhmtorch_cpu import BHM3D_PYTORCH
from bhmtorch_cpu import BHM_REGRESSION_PYTORCH
from plotly.subplots import make_subplots

# plotly.io.orca.config.executable = "/home/khatch/anaconda3/envs/hilbert/bin/orca"
# import plotly.io as pio
# pio.orca.config.use_xvfb = True
# plotly.io.orca.config.executable = "/home/khatch/Documents/orca-1.3.1.AppImage"

# ==============================================================================
# BHM Plotting Class
# ==============================================================================
class BHM_PLOTTER():
    def __init__(self, args, plot_title, surface_threshold, query_dist, occupancy_plot_type='scatter', plot_denoise=0.98):
        self.args = args
        self.plot_title = plot_title
        self.surface_threshold = surface_threshold
        self.query_dist = query_dist
        self.occupancy_plot_type = occupancy_plot_type
        self.plot_denoise = plot_denoise
        print(' Successfully initialized plotly plotting class')

    def filter_predictions(self, ploti):
        """
        @param ploti: (Xq, yq, var) (3D location and occupancy prediction)
        @return toPlot: array of (Xq, yq) filtered to show only occupancy surface
            with yq greater than 'surface_threshold'
        """
        if self.occupancy_plot_type == 'volumetric':
            return ploti
        joined = torch.cat((ploti[0], ploti[1].unsqueeze(-1)), dim=-1)
        mask = (joined[:,3] >= self.surface_threshold) #+ (torch.rand_like(joined[:,3]) > self.plot_denoise)
        filtered = joined[mask, :]
        return filtered[:,:3], filtered[:,3]

    def plot_lidar_hits(self, X, y, fig):
        """
        Occupancy: Plots 3D scatterplot of LIDAR hits and misses

        @param X: 3D coordinates for each lidar observation
        @param y: occupancy observed by lidar
        @param fig: plotly fig
        """
        x_obs = X[:,0]
        y_obs = X[:,1]
        z_obs = X[:,2]
        colorbar_len = 1
        colorbar_y = 0.5
        fig.append_trace(
            go.Scatter3d(
                x=x_obs,
                y=y_obs,
                z=z_obs,
                mode='markers',
                marker=dict(
                    color=y.flatten(),
                    colorscale=[[0, 'blue'],[0.1, 'blue'],[0.9, 'red'],[1, 'red']],
                    cmax=y.max().item(),
                    cmin=y.min().item(),
                    colorbar=dict(
                        x=0.24,
                        len=colorbar_len,
                        y=colorbar_y
                    ),
                    opacity=0.1,
                    size=2,
                )
            ),
            row=1,
            col=1
        )

    def plot_predictions(self, toPlot, fig, iframe):
        """
        Occupancy: Plots volumetric plot of predictions

        @param toPlot: array of (Xq, yq, vars) (3D location and occupancy prediction)
        @param fig: plotly fig
        @param iframe: ith frame
        """
        Xqs = torch.zeros((1, 3))
        yqs = torch.ones(1)
        vars = torch.zeros(1)
        for ploti in toPlot:
            var = ploti[2]
            if self.surface_threshold[0] > 0:
                ploti = self.filter_predictions(ploti)
            Xq, yq = ploti[0], ploti[1]
            if Xq.shape[0] <= 1: continue
            Xqs = torch.cat((Xqs, Xq), dim=0)
            yqs = torch.cat((yqs, yq), dim=0)
            vars = torch.cat((vars, var), dim=0)
        vars = vars[1:]
        yqs = yqs[1:]
        print('Num points plotted after filtering: {}'.format(yqs.shape[0]))

        colorbar_len = 1
        colorbar_y = 0.5
        if self.occupancy_plot_type == 'volumetric':
            fig.add_trace(
                go.Volume(
                    x=Xqs[1:, 0],
                    y=Xqs[1:, 1],
                    z=Xqs[1:, 2],
                    isomin=0,
                    isomax=1,
                    value=yqs,
                    opacity=0.05,
                    surface_count=40,
                    colorscale="Jet",
                    opacityscale=[[0, 0], [self.surface_threshold[0], 0], [1, 1]],
                    colorbar=dict(
                        x=0.47,
                        len=colorbar_len,
                        y=colorbar_y
                    ),
                    cmax=1,
                    cmin=self.surface_threshold[0],
                ),
                row=1,
                col=2
            )
        elif self.occupancy_plot_type == 'scatter':
            fig.add_trace(
                go.Scatter3d(
                    x=Xqs[1:, 0],
                    y=Xqs[1:, 1],
                    z=Xqs[1:, 2],
                    mode='markers',
                    marker=dict(
                        color=yqs,
                        colorscale="Jet",
                        cmax=yqs.max().item(),
                        cmin=yqs.min().item(),
                        colorbar=dict(
                            x=0.64,
                            len=colorbar_len,
                            y=colorbar_y
                        ),
                        opacity=0.1,
                        symbol='square',
                    ),
                ),
                row=1,
                col=2
            )

        # Add variance plot
        fig.add_trace(
            go.Scatter3d(
                x=Xqs[1:, 0],
                y=Xqs[1:, 1],
                z=Xqs[1:, 2],
                mode='markers',
                marker=dict(
                    color=vars,
                    colorscale="Jet",
                    cmax=vars.max().item(),
                    cmin=vars.min().item(),
                    colorbar=dict(
                        x=0.74,
                        len=colorbar_len,
                        y=colorbar_y
                    ),
                    opacity=0.05,
                    symbol='square',
                ),
            ),
            row=1,
            col=3
        )

    def plot_marching_cubes(self, toPlot, fig, iframe):
        if self.args.query_dist[0] > 0 and self.args.query_dist[1] > 0 and self.args.query_dist[2] > 0:
            #if all query-distances are positive, then we have no slices and can proceed with marching cubes
            yqs = torch.ones(1)
            for ploti in toPlot:
                if self.surface_threshold[0] > 0:
                    ploti = self.filter_predictions(ploti)
                Xq, yq = ploti[0], ploti[1]
                if Xq.shape[0] <= 1: continue
                yqs = torch.cat((yqs, yq), dim=0)
            yqs = yqs[1:]
            xx = []
            yy = []
            zz = []
            fn_train, cell_max_min, cell_resolution = utils_filereader.format_config(self.args)
            g, X, y_occupancy, sigma, partitions = utils_filereader.read_frame(self.args, iframe, fn_train, cell_max_min)
            for i, segi in enumerate(partitions):
                # query the model
                xx.extend(torch.arange(segi[0], segi[1], self.args.query_dist[0]).tolist())
                yy.extend(torch.arange(segi[2], segi[3], self.args.query_dist[1]).tolist())
                zz.extend(torch.arange(segi[4], segi[5], self.args.query_dist[2]).tolist())

            surface = yqs.reshape((len(xx), len(yy), len(zz))).numpy()
            mcubes_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(surface)
            trimesh.exchange.export.export_mesh(mcubes_mesh, "plots/surface/out.stl", file_type='stl')

            vertices, simplices, normals, values = measure.marching_cubes(surface, level=None, spacing=(self.query_dist[0], self.query_dist[1], self.query_dist[2]))
            x, y, z = zip(*vertices)
            # rescale to center at zero
            x -= max(x)/2
            y -= max(y)/2
            z -= max(z)/2

            # Add plot for marching cubes
            fig_mcubes = ff.create_trisurf(
                x=x,
                y=y,
                z=z,
                show_colorbar=True,
                plot_edges=True,
                simplices=simplices,
            )

            fig.add_trace(fig_mcubes.data[0], row=1, col=4)

    def plot_hits_surface(self, X, fig):
        """
        Regression: Plots 3D scatter plot and 2D heightmap contour for LIDAR hits

        @param X: 3D coordinates for each lidar observation
        @param fig: plotly fig
        """
        x_obs = X[:,0]
        y_obs = X[:,1]
        z_obs = X[:,2]
        colorbar_len = 1.05
        colorbar_thickness = 15
        colorbar_yaxis = 0.5
        fig.append_trace(
            go.Scatter3d(
                x=x_obs,
                y=y_obs,
                z=z_obs,
                scene='scene1',
                mode='markers',
                marker=dict(
                    color=z_obs,
                    colorscale='Jet',
                    cmax=z_obs.max().item(),
                    cmin=z_obs.min().item(),
                    colorbar=dict(
                        x=0.29,
                        len=colorbar_len,
                        y=colorbar_yaxis,
                        thickness=colorbar_thickness
                    ),
                    size=1,
                )
            ),
            row=1,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=x_obs,
                y=y_obs,
                mode='markers',
                marker=dict(
                    color=z_obs,
                    colorscale='jet',
                    cmax=z_obs.max().item(),
                    cmin=z_obs.min().item(),
                    size=5,
                ),
                showlegend=False,
            ),
            row=2,
            col=1
        )

    def plot_mean_var(self, meanVarPlot, fig, cell_max_min):
        """
        Regression: Plot mean and variance in 3D and 2D based off of predictions

        @param meanVarPlot: array of (Xq, yq)
        @param fig: plotly fig
        """
        Xqs, means, vars = meanVarPlot[0], meanVarPlot[1], torch.diag(meanVarPlot[2])
        x = torch.arange(cell_max_min[0], cell_max_min[1]+self.query_dist[0], self.query_dist[0])
        y = torch.arange(cell_max_min[2], cell_max_min[3]+self.query_dist[1], self.query_dist[1])
        colorbar_len = 1.05
        colorbar_thickness = 15
        colorbar_yaxis = 0.5
        fig.append_trace(
            go.Surface(
                x=x,
                y=y,
                z=means.view(x.shape[0], y.shape[0]).T,
                scene='scene2',
                colorscale='Jet',
                colorbar=dict(
                    x=0.65,
                    len=colorbar_len,
                    y=colorbar_yaxis,
                    thickness=colorbar_thickness
                )
            ),
            row=1,
            col=2
        )
        fig.append_trace(
            go.Surface(
                x=x,
                y=y,
                z=vars.view(x.shape[0], y.shape[0]).T,
                scene='scene3',
                colorscale='Jet',
                colorbar=dict(
                    x=1,
                    len=colorbar_len,
                    y=colorbar_yaxis,
                    thickness=colorbar_thickness
                )
            ),
            row=1,
            col=3
        )
        fig.add_trace(
            go.Contour(
                x=Xqs[:,0],
                y=Xqs[:,1],
                z=means,
                colorscale='jet',
                showscale=False,
                contours=dict(
                    start=means.min().item(),
                    end=means.max().item(),
                    size=0.5,  # use for toy dataset
                ),
            ),
            row=2,
            col=2
        )
        fig.add_trace(
            go.Contour(
                x=Xqs[:,0],
                y=Xqs[:,1],
                z=vars,
                colorscale='jet',
                showscale=False,
                contours=dict(
                    start=vars.min().item(),
                    end=vars.max().item(),
                    size=0.0005,
                ),
            ),
            row=2,
            col=3,
        )

    def plot_occupancy_frame(self, toPlot, X, y, i):
        """
        @param: toPlot array of (Xq, yq) (3D location and occupancy prediction)
        @param: X 3D coordinates for each lidar observation
        @param: y occupancy observed by lidar
        @param: i frame i
        @returns: []
        Plots a single frame (i) of occupancy BHM predictions and LIDAR observations
        """
        time1 = time.time()
        specs = [[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}, {"type": "scene"}]]
        titles = ['Lidar Hits', 'Occupancy Prediction', 'Variance', 'Marching cubes']
        fig = make_subplots(
            rows=1,
            cols=4,
            specs=specs,
            subplot_titles=titles
        )
        self.plot_lidar_hits(X, y, fig)
        self.plot_predictions(toPlot, fig, i)
        self.plot_marching_cubes(toPlot, fig, i)
        camera = dict(
            eye=dict(x=3.2, y=-3.2, z=3.2)
        )
        fig.layout.scene1.camera = camera
        fig.layout.scene2.camera = camera
        fig.layout.scene3.camera = camera
        fig.layout.scene4.camera = camera
        fig.update_layout(title='{}_occupancy_frame{}'.format(self.plot_title, i), height=800)
        plotly.offline.plot(fig, filename=os.path.abspath('./plots/surface/{}_frame{}.html'.format(self.plot_title, i)), auto_open=True)
        print('Completed plotting in %2f s' % (time.time()-time1))

    def plot_regression_frame(self, meanVarPlot, X, i, cell_max_min):
        """
        @param: meanVarPlot mean and variance
        @param: X 3D coordinates for each lidar observation
        @param: y occupancy observed by lidar
        @param: i frame i
        @param: cell_max_min (tuple of 6 ints) — bounding area observed

        @returns: []

        Plots a single frame (i) of regression. Shows 2D and 3D representations
        of LIDAR hits, mean and variance
        """
        time1 = time.time()
        specs = [[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}], [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]]
        titles = ['3D Lidar Hits', '3D Mean Prediction', '3D Variance Prediction', '2D Lidar Hits', '2D Mean Prediction', '2D Variance Prediction']
        fig = make_subplots(
            rows=2,
            cols=3,
            specs=specs,
            subplot_titles=titles,
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
        )
        self.plot_hits_surface(X, fig)
        self.plot_mean_var(meanVarPlot, fig, cell_max_min)
        camera = dict(
            eye=dict(x=1.25, y=-1.25, z=1.25)
        )
        fig.layout.scene1.camera = camera
        fig.layout.scene2.camera = camera
        fig.layout.scene3.camera = camera
        fig.update_layout(title='{}_regression_frame{}'.format(self.plot_title, i))
        fig.update_xaxes(matches='x', title='x')
        fig.update_yaxes(matches='y', title='y')

        ###===###
        if not os.path.isdir("./plots/regression"):
            os.makedirs("./plots/regression")
        ###///###

        plotly.offline.plot(fig, filename=os.path.abspath('./plots/regression/{}_frame{}.html'.format(self.plot_title, i)), auto_open=False)#True)
        print('Completed plotting in %2f s' % (time.time()-time1))

    def _old_filter_predictions_velocity(self, ploti, surface_threshold):
        """
        @param ploti: (Xq, yq, var) (3D location and occupancy prediction)
        @return toPlot: array of (Xq, yq) filtered to show only occupancy surface
            with yq greater than 'surface_threshold'
        """
        if self.occupancy_plot_type == 'volumetric':
            return ploti

        # print("ploti[0].shape:", ploti[0].shape)
        # print("ploti[1].shape:", ploti[1].shape)
        # print("ploti[0]:", ploti[0])
        # print("ploti[1]:", ploti[1])

        joined = torch.cat((ploti[0].unsqueeze(0), ploti[1].unsqueeze(-1)), dim=-1)

        # print("self.surface_threshold:", self.surface_threshold)
        # print("ploti[0].shape:", ploti[0].shape)
        # print("ploti[1].shape:", ploti[1].shape)
        # print("ploti[0].unsqueeze(0).shape:", ploti[0].unsqueeze(0).shape)
        # print("ploti[1].unsqueeze(-1).shape:", ploti[1].unsqueeze(-1).shape)
        # print("joined.shape:", joined.shape)
        # exit()
        mask = (joined[:,3] >= surface_threshold) #+ (torch.rand_like(joined[:,3]) > self.plot_denoise)
        # mask = (joined[:,3] >= self.surface_threshold) #+ (torch.rand_like(joined[:,3]) > self.plot_denoise)
        filtered = joined[mask, :]
        # print("mask:", mask)
        # print()
        return filtered[:,:3], filtered[:,3]

    def _filter_predictions_velocity(self, X, y):
        """
        :param X: Nx3 position
        :param y: N values
        :return: thresholded X, y vals
        """
        if len(self.surface_threshold) == 1:
            mask = y.squeeze() >= self.surface_threshold[0]
        else:
            min_mask = y.squeeze() >= self.surface_threshold[0]
            max_mask = y.squeeze() <= self.surface_threshold[1]
            mask = torch.logical_and(min_mask, max_mask)

        return X[mask, :], y[mask,:]

    def _plot_velocity_scatter(self, Xqs, yqs, fig, row, col, plot_args=None):
        """
        # generic method for any plot
        :param Xqs: filtered Nx3 position
        :param yqs:  filtered N values
        :param fig:
        :param row:
        :param col:print("Number of points after filtering: ", Xq_mv.shape[0])
        :param plot_args: symbol, size, opacity, cbar_x_pos, cbar_min, cbar_max
        """

        print(" Plotting row {}, col {}".format(row, col))

        # marker and colorbar arguments
        if plot_args is None:
            symbol, size, opacity, cbar_x_pos, cbar_min, cbar_max = 'square', 8, 0.2, False, yqs[:,0].min(), yqs[:,0].max()
        else:
            symbol, size, opacity, cbar_x_pos, cbar_min, cbar_max = plot_args
        if cbar_x_pos is not False:
            colorbar = dict(x=cbar_x_pos,
                            len=1,
                            y=0.5
                        )
        else:
            colorbar = dict()

        # plot
        fig.add_trace(
            go.Scatter3d(
                x=Xqs[:,0],
                y=Xqs[:,1],
                z=Xqs[:,2],
                mode='markers',
                marker=dict(
                    color=yqs[:,0],
                    colorscale="Jet",
                    cmax=cbar_max,
                    cmin=cbar_min,
                    colorbar=colorbar,
                    opacity=opacity,
                    symbol=symbol,
                    size=size
                ),
            ),
            row=row,
            col=col
        )

    def _plot_velocity_1by3(self, X, y_vy, Xq_mv, mean_y, i):
        """
        # This plot is good for radar data
        :param X: ground truth positions
        :param y_vy: ground truth y velocity
        :param Xq_mv: query X positions
        :param mean_y: predicted y velocity mean
        :param i: ith frame
        """
        print(" Plotting 1x3 subplots")

        # setup plot
        specs = [[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}],]
        titles = ["y-velocity", "y-velocity preds mean", "y-velocity preds var"]
        fig = make_subplots(
            rows=1,
            cols=3,
            specs=specs,
            subplot_titles=titles
        )

        # calc error - possible only when Xq_mv = X
        #if self.self.args.query_dist[0] <= 0 and self.self.args.query_dist[1] <= 0 and self.self.args.query_dist[2] <= 0:
        #    print(" RMSE:", torch.sqrt(torch.mean((y_vy - mean_y)**2)).item())

        # filter by surface threshold
        print(" Surface_thresh: ", self.surface_threshold)
        print(" Number of points before filtering: {}".format(Xq_mv.shape[0]))
        Xq_mv, mean_y = self._filter_predictions_velocity(Xq_mv, mean_y)
        print(" Number of points after filtering: {}".format(Xq_mv.shape[0]))

        # set colorbar
        cbar_min = min(mean_y.min().item(), y_vy.min().item())
        cbar_max = max(mean_y.max().item(), y_vy.max().item())
        # fig.update_layout(coloraxis={'colorscale':'Jet', "cmin":cbar_min, "cmax":max_c}) # global colrobar

        # plot
        # plot_args - symbol, size, opacity, cbar_x_pos, cbar_min, cbar_max
        plot_args_data =      ['circle', 1, 0.7, 0.3, cbar_min, cbar_max]
        plot_args_pred_mean = ['square', 3, 0.1, 0.6, cbar_min, cbar_max]
        # plot_args_data = ['circle', 5, 0.7, 0.3, y_vy.min().item(), y_vy.max().item()]
        # plot_args_pred_mean = ['circle', 5, 0.7, 0.6, y_vy.min().item(), y_vy.max().item()]
        print("X:", X.shape)
        self._plot_velocity_scatter(X.float(), y_vy, fig, 1, 1, plot_args_data)
        self._plot_velocity_scatter(Xq_mv.float(), mean_y.float(), fig, 1, 2, plot_args_pred_mean)

        # update camera
        camera = dict(
            eye=dict(x=2.25, y=-2.25, z=1.25)
        )
        fig.layout.scene1.camera = camera
        fig.layout.scene2.camera = camera

        # update plot settings
        layout = dict(xaxis=dict(nticks=4, range=[self.args.area_min[0], self.args.area_max[0]], ),
                      yaxis=dict(nticks=4, range=[self.args.area_min[1], self.args.area_max[1]], ),
                      zaxis=dict(nticks=4, range=[self.args.area_min[2], self.args.area_max[2]], ),
                      aspectmode="manual",
                      aspectratio=dict(x=2, y=2, z=2))
                        # width=700,
                        # margin=dict(r=20, l=10, b=10, t=10))
        fig.update_layout(scene1=layout, scene2=layout)

        fig.update_layout(title='{}_velocity_frame{}'.format(self.plot_title, i), height=500)
        filename = os.path.abspath('./plots/velocity/{}_frame{}.html'.format(self.plot_title, i))
        plotly.offline.plot(fig, filename=filename, auto_open=False)
        print(' Plot saved as ' + filename)

    def _plot_velocity_2by3(self, X, y_vx, y_vy, y_vz, Xq_mv, mean_x, mean_y, mean_z, i):
        """
        # This plot is good when 3 (or less) directional components of the velocity are available
        :param X: ground truth positions
        :param y_vx: ground truth x velocity
        :param y_vy: ground truth y velocity
        :param y_vz: ground truth z velocity
        :param Xq_mv: query X positions
        :param mean_x: predicted x velocity mean
        :param mean_y: predicted y velocity mean
        :param mean_z: predicted z velocity mean
        :param i: ith frame
        """
        print(" Plotting 2x3 subplots")

        # setup plot
        specs = [[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}],[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]]
        titles = ["x-velocity", "y-velocity", "z-velocity", "x-velocity preds mean", "y-velocity preds mean", "z-velocity preds mean",]
        fig = make_subplots(
            rows=2,
            cols=3,
            specs=specs,
            subplot_titles=titles
        )

        # calc error - possible only when Xq_mv = X
        #if self.self.args.query_dist[0] <= 0 and self.self.args.query_dist[1] <= 0 and self.self.args.query_dist[2] <= 0:
        #    print(" RMSE:", torch.sqrt(torch.mean((y_vy - mean_y)**2)).item())

        # filter by surface threshold
        print(" Surface_thresh: ", self.surface_threshold)
        print(" Number of points before filtering: {}".format(Xq_mv.shape[0]))
        Xq_mv_x, mean_x = self._filter_predictions_velocity(Xq_mv, mean_x)
        Xq_mv_y, mean_y = self._filter_predictions_velocity(Xq_mv, mean_y)
        Xq_mv_z, mean_z = self._filter_predictions_velocity(Xq_mv, mean_z)
        print(" Number of points after filtering x-plot:{}, y-plot:{}, z-plot:{}".format(Xq_mv_x.shape[0], Xq_mv_y.shape[0], Xq_mv_z.shape[0]))

        # plot
        for Xq_mv, mean, y_v, col, cbar_x_pos in [(Xq_mv_x, mean_x, y_vx, 1, 0.3), (Xq_mv_y, mean_y, y_vy, 2, 0.7), (Xq_mv_z, mean_z, y_vz, 3, 1.0)]:
            if y_v.shape[0]*mean.shape[0] > 0: #if there are filtered points and mean preds
                # set colorbar
                cbar_min = min(mean.min().item(), y_v.min().item())
                cbar_max = max(mean.max().item(), y_v.max().item())
                # fig.update_layout(coloraxis={'colorscale':'Jet', "cmin":cbar_min, "cmax":max_c}) # global colorbar

                # plot_args - symbol, size, opacity, cbar_x_pos, cbar_min, cbar_max
                plot_args_data =      ['circle', 1, 0.7, cbar_x_pos, cbar_min, cbar_max]
                plot_args_pred_mean = ['square', 1, 0.2, False, cbar_min, cbar_max]
                # plot_args_data = ['circle', 5, 0.7, 0.3, y_vy.min().item(), y_vy.max().item()]
                # plot_args_pred_mean = ['circle', 5, 0.7, 0.6, y_vy.min().item(), y_vy.max().item()]
                self._plot_velocity_scatter(X.float(), y_v, fig, 1, col, plot_args_data)
                self._plot_velocity_scatter(Xq_mv.float(), mean.float(), fig, 2, col, plot_args_pred_mean) ###$$$###


        # update camera
        camera = dict(
            eye=dict(x=2.25, y=-2.25, z=1.25)
        )
        fig.layout.scene1.camera = camera
        fig.layout.scene2.camera = camera ###$$$###
        fig.layout.scene3.camera = camera
        fig.layout.scene4.camera = camera
        fig.layout.scene5.camera = camera
        fig.layout.scene6.camera = camera

        # update plot settings
        layout = dict(xaxis=dict(nticks=4, range=[self.args.area_min[0], self.args.area_max[0]], ),
             yaxis=dict(nticks=4, range=[self.args.area_min[1], self.args.area_max[1]], ),
             zaxis=dict(nticks=4, range=[self.args.area_min[2], self.args.area_max[2]], ),
             aspectmode="manual",
             aspectratio=dict(x=2, y=2, z=2))
        fig.update_layout(scene1=layout,scene2=layout,scene3=layout,scene4=layout,scene5=layout,scene6=layout)

        fig.update_layout(title='{}_velocity_frame{}'.format(self.plot_title, i), height=800)
        filename = os.path.abspath('./plots/velocity/{}_frame{}.html'.format(self.plot_title, i))
        plotly.offline.plot(fig, filename=filename, auto_open=False)
        print(' Plot saved as '+filename)


        pdf_filename = os.path.abspath('./plots/velocity/{}_frame{}.pdf'.format(self.plot_title, i))
        fig.write_image(pdf_filename, width=1500, height=900)
        print(' Plot also saved as ' + pdf_filename)

        svg_filename = os.path.abspath('./plots/velocity/{}_frame{}.svg'.format(self.plot_title, i))
        fig.write_image(svg_filename, width=1500, height=900)
        print(' Plot also saved as ' + svg_filename)

    def plot_velocity_frame(self, X, y_vx, y_vy, y_vz, Xq_mv, mean_x, mean_y, mean_z, i):
        time1 = time.time()

        # print(torch.sum(y_vx), torch.sum(y_vy), torch.sum(y_vz))
        if torch.sum(y_vx)*torch.sum(y_vy) == 0: #1x3 plot for radar
            self._plot_velocity_1by3(X, y_vy, Xq_mv, mean_y, i)
        else: #2x3 plot for other
            self._plot_velocity_2by3(X, y_vx, y_vy, y_vz, Xq_mv, mean_x, mean_y, mean_z, i)

        print(' Total plotting time=%2f s' % (time.time()-time1))
