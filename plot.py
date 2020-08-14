import argparse
import json
import os
import pandas as pd
import plotly
import plotly.graph_objects as go
import time
import torch

from bhmtorch_cpu import BHM3D_PYTORCH
from bhmtorch_cpu import BHM_REGRESSION_PYTORCH
from plotly.subplots import make_subplots

# ==============================================================================
# BHM Plotting Class
# ==============================================================================
class BHM_PLOTTER():
    def __init__(self, args, plot_title, surface_threshold, q_resolution, occupancy_plot_type='scatter', plot_denoise=0.98):
        self.args = args
        self.plot_title = plot_title
        self.surface_threshold = surface_threshold
        self.q_resolution = q_resolution
        self.occupancy_plot_type = occupancy_plot_type
        self.plot_denoise = plot_denoise
        print('Successfully initialized plotly plotting class')

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
                        x=0.28,
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

    def plot_predictions(self, toPlot, fig):
        """
        Occupancy: Plots volumetric plot of predictions

        @param toPlot: array of (Xq, yq, vars) (3D location and occupancy prediction)
        @param fig: plotly fig
        """
        Xqs = torch.zeros((1,3))
        yqs = torch.ones(1)
        vars = torch.zeros(1)
        for ploti in toPlot:
            var = ploti[2]
            if self.surface_threshold > 0:
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
                    x=Xqs[1:,0],
                    y=Xqs[1:,1],
                    z=Xqs[1:,2],
                    isomin=0,
                    isomax=1,
                    value=yqs,
                    opacity=0.05,
                    surface_count=40,
                    colorscale="Jet",
                    opacityscale=[[0,0],[self.surface_threshold,0],[1,1]],
                    colorbar=dict(
                        x=0.65,
                        len=colorbar_len,
                        y=colorbar_y
                    ),
                    cmax=1,
                    cmin=self.surface_threshold,
                ),
                row=1,
                col=2
            )
        elif self.occupancy_plot_type == 'scatter':
            fig.add_trace(
                go.Scatter3d(
                    x=Xqs[1:,0],
                    y=Xqs[1:,1],
                    z=Xqs[1:,2],
                    mode='markers',
                    marker=dict(
                        color=yqs,
                        colorscale = "Jet",
                        cmax=yqs.max().item(),
                        cmin=yqs.min().item(),
                        colorbar=dict(
                            x=0.65,
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
                x=Xqs[1:,0],
                y=Xqs[1:,1],
                z=Xqs[1:,2],
                mode='markers',
                marker=dict(
                    color=vars,
                    colorscale = "Jet",
                    cmax=vars.max().item(),
                    cmin=vars.min().item(),
                    colorbar=dict(
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
        x = torch.arange(cell_max_min[0], cell_max_min[1]+self.q_resolution[0], self.q_resolution[0])
        y = torch.arange(cell_max_min[2], cell_max_min[3]+self.q_resolution[1], self.q_resolution[1])
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
        specs = [[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]]
        titles = ['Lidar Hits', 'Occupancy Prediction', 'Variance']
        fig = make_subplots(
            rows=1,
            cols=3,
            specs=specs,
            subplot_titles=titles
        )
        self.plot_lidar_hits(X, y, fig)
        self.plot_predictions(toPlot, fig)
        camera = dict(
            eye=dict(x=2.25, y=-2.25, z=1.25)
        )
        fig.layout.scene1.camera = camera
        fig.layout.scene2.camera = camera
        fig.layout.scene3.camera = camera
        fig.update_layout(title='{}_occupancy_frame{}'.format(self.plot_title, i), height=800)
        plotly.offline.plot(fig, filename=os.path.abspath('./plots/occupancy/{}_frame{}.html'.format(self.plot_title, i)), auto_open=True)
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

    def filter_predictions_velocity(self, ploti, surface_threshold):
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

    def plot_velocity_stuff(self, toPlot, fig, surface_threshold, row, col):
        """
        Occupancy: Plots volumetric plot of predictions

        @param toPlot: array of (Xq, yq, vars) (3D location and occupancy prediction)
        @param fig: plotly fig
        """

        print("{}, {}: {}".format(row, col, surface_threshold))

        Xqs = torch.zeros((1,3))
        yqs = torch.ones(1)
        # vars = torch.zeros(1)
        for ploti in toPlot:
            # # var = ploti[2]
            # print("ploti[1]:", ploti[1])
            if surface_threshold is not None:
                ploti = self.filter_predictions_velocity(ploti, surface_threshold)
                # print("Surface threshold is not None")
            else:
                ploti = (ploti[0].unsqueeze(0), ploti[1])
            Xq, yq = ploti[0], ploti[1]
            Xqs = torch.cat((Xqs, Xq), dim=0)
            yqs = torch.cat((yqs, yq), dim=0)

        yqs = yqs[1:]
        print('Num points plotted after filtering: {}'.format(yqs.shape[0]))

        colorbar_len = 1
        colorbar_y = 0.5


        fig.add_trace(
            go.Scatter3d(
                x=Xqs[1:,0],
                y=Xqs[1:,1],
                z=Xqs[1:,2],
                mode='markers',
                marker=dict(
                    color=yqs,
                    coloraxis="coloraxis",
                    opacity=0.1,
                    symbol='square',
                ),
            ),
            row=row,
            col=col
        )

        # fig.add_trace(
        #     go.Scatter3d(
        #         x=Xqs[1:,0],
        #         y=Xqs[1:,1],
        #         z=Xqs[1:,2],
        #         mode='markers',
        #         marker=dict(
        #             color=yqs,
        #             colorscale = "Jet",
        #             cmax=yqs.max().item(),
        #             cmin=yqs.min().item(),
        #             colorbar=dict(
        #                 x=0.65,
        #                 len=colorbar_len,
        #                 y=colorbar_y
        #             ),
        #             opacity=0.1,
        #             symbol='square',
        #         ),
        #     ),
        #     row=row,
        #     col=col
        # )

    def plot_velocity_frame(self, X, y_vx, y_vy, y_vz, Xq_mv, mean_x, mean_y, mean_z, i):
        time1 = time.time()
        # specs = [[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]]
        specs = [[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
                 [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]]

        titles = ["x-velocity", "y-velocity", "z-velocity",
                  "x-velocity preds", "y-velocity preds", "z-velocity preds"]
        titles = ["x-velocity", "y-velocity", "z-velocity"]
        # titles = ["x-velocity preds", "y-velocity preds", "z-velocity preds"]
        fig = make_subplots(
            rows=2,
            cols=3,
            specs=specs,
            subplot_titles=titles
        )

        print("Xq_mv.shape:", Xq_mv.shape)
        print("X.shape:", X.shape)
        #
        print("mean_x.shape:", mean_x.shape)
        # print("X[:, 0].shape:", X[:, 0].shape)

        print("y_vx.shape:", y_vx.shape)
        # exit()

        min_c = min(mean_x.min().item(), mean_y.min().item(), mean_z.min().item(), y_vx.min().item(), y_vy.min().item(), y_vz.min().item())
        max_c = max(mean_x.max().item(), mean_y.max().item(), mean_z.max().item(), y_vx.max().item(), y_vy.max().item(), y_vz.max().item())

        fig.update_layout(coloraxis={'colorscale':'Jet', "cmin":min_c, "cmax":max_c})

        self.plot_velocity_stuff(zip(X.float(), y_vx), fig, None, 1, 1)
        self.plot_velocity_stuff(zip(X.float(), y_vy), fig,  None, 1, 2)
        self.plot_velocity_stuff(zip(X.float(), y_vz), fig, None, 1, 3)

        # self.plot_velocity_stuff(zip(Xq_mv.float(), mean_x.float()), fig, 15, 2, 1)
        # self.plot_velocity_stuff(zip(Xq_mv.float(), mean_y.float()), fig, 15, 2, 2)
        # self.plot_velocity_stuff(zip(Xq_mv.float(), mean_z.float()), fig, None, 2, 3)
        self.plot_velocity_stuff(zip(Xq_mv.float(), mean_x.float()), fig, None if torch.all(mean_x == 0) else self.surface_threshold, 2, 1)
        self.plot_velocity_stuff(zip(Xq_mv.float(), mean_y.float()), fig, None if torch.all(mean_y == 0) else self.surface_threshold, 2, 2)
        self.plot_velocity_stuff(zip(Xq_mv.float(), mean_z.float()), fig, None if torch.all(mean_z == 0) else self.surface_threshold, 2, 3)
        # self.plot_velocity_stuff(zip(X.float(), y_vx), fig, None if torch.all(y_vx == 0) else self.surface_threshold, 1, 1)

        print("mean_x:", mean_x)
        print("mean_y:", mean_y)
        print("mean_z:", mean_z)

        # camera = dict(
        #     eye=dict(x=1.35, y=-1.35, z=0.75)
        # )
        # camera = dict(
        #     eye=dict(x=1.125, y=-1.125, z=0.625)
        # )
        camera = dict(
            eye=dict(x=2.25, y=-2.25, z=1.25)
        )

        fig.layout.scene1.camera = camera
        fig.layout.scene2.camera = camera
        fig.layout.scene3.camera = camera
        fig.layout.scene4.camera = camera
        fig.layout.scene5.camera = camera
        fig.layout.scene6.camera = camera

        fig.update_layout(scene1=dict(xaxis = dict(nticks=4, range=[self.args.area_min[0], self.args.area_max[0]],),
                                     yaxis = dict(nticks=4, range=[self.args.area_min[1], self.args.area_max[1]],),
                                     zaxis = dict(nticks=4, range=[self.args.area_min[2], self.args.area_max[2]],),
                                     aspectmode="manual",
                                     aspectratio=dict(x=2, y=2, z=1)),)
                                 # width=700,
                                 # margin=dict(r=20, l=10, b=10, t=10))

        fig.update_layout(scene2=dict(xaxis = dict(nticks=4, range=[self.args.area_min[0], self.args.area_max[0]],),
                                     yaxis = dict(nticks=4, range=[self.args.area_min[1], self.args.area_max[1]],),
                                     zaxis = dict(nticks=4, range=[self.args.area_min[2], self.args.area_max[2]],),
                                     aspectmode="manual",
                                     aspectratio=dict(x=2, y=2, z=1)),)

        fig.update_layout(scene3=dict(xaxis = dict(nticks=4, range=[self.args.area_min[0], self.args.area_max[0]],),
                                     yaxis = dict(nticks=4, range=[self.args.area_min[1], self.args.area_max[1]],),
                                     zaxis = dict(nticks=4, range=[self.args.area_min[2], self.args.area_max[2]],),
                                     aspectmode="manual",
                                     aspectratio=dict(x=2, y=2, z=1)),)

        fig.update_layout(scene4=dict(xaxis = dict(nticks=4, range=[self.args.area_min[0], self.args.area_max[0]],),
                                     yaxis = dict(nticks=4, range=[self.args.area_min[1], self.args.area_max[1]],),
                                     zaxis = dict(nticks=4, range=[self.args.area_min[2], self.args.area_max[2]],),
                                     aspectmode="manual",
                                     aspectratio=dict(x=2, y=2, z=1)),)

        fig.update_layout(scene5=dict(xaxis = dict(nticks=4, range=[self.args.area_min[0], self.args.area_max[0]],),
                                     yaxis = dict(nticks=4, range=[self.args.area_min[1], self.args.area_max[1]],),
                                     zaxis = dict(nticks=4, range=[self.args.area_min[2], self.args.area_max[2]],),
                                     aspectmode="manual",
                                     aspectratio=dict(x=2, y=2, z=1)),)

        fig.update_layout(scene6=dict(xaxis = dict(nticks=4, range=[self.args.area_min[0], self.args.area_max[0]],),
                                     yaxis = dict(nticks=4, range=[self.args.area_min[1], self.args.area_max[1]],),
                                     zaxis = dict(nticks=4, range=[self.args.area_min[2], self.args.area_max[2]],),
                                     aspectmode="manual",
                                     aspectratio=dict(x=2, y=2, z=1)),)

        fig.update_layout(title='{}_occupancy_frame{}'.format(self.plot_title, i), height=800)
        plotly.offline.plot(fig, filename=os.path.abspath('./plots/velocity/{}_frame{}.html'.format(self.plot_title, i)), auto_open=False)#True)
        print('Completed plotting in %2f s' % (time.time()-time1))
