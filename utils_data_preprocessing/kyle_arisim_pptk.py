import pandas as pd
import numpy as np
import pptk
import sys

df0 = pd.read_csv('datasets/kyle_airsim1/formatted_data0.csv').to_numpy()
df1 = pd.read_csv('datasets/kyle_airsim1/formatted_data1.csv').to_numpy()
df2 = pd.read_csv('datasets/kyle_airsim1/formatted_data2.csv').to_numpy()
df3 = pd.read_csv('datasets/kyle_airsim1/formatted_data3.csv').to_numpy()

traj = np.concatenate((0*np.ones(df0.shape[0]), 1*np.ones(df1.shape[0]), 2*np.ones(df2.shape[0]), 3*np.ones(df3.shape[0])))
df = np.vstack((df0, df1, df2, df3))
df = np.hstack((df, traj[:,None]))

df[:,0] = 0.0
df[:,3] *= -6
df[:,1:4] /= 100
np.savetxt('datasets/kyle_airsim1/formatted_data0_4.csv', df, delimiter=',', header='t, X, Y, Z, V_X, V_Y, V_Z, traj_id', comments='')#, fmt=' '.join(['%i'] + ['%1.8f']*7))


print(np.min(df, axis=0), np.max(df, axis=0))

v0 = pptk.viewer(df[:,1:4], df[:,7], df[:,4], df[:,5], df[:,6]); v0.set(point_size=0.1)