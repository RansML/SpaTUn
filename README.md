# SpaTUn: Spatio-Temporal Uncertainty

![Alt text](./plots/Regression_sample.png?raw=true "Regression Sample")
![Alt text](./plots/Occupancy_sample.png?raw=true "Classification Sample")

Description.

### Prerequisites

  Install required frameworks:

  PyTorch 1.3.0
  ```
  $ pip3 install torch torchvision
  ```
  Plotly 4.6.0
  ```
  $ pip install plotly==4.6.0
  ```

  Clone the repository
  ```
  git clone https://github.com/RansML/SpaTUn.git
  ```

## Example Configurations

The BHM training and plotting module provides a way to quickly save and load experiment configurations via the configs folder. This can be done by setting --config [config_file_name] to load a config, or --save_config [save_to_path] to save a config. For a full list of editable hyper-parameters, refer to the documentation in train.py.

### Sample 3D Bernoulli (Classification) Configurations

Toy3 Dataset: Simple dataset with LIDAR hits along with adjustable uncertainty (sigma). The dataset is located in datasets/toy3/toy3.csv. To run the sample, use:
```
python3 spatun.py --config toy3_occupancy
```

bernoulli-surface Dataset: Dataset with samples on the mesh of a shell. The mesh is located in ply_files. To make a csv from sample hit points on the shell, run:
```
python3 make_csv.py ply_files/shell.ply datasets/bernoulli-surface/shell.csv
```
Change the parameter 'dataset_path' in the bernoulli-surface config file to the path of the outputted csv file and run:
```
python3 spatun.py --config bernoulli-surface
```
### Sample 2D Gaussian (Regression) Configurations

Toy Dataset: Simple dataset with LIDAR hits along with adjustable uncertainty (sigma). The dataset is located in datasets/toy/toy.csv. To run the sample, use:
```
python3 spatun.py --config toy_regression
```

Toy2 Dataset: Simple dataset with LIDAR hits along with adjustable uncertainty (sigma). The dataset is located in datasets/toy2/toy2.csv. To run the sample, use:
```
python3 spatun.py --config toy2_regression
```

Toy3 Dataset: Simple dataset with LIDAR hits along with adjustable uncertainty (sigma). The dataset is located in datasets/toy3/toy3.csv. To run the sample, use:
```
python3 spatun.py --config toy3_regression
```

### Sample 3D Gaussian (Regression) Configurations

Toy 3D velocty dataset in /datasets/toy_velocity/toy_velocity_1.csv:
```
python spatun.py --config toy_velocity_gauss
```

Carla radar dataset in /velocity1/radar_carla_test1_frame_250.csv:
```
python spatun.py --config velocity1_gauss
```

For Kyle and Ransalu
```
python spatun.py --config kyle_ransalu/1_toy1_vel --mode tqp
```
