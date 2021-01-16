# SpaTUn: Spatio-Temporal Uncertainty

![Alt text](./plots/Regression_sample.png?raw=true "Regression Sample")
![Alt text](./plots/Occupancy_sample.png?raw=true "Classification Sample")

Description.

| Model Type | Likelihood |Examples |
| ----------- | --- |----------- |
| occupancy3d     | Bernoulli | 3D continuous occupancy maps  |
| scalarfield2d   | Gaussian, Gamma | Elevation maps         |
| scalarfield3d   | Gaussian, Gamma | Position maps, Velocity maps |
| vetorfield3d    | Gaussian | Velocity maps, acceleration maps        |
| surface3d       | Bernoulli | 3D surfaces            |

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

### occupancy3d configurations (Vivian ~~and Jason~~)

Toy3 Dataset: Simple dataset with LIDAR hits along with adjustable uncertainty (sigma). The dataset is located in datasets/toy3/toy3.csv. To run the sample, use:
```
python3 spatun.py --config toy3_occupancy
```

### scalarfield2d configurations (Lydia, ~~Jason, and Ahmed~~)

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

### scalarfield3d/vetorfield3d configurations (Kyle and Ransalu)

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

### surface3d configurations (Vivian)

Toy Dataset: Simple dataset with LIDAR hits along with adjustable uncertainty (sigma). The dataset is located in datasets/toy/toy.csv. To run the sample, use:
```
python3 spatun.py --config toy_regression
```
