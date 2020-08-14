# 3D Bayesian Hilbert Maps

![Alt text](./plots/Regression_sample.png?raw=true "Regression Sample")
![Alt text](./plots/Occupancy_sample.png?raw=true "Classification Sample")

Through Bayesian Hilbert mapping (BHM), we are able to make fast occupancy predictions of our environment. Through research that we conducted last quarter, we have seen this approach work well in the 3D case when predicting occupancy based off of LIDAR observations. As a next step, we utilize Bayesian Hilbert maps to perform regression, which allows us to generate mean and variance mappings using simulated real environments such as AirSim. To best consolidate and visualize this information, we utilize interactive 3D plotting frameworks to showcase our predictions. In this repository, we implement 3D Bayesian Hilbert Maps along with a Convolutional and a Wasserstein RBF Kernel. Also included is a modular training and plotting class which summarizes regression and classification results on datasets formatted with schema: (t,X,Y,Z,occupancy,sigma_x, sigma_y, sigma_z).

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
  git clone https://github.com/JasonZheng20/Jason_BHM_3D.git
  ```

## Example Configurations

The BHM training and plotting module provides a way to quickly save and load experiment configurations via the configs folder. This can be done by setting --config [config_file_name] to load a config, or --save_config [save_to_path] to save a config. For a full list of editable hyper-parameters, refer to the documentation in train.py.

### Sample Classification Configurations

Toy3 Dataset: Simple dataset with LIDAR hits along with adjustable uncertainty (sigma). The dataset is located in datasets/toy3/toy3.csv. To run the sample, use:
```
python3 train.py --config toy3_occupancy
```

### Sample Regression Configurations

Toy Dataset: Simple dataset with LIDAR hits along with adjustable uncertainty (sigma). The dataset is located in datasets/toy/toy.csv. To run the sample, use:
```
python3 train.py --config toy_regression
```

Toy2 Dataset: Simple dataset with LIDAR hits along with adjustable uncertainty (sigma). The dataset is located in datasets/toy2/toy2.csv. To run the sample, use:
```
python3 train.py --config toy2_regression
```

Toy3 Dataset: Simple dataset with LIDAR hits along with adjustable uncertainty (sigma). The dataset is located in datasets/toy3/toy3.csv. To run the sample, use:
```
python3 train.py --config toy3_regression
```
