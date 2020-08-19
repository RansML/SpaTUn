"""
# 3D Bayesian Hilbert Maps with pytorch
# Ransalu Senanayake and Jason Zheng
"""
import math
import numpy as np
import torch
import statsmodels.api as sm

from kernel import rbf_kernel_conv, rbf_kernel_wasserstein, rbf_kernel


class BHM3D_PYTORCH():
    def __init__(self, gamma=0.05, grid=None, cell_resolution=(5, 5), cell_max_min=None, X=None, nIter=2, sigma=None, mu=None, sig=None, kernel_type='conv', device='cpu'):
        """
        :param gamma: RBF bandwidth
        :param grid: if there are prespecified locations to hinge the RBF
        :param cell_resolution: if 'grid' is 'None', resolution to hinge RBFs
        :param cell_max_min: if 'grid' is 'None', realm of the RBF field
        :param X: a sample of lidar locations to use when both 'grid' and 'cell_max_min' are 'None'
        """
        self.nIter = nIter
        self.rbf_kernel_type = kernel_type
        self.sigma = sigma

        if device == 'cpu':
            self.device = torch.device("cpu")
        elif device == "gpu":
            self.device = torch.device("cuda:0")

        self.gamma = torch.tensor(gamma, device=self.device)
        # Cannot have a size 2 gamma for classification
        if self.gamma.shape[0] == 2:
            self.gamma = self.gamma[0]
        if grid is not None:
            self.grid = grid
        else:
            self.grid = self.__calc_grid_auto(cell_resolution, cell_max_min, X)
        print(' Number of hinge points={}'.format(self.grid.shape[0]))

    def updateGrid(self, grid):
        self.grid = grid

    def updateMuSig(self, mu, sig):
        self.mu = mu
        self.sig = sig

    def updateEpsilon(self, epsilon):
        self.epsilon = epsilon

    def __calc_grid_auto(self, cell_resolution, max_min, X):
        """
        :param X: a sample of lidar locations
        :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
        :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max, z_min, z_max)
        :return: numpy array of size (# of RNFs, 2) with grid locations
        """
        X = X.numpy()

        if max_min is None:
            # if 'max_min' is not given, make a boundarary based on X
            # assume 'X' contains samples from the entire area
            expansion_coef = 1.2
            x_min, x_max = expansion_coef*X[:, 0].min(), expansion_coef*X[:, 0].max()
            y_min, y_max = expansion_coef*X[:, 1].min(), expansion_coef*X[:, 1].max()
        else:
            x_min, x_max = max_min[0], max_min[1]
            y_min, y_max = max_min[2], max_min[3]
            z_min, z_max = max_min[4], max_min[5]

        xx, yy, zz = torch.meshgrid(torch.arange(x_min, x_max, cell_resolution[0]), \
                             torch.arange(y_min, y_max, cell_resolution[1]), \
                             torch.arange(z_min, z_max, cell_resolution[2]))

        return torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)

    def __sparse_features(self, X, sigma, rbf_kernel_type='conv'):
        """
        :param X: inputs of size (N,3)
        :return: hinged features with intercept of size (N, # of features + 1)
        """
        if rbf_kernel_type == 'conv':
            rbf_features = rbf_kernel_conv(X, self.grid, gamma=self.gamma, sigma=sigma, device=self.device)
        elif rbf_kernel_type == 'wass':
            rbf_features = rbf_kernel_wasserstein(X, self.grid, gamma=self.gamma, sigma=sigma, device=self.device)
        else:
            rbf_features = rbf_kernel(X, self.grid, gamma=self.gamma)
        return rbf_features

    def __calc_posterior(self, X, y, epsilon, mu0, sig0):
        """
        :param X: input features
        :param y: labels
        :param epsilon: per dimension local linear parameter
        :param mu0: mean
        :param sig0: variance
        :return: new_mean, new_varaiance
        """
        logit_inv = torch.sigmoid(epsilon)
        lam = 0.5 / epsilon * (logit_inv - 0.5)

        sig = 1/(1/sig0 + 2*torch.sum( (X.t()**2)*lam, dim=1))

        mu = sig*(mu0/sig0 + torch.mm(X.t(), y - 0.5).squeeze())

        return mu, sig

    def fit(self, X, y):
        """
        :param X: raw data
        :param y: labels
        """
        X = self.__sparse_features(X, self.sigma, self.rbf_kernel_type)
        N, D = X.shape[0], X.shape[1]

        self.epsilon = torch.ones(N, dtype=torch.float32)
        if not hasattr(self, 'mu'):
            self.mu = torch.zeros(D, dtype=torch.float32)
            self.sig = 10000 * torch.ones(D, dtype=torch.float32)

        for i in range(self.nIter):
            print("  Parameter estimation: iter={}".format(i))

            # E-step
            self.mu, self.sig = self.__calc_posterior(X, y, self.epsilon, self.mu, self.sig)

            # M-step
            self.epsilon = torch.sqrt(torch.sum((X**2)*self.sig, dim=1) + (X.mm(self.mu.reshape(-1, 1))**2).squeeze())
        return self.mu, self.sig

    def predict(self, Xq):
        """
        :param Xq: raw inquery points
        :return: mean occupancy (Laplace approximation)
        """
        Xq = self.__sparse_features(Xq, None, self.rbf_kernel_type)

        mu_a = Xq.mm(self.mu.reshape(-1, 1)).squeeze()
        sig2_inv_a = torch.sum((Xq ** 2) * self.sig, dim=1)
        k = 1.0 / torch.sqrt(1 + math.pi * sig2_inv_a / 8)

        return torch.sigmoid(k*mu_a), sig2_inv_a

    def predictSampling(self, Xq, nSamples=50):
        """
        :param Xq: raw inquery points
        :param nSamples: number of samples to take the average over
        :return: sample mean and standard deviation of occupancy
        """
        Xq = self.__sparse_features(Xq, None, self.rbf_kernel_type)

        qw = torch.distributions.MultivariateNormal(self.mu, torch.diag(self.sig))
        w = qw.sample((nSamples,)).t()

        mu_a = Xq.mm(w).squeeze()
        probs = torch.sigmoid(mu_a)

        mean = torch.mean(probs, dim=1).squeeze()
        std = torch.std(probs, dim=1).squeeze()

        return mean, std


class BHM_REGRESSION_PYTORCH():
    def __init__(self, alpha, beta, gamma=0.05, grid=None, cell_resolution=(5, 5), cell_max_min=None, X=None, nIter=2, sigma=None, mu_sig=None, kernel_type='conv', device='cpu'):
        """
        :param gamma: RBF bandwidth
        :param grid: if there are prespecified locations to hinge the RBF
        :param cell_resolution: if 'grid' is 'None', resolution to hinge RBFs
        :param cell_max_min: if 'grid' is 'None', realm of the RBF field
        :param X: a sample of lidar locations to use when both 'grid' and 'cell_max_min' are 'None'
        """
        self.alpha = alpha
        self.beta = beta
        self.nIter = nIter
        self.rbf_kernel_type = kernel_type
        self.sigma = sigma

        if device == 'cpu':
            self.device = torch.device("cpu")
        elif device == "gpu":
            self.device = torch.device("cuda:0")

        # Regression cannot have gamma or sigma > dim 2
        self.gamma = torch.tensor(gamma, device=self.device)
        if self.gamma.shape[0] > 2:
            self.gamma = self.gamma[:2]

        if grid is not None:
            self.grid = grid
        else:
            self.grid = self.__calc_grid_auto(cell_resolution, cell_max_min, X)

    def updateGrid(self, grid):
        self.grid = grid

    def updateMuSig(self, mu, sig):
        self.mu = mu
        self.sig = sig

    def __calc_grid_auto(self, cell_resolution, max_min, X):
        """
        :param X: a sample of lidar locations
        :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
        :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max, z_min, z_max)
        :return: numpy array of size (# of RNFs, 2) with grid locations
        """
        X = X.numpy()

        if max_min is None:
            # if 'max_min' is not given, make a boundarary based on X
            # assume 'X' contains samples from the entire area
            expansion_coef = 1.2
            x_min, x_max = expansion_coef*X[:, 0].min(), expansion_coef*X[:, 0].max()
            y_min, y_max = expansion_coef*X[:, 1].min(), expansion_coef*X[:, 1].max()
        else:
            x_min, x_max = max_min[0], max_min[1]
            y_min, y_max = max_min[2], max_min[3]

        xx, yy = torch.meshgrid(torch.arange(x_min, x_max, cell_resolution[0]), \
                             torch.arange(y_min, y_max, cell_resolution[1]))

        return torch.stack([xx.flatten(), yy.flatten()], dim=1)

    def __sparse_features(self, X, sigma, rbf_kernel_type='conv'):
        """
        :param X: inputs of size (N,3)
        :return: hinged features with intercept of size (N, # of features + 1)
        """
        if rbf_kernel_type == 'conv':
            rbf_features = rbf_kernel_conv(X, self.grid, gamma=self.gamma, sigma=sigma, device=self.device)
        elif rbf_kernel_type == 'wass':
            rbf_features = rbf_kernel_wasserstein(X, self.grid, gamma=self.gamma, sigma=sigma, device=self.device)
        else:
            rbf_features = rbf_kernel(X, self.grid, gamma=self.gamma)
        return rbf_features

    def __calc_posterior(self, X, y):
        """
        :param X: input features
        :param y: labels
        :return: new_mean, new_varaiance
        """
        order = X.shape[1]
        theta = X.numpy()

        A = self.beta*theta.T.dot(theta) + self.alpha*np.eye((order))
        sig = np.linalg.pinv(A)
        mu = self.beta*sig.dot(theta.T.dot(y))

        return torch.tensor(mu, dtype=torch.float32), torch.tensor(sig, dtype=torch.float32)

    def fit(self, X, y):
        """
        :param X: raw data
        :param y: labels
        """
        X = self.__sparse_features(X, self.sigma, self.rbf_kernel_type)
        N, D = X.shape[0], X.shape[1]

        print("self.grid:", self.grid)
        print("X:\n", X)
        print("X.shape:", X.shape)

        self.mu, self.sig = self.__calc_posterior(X, y)
        return self.mu, self.sig

    def predict(self, Xq):
        """
        :param Xq: raw inquery points
        :return: mean occupancy (Laplace approximation)
        """
        Xq = self.__sparse_features(Xq, None, self.rbf_kernel_type)

        print("\nXq.shape:", Xq.shape)
        print("self.mu.shape:", self.mu.shape)
        print("self.mu.reshape(-1, 1).shape:", self.mu.reshape(-1, 1).shape)

        mu_a = Xq.mm(self.mu.reshape(-1, 1)).squeeze()
        print("mu_a.shape:", mu_a.shape)
        print("self.sig.shape:", self.sig.shape)
        sig2_inv_a = 1/self.beta + Xq.mm(self.sig).mm(Xq.t())
        print("sig2_inv_a.shape:", sig2_inv_a.shape)
        return mu_a, sig2_inv_a


    def predictSampling(self, Xq, nSamples=50):
        """
        :param Xq: raw inquery points
        :param nSamples: number of samples to take the average over
        :return: sample mean and standard deviation of occupancy
        """
        Xq = self.__sparse_features(Xq, None, self.rbf_kernel_type)

        qw = torch.distributions.MultivariateNormal(self.mu, torch.diag(self.sig))
        w = qw.sample((nSamples,)).t()

        mu_a = Xq.mm(w).squeeze()
        probs = torch.sigmoid(mu_a)

        mean = torch.mean(probs, dim=1).squeeze()
        std = torch.std(probs, dim=1).squeeze()

        return mean, std


class BHM_VELOCITY_PYTORCH:
    def __init__(self, alpha=None, beta=None, gamma=0.05, grid=None, cell_resolution=(5, 5), cell_max_min=None, X=None, nIter=2, kernel_type='rbf', likelihood_type="gamma", device='cpu', w_hatx=None, w_haty=None, w_hatz=None):
        # super().__init__(alpha, beta, gamma, grid, cell_resolution, cell_max_min, X, nIter, sigma, mu_sig, kernel_type, device)
        self.nIter = nIter
        self.rbf_kernel_type = kernel_type
        self.likelihood_type = likelihood_type

        if device == 'cpu':
            self.device = torch.device("cpu")
        elif device == "gpu":
            self.device = torch.device("cuda:0")

        self.alpha = alpha
        self.beta = beta

        self.gamma = torch.tensor(gamma, device=self.device)
        if self.gamma.shape[0] > 2:
            self.gamma = self.gamma[:2]

        if grid is not None:
            self.grid = grid
        else:
            self.grid = self.__calc_grid_auto(cell_resolution, cell_max_min, X)

        if w_hatx is not None:
            self.w_hatx = w_hatx

        if w_haty is not None:
            self.w_haty = w_haty

        if w_hatz is not None:
            self.w_hatz = w_hatz

    def updateMuSig(self, mu_x, sig_x, mu_y, sig_y, mu_z, sig_z):
        self.mu_x = mu_x
        self.sig_x = sig_x

        self.mu_y = mu_y
        self.sig_y = sig_y

        self.mu_z = mu_z
        self.sig_z = sig_z

    def fit(self, X, y_vx, y_vy, y_vz, eps=1e-10):
        if self.likelihood_type == "gamma":
            return self.fit_gamma_likelihood(X, y_vx, y_vy, y_vz, eps)
        elif self.likelihood_type == "gaussian":
            return self.fit_gaussian_likelihood(X, y_vx, y_vy, y_vz)
        else:
            raise ValueError("Unsupported likelihood type: \"{}\"".format(self.likelihood_type))


    def fit_gamma_likelihood(self, X, y_vx, y_vy, y_vz, eps=1e-10):
        X = self.__sparse_features(X, self.rbf_kernel_type)

        all_ys = torch.cat((y_vx, y_vy, y_vz), dim=-1)
        print("all_ys:\n", all_ys)
        print("all_ys.shape:", all_ys.shape)
        # print("y_vx.shape:", y_vx.shape)
        # exit()

        # print("X:", X)

        X = X.double()
        y_vx = y_vx.double()
        y_vy = y_vy.double()
        y_vz = y_vz.double()

        X_ = X.cpu().detach().numpy()
        y_vx_ = np.log(y_vx + eps)
        y_vy_ = np.log(y_vy + eps)
        y_vz_ = np.log(y_vz + eps)

        lam = 0.1
        w_hatx_ = (np.linalg.pinv(X_.T.dot(X_) + lam*np.eye(X_.shape[1]))).dot(X_.T.dot(y_vx_))
        w_haty_ = (np.linalg.pinv(X_.T.dot(X_) + lam*np.eye(X_.shape[1]))).dot(X_.T.dot(y_vy_))
        w_hatz_ = (np.linalg.pinv(X_.T.dot(X_) + lam*np.eye(X_.shape[1]))).dot(X_.T.dot(y_vz_))

        y_vx = torch.log(y_vx + eps)
        y_vy = torch.log(y_vy + eps)
        y_vz = torch.log(y_vz + eps)
        lam = 0.1

        # print("X.t().mm(X) + lam*torch.eye(X.shape[1], dtype=torch.double):", X.t().mm(X) + lam*torch.eye(X.shape[1], dtype=torch.double))
        # print("torch.pinverse(X.t().mm(X) + lam*torch.eye(X.shape[1], dtype=torch.double)):", torch.pinverse(X.t().mm(X) + lam*torch.eye(X.shape[1], dtype=torch.double)))

        # self.w_hatx = torch.tensor(np.linalg.pinv(X.t().mm(X) + lam*torch.eye(X.shape[1], dtype=torch.double))).mm(X.t().mm(y_vx).double())
        # self.w_haty = torch.tensor(np.linalg.pinv(X.t().mm(X) + lam*torch.eye(X.shape[1], dtype=torch.double))).mm(X.t().mm(y_vy).double())
        # self.w_hatz = torch.tensor(np.linalg.pinv(X.t().mm(X) + lam*torch.eye(X.shape[1], dtype=torch.double))).mm(X.t().mm(y_vz).double())
        self.w_hatx = torch.pinverse(X.t().mm(X) + lam*torch.eye(X.shape[1], dtype=torch.double)).mm(X.t().mm(y_vx).double())
        self.w_haty = torch.pinverse(X.t().mm(X) + lam*torch.eye(X.shape[1], dtype=torch.double)).mm(X.t().mm(y_vy).double())
        self.w_hatz = torch.pinverse(X.t().mm(X) + lam*torch.eye(X.shape[1], dtype=torch.double)).mm(X.t().mm(y_vz).double())
        # self.w_hatx = torch.pinverse(X.t().mm(X) + lam*torch.eye(X.shape[1])).mm(X.t().mm(y_vx))
        # self.w_haty = torch.pinverse(X.t().mm(X) + lam*torch.eye(X.shape[1])).mm(X.t().mm(y_vy))
        # self.w_hatz = torch.pinverse(X.t().mm(X) + lam*torch.eye(X.shape[1])).mm(X.t().mm(y_vz))

        # print("self.w_hatx:", self.w_hatx)
        # print("w_hatx_:", w_hatx_)
        # print("self.w_hatx.shape:", self.w_hatx.shape)
        # print("w_hatx_.shape:", w_hatx_.shape)
        print("self.w_haty:", self.w_haty)
        print("w_haty_:", w_haty_)
        print("self.w_haty.shape:", self.w_haty.shape)
        print("w_haty_.shape:", w_haty_.shape)


        assert np.allclose(self.w_hatx.cpu().detach().numpy(), w_hatx_)# , rtol=100, atol=100)
        assert np.allclose(self.w_haty.cpu().detach().numpy(), w_haty_)#, rtol=100, atol=100)
        assert np.allclose(self.w_hatz.cpu().detach().numpy(), w_hatz_)#, rtol=100, atol=100)

    def fit_gaussian_likelihood(self, X, y_vx, y_vy, y_vz):
        """
        :param X: raw data
        :param y: labels
        """
        # X = self.__sparse_features(X, self.sigma, self.rbf_kernel_type)

        print("X.shape:", X.shape)
        print("self.grid.shape:", self.grid.shape)
        # X = X[:, :2]
        # self.grid = self.grid[:, :2]
        print("X.shape:", X.shape)
        print("self.grid.shape:", self.grid.shape)
        X = self.__sparse_features(X, self.rbf_kernel_type)

        print("X.shape:", X.shape)



        # self.mu, self.sig = self.__calc_posterior(X, y)
        self.mu_x, self.sig_x = self.__calc_posterior(X, y_vx)
        self.mu_y, self.sig_y = self.__calc_posterior(X, y_vy)
        self.mu_z, self.sig_z = self.__calc_posterior(X, y_vz)
        # print("self.mu_x:", self.mu_x)
        # exit()
        return self.mu_x, self.sig_x, self.mu_y, self.sig_y, self.mu_z, self.sig_z

    def predict(self, Xq):
        if self.likelihood_type == "gamma":
            return self.predict_gamma_likelihood(Xq)
        elif self.likelihood_type == "gaussian":
            return self.predict_gaussian_likelihood(Xq)
        else:
            raise ValueError("Unsupported likelihood type: \"{}\"".format(self.likelihood_type))

    def predict_gaussian_likelihood(self, Xq):
        """
        :param Xq: raw inquery points
        :return: mean occupancy (Laplace approximation)
        """

        Xq = Xq.float()
        print("Xq.shape:", Xq.shape)
        # Xq = Xq[:, :2]
        print("Xq.shape:", Xq.shape)
        # Xq = self.__sparse_features(Xq, None, self.rbf_kernel_type)
        Xq = self.__sparse_features(Xq, self.rbf_kernel_type)#.double()
        print("Xq.shape:", Xq.shape)
        # exit()

        # mu_a = Xq.mm(self.mu.reshape(-1, 1)).squeeze()
        # sig2_inv_a = 1/self.beta + Xq.mm(self.sig).mm(Xq.t())

        print("self.beta:", self.beta)

        print("Xq.shape:", Xq.shape)
        print("self.mu_x.shape:", self.mu_x.shape)
        print("self.mu_x.reshape(-1, 1).shape:", self.mu_x.reshape(-1, 1).shape)

        mu_a_x = Xq.mm(self.mu_x.reshape(-1, 1))#.squeeze()

        print("Here1")
        print("mu_a_x.shape:", mu_a_x.shape)

        print("self.sig_x.shape:", self.sig_x.shape)
        print("1/self.beta:", 1/self.beta)
        print("Xq.t().shape:", Xq.t().shape)

        sig2_inv_a_x = 1/self.beta + Xq.mm(self.sig_x).mm(Xq.t())

        print("Here2")

        mu_a_y = Xq.mm(self.mu_y.reshape(-1, 1))#.squeeze()
        print("Here3")
        sig2_inv_a_y = 1/self.beta + Xq.mm(self.sig_y).mm(Xq.t())
        print("Here4")

        mu_a_z = Xq.mm(self.mu_z.reshape(-1, 1))#.squeeze()
        print("Here5")
        sig2_inv_a_z = 1/self.beta + Xq.mm(self.sig_z).mm(Xq.t())

        print("Here6")
        # exit()


        return mu_a_x, sig2_inv_a_x, mu_a_y, sig2_inv_a_y, mu_a_z, sig2_inv_a_z

    def predict_gamma_likelihood(self, Xq):
        # Xq = self.__sparse_features(Xq, self.rbf_kernel_type)
        # Xq = Xq.cpu().detach().numpy()
        # return np.exp(Xq.dot(self.w_hatx)), np.exp(Xq.dot(self.w_haty)), np.exp(Xq.dot(self.w_hatz))
        Xq = self.__sparse_features(Xq, self.rbf_kernel_type).double()

        return torch.exp(Xq.mm(self.w_hatx)), torch.exp(Xq.mm(self.w_haty)), torch.exp(Xq.mm(self.w_hatz))

    def __calc_grid_auto(self, cell_resolution, max_min, X):
        """
        :param X: a sample of lidar locations
        :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
        :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max, z_min, z_max)
        :return: numpy array of size (# of RNFs, 2) with grid locations
        """
        # X = X.numpy()

        if max_min is None:
            # if 'max_min' is not given, make a boundarary based on X
            # assume 'X' contains samples from the entire area
            expansion_coef = 1.2
            x_min, x_max = expansion_coef*X[:, 0].min(), expansion_coef*X[:, 0].max()
            y_min, y_max = expansion_coef*X[:, 1].min(), expansion_coef*X[:, 1].max()
        else:
            x_min, x_max = max_min[0], max_min[1]
            y_min, y_max = max_min[2], max_min[3]
            z_min, z_max = max_min[4], max_min[5]

        xx, yy, zz = torch.meshgrid(torch.arange(x_min, x_max, cell_resolution[0]), \
                             torch.arange(y_min, y_max, cell_resolution[1]), \
                             torch.arange(z_min, z_max, cell_resolution[2]))

        return torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)


    def __sparse_features(self, X, rbf_kernel_type='conv'):
        """
        :param X: inputs of size (N,3)
        :return: hinged features with intercept of size (N, # of features + 1)
        """
        if rbf_kernel_type == 'conv':
            assert False
            # rbf_features = rbf_kernel_conv(X, self.grid, gamma=self.gamma, sigma=sigma, device=self.device)
        elif rbf_kernel_type == 'wass':
            assert False
            # rbf_features = rbf_kernel_wasserstein(X, self.grid, gamma=self.gamma, sigma=sigma, device=self.device)
        else:
            rbf_features = rbf_kernel(X, self.grid, gamma=self.gamma)
        return rbf_features

    def __calc_posterior(self, X, y):
        """
        :param X: input features
        :param y: labels
        :return: new_mean, new_varaiance
        """
        order = X.shape[1]
        theta = X.numpy()

        A = self.beta*theta.T.dot(theta) + self.alpha*np.eye((order))
        sig = np.linalg.pinv(A)
        mu = self.beta*sig.dot(theta.T.dot(y))

        return torch.tensor(mu, dtype=torch.float32), torch.tensor(sig, dtype=torch.float32) # change to double??