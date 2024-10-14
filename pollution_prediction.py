import os
import typing
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler  # Import StandardScaler
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process.kernels import Matern, ExpSineSquared, RBF
from sklearn.model_selection import train_test_split


# Set `EXTENDED_EVALUATION` to `True` in order to visualize predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0
n_clusters = 25

class Model(object):
    def __init__(self):
        """
        initialize the model.
        """
        self.rng = np.random.default_rng(seed=0)
        self.n_clusters = n_clusters
        self.gm = GaussianMixture(n_components=self.n_clusters, random_state=0)
        self.gp_models = []  # Will hold one GP for each cluster
        self.scaler = StandardScaler()  # Initialize StandardScaler

    def generate_predictions(self, test_coordinates: np.ndarray, test_area_flags: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        If the flag is 1, there is an asymmetric cost function (underpredictions are penalized); otherwise, return the mean.
        """
        # Scale the test coordinates
        test_coordinates_scaled = self.scaler.transform(test_coordinates)

        # Predict which cluster each test point belongs to using GMM
        test_cluster_labels = self.gm.predict(test_coordinates_scaled)

        gp_mean = np.zeros(test_coordinates.shape[0], dtype=float)
        gp_std = np.zeros(test_coordinates.shape[0], dtype=float)
        predictions = np.zeros(test_coordinates.shape[0], dtype=float)

        #For each test point, use the corresponding GP model
        for i, (test_point, cluster_id, area_flag) in enumerate(zip(test_coordinates_scaled, test_cluster_labels, test_area_flags)):
            gp = self.gp_models[cluster_id]
            mean, stddev = gp.predict(test_point.reshape(1, -1), return_std=True)

            # Extract scalar values from the arrays
            mean_scalar = mean.item()
            stddev_scalar = stddev.item()

            # Store the posterior mean and stddev
            gp_mean[i] = mean_scalar
            gp_std[i] = stddev_scalar

            if area_flag:  # If the flag is 1, apply the asymmetric cost function => want to return the prediction that, in expectation, minimizes the cost
                c1 = COST_W_UNDERPREDICT
                c2 = COST_W_NORMAL
                ratio = c1 / (c1 + c2)
                
                # Use the inverse CDF of the standard normal distribution
                phi_inv = stats.norm.ppf(ratio)  # Equivalent to Phi^{-1}(c1 / (c1 + c2)), with c1 = 50 and c2 = 1
                
                # Optimal action (a*)
                optimal_prediction = mean_scalar + stddev_scalar * phi_inv
                predictions[i] = optimal_prediction 
            else:
                # If the flag is 0, return the mean prediction
                predictions[i] = mean_scalar 

        return predictions, gp_mean, gp_std

    def train_model(self, train_targets: np.ndarray, train_coordinates: np.ndarray, train_area_flags: np.ndarray):
        """
        Fit your model on the given training data.
        Fit GMM on training coordinates, then train one GP model for each cluster.
        """

        # Scale the training coordinates
        train_coordinates_scaled = self.scaler.fit_transform(train_coordinates)

        # Fit Gaussian Mixture Model to the training coordinates (as it is too expensive to train a single GP for all the points)
        self.gm.fit(train_coordinates_scaled)
        cluster_labels = self.gm.predict(train_coordinates_scaled)
        i=0
        # Train one GP for each GMM cluster
        for cluster_id in range(self.n_clusters):
            print("training for i = ", i)
            # Select data points belonging to this cluster
            cluster_points = train_coordinates_scaled[cluster_labels == cluster_id]
            cluster_targets = train_targets[cluster_labels == cluster_id]
            
            # Define a GP with a custom powerful kernel
            kernel = 1.0 * Matern(length_scale=1.0, nu=1.5, length_scale_bounds=(1e-9, 1e9)) * \
                 ExpSineSquared(length_scale=1.0, periodicity=1.0, length_scale_bounds=(1e-9, 1e9), periodicity_bounds=(1e-9, 1e9)) + \
                 RBF(length_scale=1.0, length_scale_bounds=(1e-9, 1e9))

            # Initialize the GP with the custom kernel
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=20,  
                normalize_y=True
            )
            
            # Train the GP on the data points in this cluster
            gp.fit(cluster_points, cluster_targets)
            self.gp_models.append(gp)
            i+=1

        
def calculate_cost(ground_truth: np.ndarray, predictions: np.ndarray, area_flags: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param area_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(area_flag) for area_flag in area_flags]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


def check_within_circle(coordinate, circle_parameters):
    """
    Checks if a coordinate is inside a circle.
    :param coordinate: 2D coordinate
    :param circle_parameters: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coordinate[0] - circle_parameters[0])**2 + (coordinate[1] - circle_parameters[1])**2 < circle_parameters[2]**2

def identify_city_area_flags(grid_coordinates):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param grid_coordinates: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                    [0.79915856, 0.46147936, 0.1567626 ],
                    [0.26455561, 0.77423369, 0.10298338],
                    [0.6976312,  0.06022547, 0.04015634],
                    [0.31542835, 0.36371077, 0.17985623],
                    [0.15896958, 0.11037514, 0.07244247],
                    [0.82099323, 0.09710128, 0.08136552],
                    [0.41426299, 0.0641475,  0.04442035],
                    [0.09394051, 0.5759465,  0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755969],
                    [0.88172021, 0.2724369,  0.04483477],
                    [0.9425836,  0.6339977,  0.04979664]])
    
    area_flags = np.zeros((grid_coordinates.shape[0],))

    for i,coordinate in enumerate(grid_coordinates):
        area_flags[i] = any([check_within_circle(coordinate, circ) for circ in circles])

    return area_flags

def execute_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_grid = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    grid_area_flags = identify_city_area_flags(visualization_grid)
    
    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.generate_predictions(visualization_grid, grid_area_flags)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization for air pollution predictions')
    im = ax.imshow(predictions, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def extract_area_information(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features (2D NumPy array)
    :param test_x: Test features (2D NumPy array)
    :return: Tuple of (training features' 2D coordinates, training features' city_area flags,
        test features' 2D coordinates, test features' city_area flags)
    """
    # Assuming first two columns are coordinates (latitude, longitude or x, y)
    # and the third column is the area flag (1 for residential area, 0 for others)
    
    # Extract 2D coordinates (first two columns)
    train_coordinates = train_x[:, :2]  # First two columns
    test_coordinates = test_x[:, :2]  # First two columns

    # Extract area flags (third column)
    train_area_flags = train_x[:, 2].astype(bool)  # Third column as area flag
    test_area_flags = test_x[:, 2].astype(bool)  # Third column as area flag

    # Ensure shapes are consistent
    assert train_coordinates.shape[0] == train_area_flags.shape[0]
    assert test_coordinates.shape[0] == test_area_flags.shape[0]
    assert train_coordinates.shape[1] == 2
    assert test_coordinates.shape[1] == 2
    assert train_area_flags.ndim == 1
    assert test_area_flags.ndim == 1

    return train_coordinates, train_area_flags, test_coordinates, test_area_flags

# Load the training dateset and test features
def main():
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information
    train_coordinates, train_area_flags, test_coordinates, test_area_flags = extract_area_information(train_x, test_x)
    #train_coordinates = train_coordinates[:1000]
    #train_area_flags = train_area_flags[:1000]
    #train_y = train_y[:1000]

    #do the split to have a validation set
    train_coordinates, val_coordinates, train_area_flags, val_area_flags, train_y, val_y = train_test_split(train_coordinates, train_area_flags, train_y, test_size=0.05, random_state=42)
    # Fit the model

    print('Training model')
    model = Model()
    model.train_model(train_y, train_coordinates, train_area_flags)

    # Predict on the validation features and calculate the cost
    print('Predicting on validation features')
    predictions, gp_mean, gp_std = model.generate_predictions(val_coordinates, val_area_flags)
    cost = calculate_cost(val_y, predictions, val_area_flags)
    print(f'Validation cost: {cost}')

    # Predict on the test features
    print('Predicting on test features')
    predictions = model.generate_predictions(test_coordinates, test_area_flags)
    print(predictions)

    if EXTENDED_EVALUATION:
        execute_extended_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()
