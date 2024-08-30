

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA

from sklearn.svm import SVC
import numpy as np
import seaborn as sns



n_samples_outer = 500
n_samples_inner = 500
factor_outer_2 = 0.9
factor_outer = 0.8
factor_inner = 0.3
noise = 0.05
random_state = 1



factor_2 = 2
X_outer_2, y_outer_2 = make_circles(n_samples=n_samples_outer, factor=factor_outer_2, noise=noise/factor_2, random_state=random_state)
X_outer_2 *= factor_2


factor_1 = 1.3
X_outer, y_outer = make_circles(n_samples=n_samples_outer, factor=factor_outer, noise=noise/factor_1, random_state=random_state)
X_outer *= factor_1


factor_0 = 0.5
X_inner, y_inner = make_circles(n_samples=n_samples_inner, factor=factor_inner, noise=noise/factor_0, random_state=random_state)
X_inner = X_inner*factor_0


# Concatenate the data points and labels
X2 = np.concatenate((X_outer_2,X_outer, X_inner))
y2 = np.concatenate((y_outer_2,y_outer, y_inner)) 


X_train, X_test, y_train, y_test = train_test_split(X2, y2, stratify=y2, random_state=0)


# Plot for the first dataset
plt.scatter(X2[:, 0], X2[:, 1], c=y2, cmap=plt.cm.Paired)
plt.title('Dataset 1')
plt.xlim([-4,4])
plt.ylim([-4,4])

plt.tight_layout()
plt.show()








def gaussian_kernel(X, Y, sigma=1.0):

    # Compute pairwise squared Euclidean distances
    pairwise_distances_sq = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1, keepdims=True).T

    # Compute the Gaussian kernel matrix
    kernel_matrix = np.exp(-pairwise_distances_sq )
    
    return kernel_matrix


def displacement_kernel(sigma):
    def displacement(X,Y):
        pairwise_distances_sq = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1, keepdims=True).T

        kernel_matrix = np.exp(-sigma**2*pairwise_distances_sq )
        
        kernel_matrix *= (sigma**2*pairwise_distances_sq - 1)**2

        return kernel_matrix
    return displacement


def displacement_kernel_2(sigma):
    def displacement(X,Y):
        pairwise_distances_sq = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1, keepdims=True).T
        
        kernel_matrix = np.exp(-sigma**2*pairwise_distances_sq)/4
        
        kernel_matrix *= (2- sigma**2*4*pairwise_distances_sq + sigma**4*pairwise_distances_sq**2)**2

        return kernel_matrix
    return displacement


def displacement_kernel_3(sigma):
    def displacement(X,Y):
        pairwise_distances_sq = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1, keepdims=True).T

        kernel_matrix = np.exp(-sigma**2*pairwise_distances_sq)/36
        
        kernel_matrix *= (-6 + sigma**2*18*pairwise_distances_sq - sigma**4*9*pairwise_distances_sq**2 + sigma**6*pairwise_distances_sq**3)**2

        return kernel_matrix
    return displacement


def displacement_kernel_4(sigma):
    def displacement(X,Y):
        pairwise_distances_sq = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1, keepdims=True).T

        kernel_matrix = np.exp(-sigma**2*pairwise_distances_sq)/576
        
        kernel_matrix *= (24 - sigma**2*96*pairwise_distances_sq + sigma**4*72*pairwise_distances_sq**2 - sigma**6*16*pairwise_distances_sq**3 + sigma**8*pairwise_distances_sq**4)**2

        return kernel_matrix
    return displacement

def displacement_kernel_5(sigma):
    def displacement(X,Y):
        pairwise_distances_sq = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1, keepdims=True).T

        kernel_matrix = np.exp(-sigma**2*pairwise_distances_sq)/14400
        
        kernel_matrix *= (-120 + sigma**2*600*pairwise_distances_sq - sigma**4*600*pairwise_distances_sq**2 + sigma**6*200*pairwise_distances_sq**3 - sigma**8*25*pairwise_distances_sq**4 + sigma**10*1*pairwise_distances_sq**5)**2

        return kernel_matrix
    return displacement

def displacement_kernel_6(sigma):
    def displacement(X,Y):
        pairwise_distances_sq = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1, keepdims=True).T

        kernel_matrix = np.exp(-sigma**2*pairwise_distances_sq)/518400
        
        kernel_matrix *= (720 - sigma**2*4320*pairwise_distances_sq + sigma**4*5400*pairwise_distances_sq**2 - sigma**6*2400*pairwise_distances_sq**3 + sigma**8*450*pairwise_distances_sq**4 - sigma**10*36*pairwise_distances_sq**5+ sigma**12*1*pairwise_distances_sq**6)**2

        return kernel_matrix
    return displacement



from skopt import BayesSearchCV
from skopt.space import Real
from tqdm import tqdm

# Define the search space for hyperparameters
param_space = {'C': Real(0.1, 150, prior='log-uniform'),
               'gamma': Real(0.0001, 100, prior='log-uniform')}

# Initialize BayesSearchCV
bayes_search = BayesSearchCV(
    SVC(kernel='rbf'),
    param_space,
    n_iter=1000,  # Number of iterations
    cv=5,  # Cross-validation folds
    verbose=0,
    n_jobs=-1
)

with tqdm(total=bayes_search.total_iterations) as pbar:
    # Define a function to update the progress bar after each iteration
    def callback(res):
        pbar.update(1)

    # Perform Bayesian optimization
    bayes_search.fit(X_train, y_train, callback=callback)

# Get the best parameters
print("Best parameters Gaussian:", bayes_search.best_params_)

