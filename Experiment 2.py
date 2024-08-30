

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

y_outer_2[X_outer_2[:,1]>0] = 1 - y_outer_2[X_outer_2[:,1]>0]


factor_1 = 1.3
X_outer, y_outer = make_circles(n_samples=n_samples_outer, factor=factor_outer, noise=noise/factor_1, random_state=random_state)
X_outer *= factor_1

y_outer[X_outer[:,1]>0] = 1 - y_outer[X_outer[:,1]>0]




factor_0 = 0.5
X_inner, y_inner = make_circles(n_samples=n_samples_inner, factor=factor_inner, noise=noise/factor_0, random_state=random_state)
X_inner = X_inner*factor_0

y_inner[X_inner[:,1]>0] = 1 - y_inner[X_inner[:,1]>0]


# Concatenate the data points and labels
X2 = np.concatenate((X_outer_2,X_outer, X_inner))
y2 = np.concatenate((y_outer_2,y_outer, y_inner)) 


X_train, X_test, y_train, y_test = train_test_split(X2, y2, stratify=y2, random_state=0)



# Plot for the 2nd dataset
plt.scatter(X2[:, 0], X2[:, 1], c=y2, cmap=plt.cm.Paired)
plt.title('Dataset 2')
plt.xlim([-4,4])
plt.ylim([-4,4])

plt.tight_layout()
plt.show()








def gaussian_kernel(X, Y, sigma=1.0):

    # Compute pairwise squared Euclidean distances
    pairwise_distances_sq = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1, keepdims=True).T
    # print(np.shape(pairwise_distances_sq))

    # Compute the Gaussian kernel matrix
    kernel_matrix = np.exp(-pairwise_distances_sq )
    
    return kernel_matrix


def displacement_kernel(sigma):
    def displacement(X,Y):
        pairwise_distances_sq = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1, keepdims=True).T
        # print(pairwise_distances_sq)
        # Compute the Gaussian kernel matrix
        kernel_matrix = np.exp(-sigma**2*pairwise_distances_sq )
        
        kernel_matrix *= (sigma**2*pairwise_distances_sq - 1)**2

        return kernel_matrix
    return displacement

#if sigma is 0.5 then rank 2 becomes the exact same as rank 1 

def displacement_kernel_2(sigma):
    def displacement(X,Y):
        pairwise_distances_sq = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1, keepdims=True).T
        # print(pairwise_distances_sq)
        # Compute the Gaussian kernel matrix
        kernel_matrix = np.exp(-sigma**2*pairwise_distances_sq)/4
        
        kernel_matrix *= (2- sigma**2*4*pairwise_distances_sq + sigma**4*pairwise_distances_sq**2)**2

        return kernel_matrix
    return displacement


def displacement_kernel_3(sigma):
    def displacement(X,Y):
        pairwise_distances_sq = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1, keepdims=True).T
        # print(pairwise_distances_sq)
        # Compute the Gaussian kernel matrix
        kernel_matrix = np.exp(-sigma**2*pairwise_distances_sq)/36
        
        kernel_matrix *= (-6 + sigma**2*18*pairwise_distances_sq - sigma**4*9*pairwise_distances_sq**2 + sigma**6*pairwise_distances_sq**3)**2

        return kernel_matrix
    return displacement


def displacement_kernel_4(sigma):
    def displacement(X,Y):
        pairwise_distances_sq = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1, keepdims=True).T
        # print(pairwise_distances_sq)
        # Compute the Gaussian kernel matrix
        kernel_matrix = np.exp(-sigma**2*pairwise_distances_sq)/576
        
        kernel_matrix *= (24 - sigma**2*96*pairwise_distances_sq + sigma**4*72*pairwise_distances_sq**2 - sigma**6*16*pairwise_distances_sq**3 + sigma**8*pairwise_distances_sq**4)**2

        return kernel_matrix
    return displacement

def displacement_kernel_5(sigma):
    def displacement(X,Y):
        pairwise_distances_sq = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1, keepdims=True).T
        # print(pairwise_distances_sq)
        # Compute the Gaussian kernel matrix
        kernel_matrix = np.exp(-sigma**2*pairwise_distances_sq)/14400
        
        kernel_matrix *= (-120 + sigma**2*600*pairwise_distances_sq - sigma**4*600*pairwise_distances_sq**2 + sigma**6*200*pairwise_distances_sq**3 - sigma**8*25*pairwise_distances_sq**4 + sigma**10*1*pairwise_distances_sq**5)**2

        return kernel_matrix
    return displacement

def displacement_kernel_6(sigma):
    def displacement(X,Y):
        pairwise_distances_sq = np.sum(X**2, axis=1, keepdims=True) - 2 * np.dot(X, Y.T) + np.sum(Y**2, axis=1, keepdims=True).T
        # print(pairwise_distances_sq)
        # Compute the Gaussian kernel matrix
        kernel_matrix = np.exp(-sigma**2*pairwise_distances_sq)/518400
        
        kernel_matrix *= (720 - sigma**2*4320*pairwise_distances_sq + sigma**4*5400*pairwise_distances_sq**2 - sigma**6*2400*pairwise_distances_sq**3 + sigma**8*450*pairwise_distances_sq**4 - sigma**10*36*pairwise_distances_sq**5+ sigma**12*1*pairwise_distances_sq**6)**2

        return kernel_matrix
    return displacement



svm_classifier_custom_kernel_1 = SVC(kernel=displacement_kernel(sigma=1))
svm_classifier_custom_kernel_2 = SVC(kernel=displacement_kernel_2(sigma = 1))
svm_classifier_custom_kernel_3 = SVC(kernel=displacement_kernel_3(sigma = 1))
svm_classifier_custom_kernel_gaussian_untuned = SVC(kernel='rbf')
svm_classifier_custom_kernel_gaussian = SVC(kernel='rbf',C = 6.16193,gamma = 63.96356998)
svm_classifier_custom_kernel_4 = SVC(kernel=displacement_kernel_4(sigma = 1))
svm_classifier_custom_kernel_sigmoid = SVC(kernel='sigmoid')

# Fit the classifiers
svm_classifier_custom_kernel_1.fit(X_train, y_train)
svm_classifier_custom_kernel_2.fit(X_train, y_train)
svm_classifier_custom_kernel_3.fit(X_train, y_train)
svm_classifier_custom_kernel_gaussian.fit(X_train, y_train)
svm_classifier_custom_kernel_gaussian_untuned.fit(X_train, y_train)
svm_classifier_custom_kernel_4.fit(X_train, y_train)
svm_classifier_custom_kernel_sigmoid.fit(X_train, y_train)


y_pred_custom_kernel_1 = svm_classifier_custom_kernel_1.predict(X_test)
y_pred_custom_kernel_2 = svm_classifier_custom_kernel_2.predict(X_test)
y_pred_custom_kernel_3 = svm_classifier_custom_kernel_3.predict(X_test)
y_pred_custom_kernel_gaussian = svm_classifier_custom_kernel_gaussian.predict(X_test)
y_pred_custom_kernel_gaussian_untuned = svm_classifier_custom_kernel_gaussian_untuned.predict(X_test)
y_pred_custom_kernel_4 = svm_classifier_custom_kernel_4.predict(X_test)
y_pred_custom_kernel_sig = svm_classifier_custom_kernel_sigmoid.predict(X_test)


print("Classification Accuracy with 1 Kernel:", accuracy_score(y_test, y_pred_custom_kernel_1))
print("Classification Accuracy with 2 Kernel:", accuracy_score(y_test, y_pred_custom_kernel_2))
print("Classification Accuracy with 3 Kernel:", accuracy_score(y_test, y_pred_custom_kernel_3))
print("Classification Accuracy with G untuned Kernel:", accuracy_score(y_test, y_pred_custom_kernel_gaussian_untuned))
print("Classification Accuracy with G Kernel:", accuracy_score(y_test, y_pred_custom_kernel_gaussian))
print("Classification Accuracy with 4 Kernel:", accuracy_score(y_test, y_pred_custom_kernel_4))
print("Classification Accuracy with S Kernel:", accuracy_score(y_test, y_pred_custom_kernel_sig))




import os

# Prepare meshgrid for plotting
xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 200), np.linspace(-2.5, 2.5, 200))

# Define the classifiers, names, and accuracies
kerns = [svm_classifier_custom_kernel_2, svm_classifier_custom_kernel_3, svm_classifier_custom_kernel_4, svm_classifier_custom_kernel_gaussian_untuned, svm_classifier_custom_kernel_gaussian]
names = ['n=2', 'n=3', 'n=4', 'Gaussian (Untuned)', 'Gaussian (Tuned)']
accs = [accuracy_score(y_test, y_pred_custom_kernel_2), accuracy_score(y_test, y_pred_custom_kernel_3), accuracy_score(y_test, y_pred_custom_kernel_4), accuracy_score(y_test, y_pred_custom_kernel_gaussian_untuned), accuracy_score(y_test, y_pred_custom_kernel_gaussian)]
import os


# Create a directory named "Experiment_1" in the same folder as the Python script
save_dir = os.path.join(os.path.dirname(__file__), "Experiment_2")

# Make the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Plot decision boundaries for each classifier individually and save them
for classifier, kernel_name, acc in zip(kerns, names, accs):
    fig, ax = plt.subplots(figsize=(10, 10))
    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 2), cmap=plt.cm.PuBu)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30, cmap=plt.cm.Paired, edgecolors='k')
    ax.set_title(f"{kernel_name}, Acc: {round(acc*100)}%", fontsize=28)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save the plot in the new folder
    file_name = os.path.join(save_dir, f"{kernel_name}.png")
    plt.savefig(file_name, dpi=500, bbox_inches="tight")
    plt.close(fig)
