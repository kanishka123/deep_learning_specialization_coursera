import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset, sigmoid, initialize_with_zeros, propagate, optimize, predict, model


# Problem Statement: You are given a dataset("data.h5") containing:

# - a training set of m_train images labeled as cat(y=1) or non-cat(y=0)
# - a test set of m_test images labeled as cat or non-cat
# - each image is of shape(num_px, num_px, 3) where 3 is for the 3 channels(RGB). Thus, each image is square(height=num_px) and (width=num_px).
# You will build a simple image-recognition algorithm that can correctly classify pictures as cat or non-cat.
# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


# We added "_orig" at the end of image datasets(train and test) because we are going to preprocess them. After preprocessing, we will end up with train_set_x and test_set_x(the labels train_set_y and test_set_y don't need any preprocessing).

# Each line of your train_set_x_orig and test_set_x_orig is an array representing an image. You can visualize an example by running the following code. Feel free also to change the index value and re-run to see other images.


# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])

print("y = " + str(train_set_y[:, index]) + ", it's a '" +
      classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")


# START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
### END CODE HERE ###

# Exercise: Reshape the training and test data sets so that images of size(num_px, num_px, 3) are flattened into single vectors of shape(num_px  ∗∗  num_px  ∗∗  3, 1).

# A trick when you want to flatten a matrix X of shape(a, b, c, d) to a matrix X_flatten of shape(b ∗∗ c ∗∗ d, a) is to use:
# X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X

# Reshape the training and test examples

# START CODE HERE ### (≈ 2 lines of code)
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T
### END CODE HERE ###

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y.shape))
print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

# One common preprocessing step in machine learning is to center and standardize your dataset, meaning that you substract the mean of the whole numpy array from each example, and then divide each example by the standard deviation of the whole numpy array. But for picture datasets, it is simpler and more convenient and works almost as well to just divide every row of the dataset by 255 (the maximum value of a pixel channel).

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0, 2]))))

# 4.2 - Initializing parameters
w, b = initialize_with_zeros(m_train)
# print("w = " + str(w))
# print("b = " + str(b))


# 4.3 - Forward and Backward propagation
# Now that your parameters are initialized, you can do the "forward" and "backward" propagation steps for learning the parameters.
# Exercise: Implement a function propagate() that computes the cost function and its gradient.

w, b, X, Y = np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
grads, cost = propagate(w, b, X, Y)
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("cost = " + str(cost))


# 4.4 - Optimization
# Now, you want to update the parameters using gradient descent.


params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate = 0.009, print_cost = False)

print("w = " + str(params["w"]))
print("b = " + str(params["b"]))
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))


# 4.5  - Prediction 
# now we predict based on the trained model 

w = np.array([[0.1124579], [0.23106775]])
b = -0.3
X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])
print("predictions = " + str(predict(w, b, X)))


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations= 2000, learning_rate = 0.005, print_cost = True)

# you see that the model is clearly overfitting the training data. Later in this specialization you will learn how to reduce overfitting, for example by using regularization

# Example of a picture that was wrongly classified.
index = 1
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
# print("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" +
#       classes[d["Y_prediction_test"][0, index]].decode("utf-8") + "\" picture.")

plt.show()

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

# Try to increase the number of iterations in the cell above and rerun the cells. You might see that the training set accuracy goes up, but the test set accuracy goes down. This is called overfitting.


# let's see different learning rates and corresponding cost function graph with iterations. 
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()




print('done')
