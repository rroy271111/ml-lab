#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn

weight =0.7
bias = 0.3
start = 0 
end = 1
step = 0.02
X = torch.arange(start,end, step).unsqueeze(dim=1)
#print(X.size)

y = weight * X + bias

X[:10], y[:10]

#print(len(X))
#print(len(y))


# In[3]:


get_ipython().system('jupyter nbconvert --to script simpleregressionmodeldata.ipynb')


# In[4]:


# Create a train/test split
train_split=int(0.8*len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]
len(X_train), len(X_test),len(y_train), len(y_test)


# In[ ]:


import matplotlib.pyplot as plt

def plot_predictions2(train_data, train_labels, test_data, test_labels, predictions=None):
    plt.figure(figsize=(10,7))

    # Ensure data is detached from the graph and in numpy format
    train_data = train_data.detach().numpy()
    train_labels = train_labels.detach().numpy()
    test_data = test_data.detach().numpy()
    test_labels = test_labels.detach().numpy()

    # Plot the training data in blue
    plt.scatter(train_data, train_labels, color='b', s=4, label='Training Data')

    # Plot test data in green
    plt.scatter(test_data, test_labels, color='g', s=4, label='Test data')

    # Plot the predictions if provided
    if predictions is not None:
        predictions = predictions.detach().numpy()
        plt.scatter(test_data, predictions, color='r', s=4, label='Predictions')


plt.legend


# In[7]:


import matplotlib.pyplot as plt

def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):
    """Plots training data, test data and compares predictions with test data"""
    plt.figure(figsize=(10,7))

    # Convert all inputs to NumPy arrays (works for CPU or GPU tensors)
    train_data = train_data.detach().cpu().numpy()
    train_labels = train_labels.detach().cpu().numpy()
    test_data = test_data.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Test data")

    # Plot predictions if provided
    if predictions is not None:
        predictions = predictions.detach().cpu().numpy()
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={'size': 14})

# Example call
plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None)


# In[ ]:


import matplotlib.pyplot as plt

def plot_predictions1(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):
    """Plots training data, test data and compares predictions with test data"""
    plt.figure(figsize=(10,7))

    #Plot training data in blue
    plt.scatter(train_data,train_labels, c="b",s=4, label="Training data")

    #Plot test data in green
    plt.scatter(test_data, test_labels, c="g",s=4, label="Test data")

    # Are the predictions correct?
    if predictions is not None:
        plt.scatter(test_data,predictions, c="r",s=4, label="Predictions")
    plt.legend(prop={'size': 14});
plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None)

