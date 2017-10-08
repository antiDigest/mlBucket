
# coding: utf-8

# In[1]:


from sklearn.neural_network import MLPClassifier

X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]  # XOR input


# In[27]:


y = [0, 1, 1, 0]  # XOR output


# In[28]:


clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(2, 4))


# In[29]:


clf.fit(X, y)  # fit dataset on XOR input and output


# In[31]:


print clf.coefs_[0], "Weights to go from input layer to 1st layer"


# In[32]:


print clf.coefs_[1], "Weights to go from 1st layer to 2nd layer"


print clf.coefs_[2], "Weights to go from 2nd layer to output layer"


print clf.loss_, "Loss, using lbfgs here"


# In[ ]:
