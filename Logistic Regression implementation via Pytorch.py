#!/usr/bin/env python
# coding: utf-8

# Digit Recognizer using Logistic regression
# Implementation of CNN
# Transfer learning and fine tuning

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import TensorDataset,DataLoader,Dataset


# In[2]:


train=pd.read_csv('/Users/nischal/Desktop/Git Hub Projects/Digit Classifier using Pytorch/digit-recognizer 2/train.csv')
test=pd.read_csv('/Users/nischal/Desktop/Git Hub Projects/Digit Classifier using Pytorch/digit-recognizer 2/train.csv')


# In[3]:


train


# Preparing label--converting to numpy

# In[4]:



labels=train.pop('label')


# In[5]:


labels=labels.to_numpy()


# In[6]:


train


# In[7]:


data=train.to_numpy()


# In[8]:


data.shape,labels.shape


# Vizualization

# In[9]:


plt.figure(figsize=(10,4))
sns.countplot(labels)
plt.title('Class Distribution')


# In[10]:


for i in range(20):
    plt.subplot(5, 5, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(data[i].reshape(28,-1))
    


# Pytorch Dataset and data loader

# In[11]:


x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=0.2)


# Convert to tensor 

# In[12]:


x_train_tensor=torch.from_numpy(x_train)
y_train_tensor=torch.from_numpy(y_train).type(torch.LongTensor) # use cases requires that the target be LongTensor type and int just can not hold the target value.

x_test_tensor=torch.from_numpy(x_test)
y_test_tensor=torch.from_numpy(y_test)


# Convert to tensor dataset

# In[13]:


train_dataset=TensorDataset(x_train_tensor,y_train_tensor)
test_dataset=TensorDataset(x_test_tensor,y_test_tensor)


# In[14]:


train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)


# Train model

# In[15]:


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') #if else statement


# In[16]:


input_size=28*28
output_size=10
hidden_size=100
learning_rate=0.001
num_epochs=100

class logisticRegression(nn.Module):
    def __init__(self, input_size,output_size):
        super(logisticRegression,self).__init__()
        self.linear=nn.Linear(input_size,output_size)
        
    def forward(self,x):
        out=self.linear(x)
        return out
    
model=logisticRegression(input_size,output_size).to(device)


# In[17]:


# Cross Entropy Loss  
criterion = nn.CrossEntropyLoss()

# SGD Optimizer 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Training and Evaluation on CPU

# In[31]:


torch.FloatTensor ==1


# In[18]:


train_loader 


# In[64]:


loss_list = []
iteration_list = []
accuracy_list = []
iter_num = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        # input and label
        train = images.view(-1, 28*28)
        labels = labels
      
        # Forward pass
        outputs = model(train.type(torch.FloatTensor))
        
        # Loss calculate
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter_num += 1
        
        if (i+1) % 50 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                
                # Forward pass
                images = images.view(-1, 28*28)
                outputs = model(images.type(torch.FloatTensor))
                print(torch.max(outputs, 1))
                input()
                # Predictions
                predicted = torch.max(outputs, 1)[1]
                print(predicted)
                # Total number of samples
                total += labels.size(0)
                
                # Total correct predictions
                correct += (predicted == labels).sum()
            
            accuracy = 100 * (correct/total)
            loss_list.append(loss.data)
            iteration_list.append(iter_num)
            accuracy_list.append(accuracy)
        if (iter_num+1) % 1000 == 0:
            # Print Loss
            print('Iteration: {} Loss: {:0.4f} Val Accuracy: {:0.4f}%'.format(iter_num+1, loss.data, accuracy))


# In[33]:


# Visualize loss
plt.plot(iteration_list,loss_list)
plt.xlabel("Num. of Iters.")
plt.ylabel("Loss")
plt.title("Logistic Regression: Loss vs Num. of Iters.")
plt.show()

# Visualize loss
plt.plot(iteration_list,accuracy_list)
plt.xlabel("Num. of Iters.")
plt.ylabel("Accuracy")
plt.title("Logistic Regression: Accuracy vs Num. of Iters.")
plt.show()


# In[34]:


model()


# In[68]:


x = test_loader.dataset.tensors[0][2001].type(torch.FloatTensor)
x = x.reshape(1,784)
x.size()


# In[71]:


torch.max(model(x),1)[1]


# In[60]:


plt.imshow(x.reshape(shape =(28,28)))


# In[ ]:




