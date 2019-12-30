#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

import meta_dataloader.TCGA

#import models.mlp, models.gcn
import numpy as np
import data.gene_graphs
import collections
import sklearn.metrics
import sklearn.model_selection
import random
from collections import OrderedDict
import pandas as pd
from torch.optim import Optimizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


print(sys.version)


# In[3]:


tasks = meta_dataloader.TCGA.TCGAMeta(download=True, 
                                      min_samples_per_class=10
                                     )
#task = tasks[113]


# In[ ]:


print(len(tasks.task_ids))


# In[198]:


for taskid in sorted(tasks.task_ids):
    print(taskid)


# In[15]:


task = meta_dataloader.TCGA.TCGATask(('PAM50Call_RNAseq', 'BRCA'))
print(task.id)
print(task._samples.shape)
print(np.asarray(task._labels).shape)
print(collections.Counter(task._labels))


# In[193]:


for task in sorted(tasks):
    print(task)
    if task.id == ('gender', 'COAD'):
        print(task.id)
        print(task._samples.shape)
        print(np.asarray(task._labels).shape)
        print(len(collections.Counter(task._labels)))


# In[347]:


def load_sets(task, valid=False):
     
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(task._samples, 
                                                                                task._labels, 
                                                                                stratify=task._labels,
                                                                                train_size=50,
                                                                                test_size=100,
                                                                                shuffle=True,
                                                                                random_state=0
                                                                                 )
    
    train_set = TensorDataset( Tensor(X_train), Tensor(y_train))
    test_set = TensorDataset( Tensor(X_test), Tensor(y_test))

    if valid:
        X_test, X_valid, y_test, y_valid = sklearn.model_selection.train_test_split(X_test, 
                                                                                y_test, 
                                                                                stratify=y_test,
                                                                                train_size=50,
                                                                                test_size=50,
                                                                                shuffle=True,
                                                                                random_state=0
                                                                               )
        valid_set = TensorDataset( Tensor(X_valid), Tensor(y_valid))
        return train_set, valid_set, test_set
    
    return train_set, test_set 


# In[5]:


from sklearn.dummy import DummyClassifier
def Majority(X_train, X_test, y_train, y_test, random_state):
    
    classifier = DummyClassifier(strategy='most_frequent', random_state=random_state)
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    accuracy = (predicted == y_test).mean()*100.
    return accuracy


# In[170]:


class LogisticRegression(torch.nn.Module):
    
    def __init__(self, seed, input_size, num_classes, learning_rate, batch_size, epochs):
        super(LogisticRegression, self).__init__()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.linear = torch.nn.Linear(input_size, num_classes)
        
        random.seed(seed)
        torch.manual_seed(seed)
    
    def forward(self, x, params=None):
        return self.linear(x)

    @property
    def criterion(self):
        return torch.nn.CrossEntropyLoss()
    
    @property
    def optimizer(self):
        return torch.optim.LBFGS(self.parameters(), lr=1)


# In[316]:


def train(model, dataset, task_id, stop_early=False):
    train_loss = []
    criterion = model.criterion
    if type(model).__name__ == "LogisticRegression":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate, weight_decay= model.weight_decay)
     
    if stop_early:
        print("")
    else:
        train_set = dataset
        valid_set = None
 
    for i in range(model.epochs):
        for batch, labels in torch.utils.data.DataLoader(train_set, batch_size=model.batch_size, shuffle=True):
            labels = torch.autograd.Variable(labels.long())
            
            def closure():
                # Forward + Backward + Optimize
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, labels)
                loss.backward()
                return loss
            
            intermediate = optimizer.step(closure)
            loss = intermediate.item()
            train_loss.append(loss)

    return model, loss


# In[174]:


def test(model, test_set):
    # Test the Model
    batch, labels = next(iter(torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)))
    model.eval()
    outputs = model(batch)
    _, predicted = torch.max(outputs.data, 1)

    predicted = predicted.numpy()
    labels = labels.numpy()
    accuracy = (predicted == labels).mean()*100.
    return accuracy


# # Logisitic Regression and Majority Experiments:

# In[201]:


avg_maj_acc, avg_lr_acc = {}, {}
weight_decay = 0.05
batch_size = 32
epochs=100
for taskid in sorted(tasks.task_ids):
    task = meta_dataloader.TCGA.TCGATask(taskid)
    input_size = task._samples.shape[1]
    num_classes = len(collections.Counter(task._labels))
    for seed in [0,1,2,3,4,5,6,7,8,9,10]:
        try:
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(task._samples, 
                                                                                        task._labels, 
                                                                                        stratify=task._labels,
                                                                                        train_size=50,
                                                                                        test_size=100,
                                                                                        shuffle=True,
                                                                                        random_state=0
                                                                                         )
            train_set = TensorDataset( Tensor(X_train), Tensor(y_train))
            test_set = TensorDataset( Tensor(X_test), Tensor(y_test))

            #-----Majority-----
            mj_acc = Majority(X_train, X_test, y_train, y_test, seed)
            print("{} Majority Accuracy is: {}" .format(task.id, mj_acc))
            if task.id in avg_maj_acc.keys():
                avg_maj_acc[task.id].append(mj_acc)
            else:
                avg_maj_acc[task.id] = [mj_acc]

            #-----Logidtic Regression-----
            lr = 1.
            LR_model = LogisticRegression(seed, input_size, num_classes,lr, batch_size, epochs)
            trained_model, train_loss = train(LR_model, train_set, task.id, False)
            lr_result = test(trained_model, test_set)
            print("{} LR Accuracy is: {}" .format(task.id, lr_result))
            if task.id in avg_lr_acc.keys():
                avg_lr_acc[task.id].append(lr_result)
            else:
                avg_lr_acc[task.id] = [lr_result]

        except:
            print("Not enough samples!")


# In[202]:


print(avg_maj_acc)


# In[203]:


print(avg_lr_acc)


# In[257]:


print(np.std(avg_lr_acc[('white_cell_count_result', 'KIRP')]))
print(np.std(avg_maj_acc[('white_cell_count_result', 'KIRP')]))


# In[252]:


def measure_mean_std(model_dict):
    mean_std_results = {}
    for task in model_dict.keys():
        mean = np.mean(model_dict[task])
        std =  np.std(model_dict[task])
        mean_std_results[task] = {'mean':mean,'std': std}
    return mean_std_results


# In[260]:


Maj_dict = measure_mean_std(avg_maj_acc)


# In[299]:


import os
path = '/Users/mandanasamiei/PycharmProjects/TCGA_Benchmark'
result_dataframe = pd.DataFrame.from_dict(Maj_dict, orient='index')
result_dataframe.to_csv(os.path.join(path,'Majority_result_10seeds.csv'))


# In[302]:


print(Maj_dict)
print(len(Maj_dict))#121


# In[ ]:


print(np.unique(Maj_dict.keys()[1]))


# In[263]:


LR_dict= measure_mean_std(avg_lr_acc)
print(LR_dict)


# In[303]:


result_dataframe = pd.DataFrame.from_dict(LR_dict, orient='index')
result_dataframe.to_csv(os.path.join(path,'LogisticRegression_result_10seeds.csv'))


# In[296]:


model_visualization(Maj_dict, list(Maj_dict.keys()), "Majority", color='#332288')


# In[295]:


model_visualization(LR_dict, list(LR_dict.keys()), "Logistic Regression", '#AA4499')


# In[363]:


import matplotlib.pyplot as plt
def model_visualization(model_dict, tasks, label="model_name", color='#332288'):
    means, stds = [], []
    for task in tasks:
        means.append(model_dict[task]['mean'])   
        stds.append(model_dict[task]['std'])
    print(means)
    width = 0.5 # the width of the bars
    colors = ['#332288', '#AA4499', '#44AA99']
    #flatten_std = [item for sublist in stds for item in sublist]
    x = np.arange(len(tasks))
    new_x = [1.5*i for i in x]
    fig, ax = plt.subplots(figsize=(50, 10))
    plt.bar(new_x, means, color=color, alpha=0.5)
    plt.xticks(new_x, tasks)
    plt.title("{} Performance".format(label))
    plt.ylabel('Accuracy')
    plt.xlabel('Tasks')
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    plt.xlim(min(new_x) - 2, max(new_x) + 2)
    plt.ylim(0, 100)
    plt.grid()
    plt.errorbar(new_x, means, stds, linestyle='None', marker='o', color='r', alpha=0.6)
    fig.tight_layout()
    plt.show()

    #plt.savefig('/{}_result_plot_with_error_bars.png'.format(label))


# In[328]:


all_visualization(Maj_dict,LR_dict)


# In[378]:


def all_visualization(Ma_dic, LR_dic, MLP_dic):
    
    Mlp_means, Mlp_std = [], []
    LR_means, LR_std = [], []
    Maj_means, Maj_std = [], []
    tasks = LR_dic.keys()
    for task in tasks:
        Maj_means.append(Ma_dic[task]['mean'])
        Maj_std.append(Ma_dic[task]['std'])
        LR_means.append(LR_dic[task]['mean'])
        LR_std.append(LR_dic[task]['std'])
        Mlp_means.append(MLP_dic[task]['mean'])
        Mlp_std.append(MLP_dic[task]['std'])
    
    x = np.arange(len(tasks))
    new_x = [1.5*i for i in x]
    fig, ax = plt.subplots(figsize=(50, 10))
    width = 0.4
    colors = ['#332288', '#AA4499', '#44AA99']
    ax.bar([p - width for p in new_x],
            Maj_means,
            width,
            alpha=0.6,
            color='#332288',
            label='Majority')
    ax.bar(new_x,
            LR_means,
            width,
            alpha=0.6,
            color='#AA4499',
            label='LR')
    ax.bar([p + width for p in new_x],
            Mlp_means,
            width,
            alpha=0.6,
            color='#44AA99',
            label='MLP')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Baseline Performances over clinical tasks')
    ax.set_xticks([p  for p in new_x])
    ax.set_xticklabels(tasks, rotation=45, ha="right")
    fig.tight_layout()

    plt.legend(['Majority', 'LR', 'MLP'], loc='upper left')
    plt.grid()
    
    # Error bars
    x_e, y_e, e = [], [], []
    for i in new_x:
        x_e.append([i-width, i, i+width])

    for i, task in enumerate(tasks):
        y_e.append([Maj_means[i], LR_means[i], Mlp_means[i]])

    for i, task in enumerate(tasks):
        e.append([Maj_std[i], LR_std[i], Mlp_std[i]])

    flatten_x = [item for sublist in x_e for item in sublist]
    flatten_y = [item for sublist in y_e for item in sublist]
    flatten_e = [item for sublist in e for item in sublist]


    plt.xlim(min(x) - 2 * width, max(x) + width * 2)
    plt.ylim(0, 100)

    plt.errorbar(flatten_x, flatten_y, flatten_e, linestyle='None', marker='o', color='r', alpha=0.4)
    plt.savefig(path + '/new_result_plot_with_error_bars.png')
    plt.show()


# In[305]:


collections.Counter(y_train)


# # MLP

# In[346]:


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, seed, input_size, num_classes, num_layers, channels, learning_rate, batch_size, epochs, patience, weight_decay):
        super(MultiLayerPerceptron, self).__init__()
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.num_layers = num_layers
        self.channels = channels
        self.output_size = num_classes
        random.seed(seed)
        torch.manual_seed(seed)
        
        nodes = []
        nodes += channels
        architecture = OrderedDict()
        for i in range(self.num_layers):
            architecture['fc' + str(i)] = nn.Linear(input_size, nodes[i])
            architecture['relu' + str(i)] = torch.nn.ReLU()
            input_size = nodes[i]

        self.features = nn.Sequential(architecture)

        self.classifier = nn.Linear(input_size, num_classes)
    
    @property
    def criterion(self):
        criterion = F.cross_entropy
        return criterion
    
    @property
    def optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def forward(self, x):
        x = x.view(-1, self.input_size)
        features = self.features(x)
        logits = self.classifier(features)
        return logits


# In[339]:


for taskid in sorted(tasks.task_ids):
    print(taskid)
    task = meta_dataloader.TCGA.TCGATask(taskid)
    print(task.id)


# In[353]:


#After hyperparameter search (in another notebook)
epochs = 250
patience = 10
batch_size = 32
lr= 0.0001 # best_lr
num_layer = 2 # best num_layer
channels = [128,64] # best channels
weight_decay = 0.0


# In[357]:


avg_mlp_acc = {}

for taskid in sorted(tasks.task_ids):
    task = meta_dataloader.TCGA.TCGATask(taskid)
    input_size = task._samples.shape[1]
    num_classes = len(collections.Counter(task._labels))
    try:
        train_set, test_set = load_sets(task, valid = False)
        for seed in range(0,10):
            MLP_model = MultiLayerPerceptron(seed, input_size, num_classes, num_layer, channels, lr, batch_size, epochs, patience, weight_decay)
            trained_model, train_loss = train(MLP_model, train_set, taskid, False)
            mlp_result = test(trained_model, test_set)
            print("\n {} MLP Accuracy is: {}" .format(taskid, mlp_result))
            if task.id in avg_mlp_acc.keys():
                avg_mlp_acc[task.id].append(mlp_result)
            else:
                avg_mlp_acc[task.id] = [mlp_result]
    except:
        print("Not enough number of samples")


# In[358]:


print(avg_mlp_acc)


# In[359]:


MLP_dict = measure_mean_std(avg_mlp_acc)


# In[360]:


import os
path = '/Users/mandanasamiei/PycharmProjects/TCGA_Benchmark'
result_dataframe = pd.DataFrame.from_dict(Maj_dict, orient='index')
result_dataframe.to_csv(os.path.join(path,'MLP_result_10seeds.csv'))


# In[361]:


print(MLP_dict)
print(len(MLP_dict))#121


# In[364]:


model_visualization(MLP_dict, list(MLP_dict.keys()), "Multi Layer Perceptron", '#44AA99')


# In[379]:


all_visualization(Maj_dict, LR_dict, MLP_dict)


# In[380]:


print(MLP_dict)


# In[ ]:





# # Model Comparison

# In[383]:


mean_mean = []
mean_std = []

Mlp_means, Mlp_std = [], []
LR_means, LR_std = [], []
Maj_means, Maj_std = [], []
tasks = Maj_dict.keys()

for task in tasks:
    Maj_means.append(Maj_dict[task]['mean'])
    Maj_std.append(Maj_dict[task]['std'])
    LR_means.append(LR_dict[task]['mean'])
    LR_std.append(LR_dict[task]['std'])
    Mlp_means.append(MLP_dict[task]['mean'])
    Mlp_std.append(MLP_dict[task]['std'])

mean_mean.append(np.mean(Maj_means))
mean_mean.append(np.mean(LR_means))
mean_mean.append(np.mean(Mlp_means))

mean_std.append(np.std(Maj_means))
mean_std.append(np.std(LR_means))
mean_std.append(np.std(Mlp_means))

print(mean_mean)
print(mean_std)


# In[394]:


barlist=plt.bar([1,2,3], mean_mean, alpha=0.7)
barlist[0].set_color('#332288')
barlist[1].set_color('#AA4499')
barlist[2].set_color('#44AA99')
plt.xticks([1, 2, 3], ['Majority','LR','MLP'])
plt.title("Model Comparison")
plt.ylabel('Model Average Performance over all tasks')
plt.xlabel('Model')
plt.ylim(0,100)
plt.errorbar([1,2,3], mean_mean, mean_std, linestyle='None', marker='o', color='r', alpha=0.7)
plt.grid()
plt.show()


# In[ ]:




