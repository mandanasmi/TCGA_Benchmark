import meta_dataloader.TCGA
import sklearn.model_selection
import sys
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from torch import Tensor
import networkx as nx
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from genegraphconv.data.gene_graphs import StringDBGraph, HetIOGraph, FunCoupGraph, HumanNetV2Graph, GeneManiaGraph, \
    RegNetGraph

graph_initializer_list = [StringDBGraph, HetIOGraph, FunCoupGraph, HumanNetV2Graph, GeneManiaGraph, RegNetGraph]
graph_names_list = ["stringdb", "hetio", "funcoup", "humannet", "genemania", "regnet"]

graph_index = 0  # Chose a graph in the list ny its index

# for graph_index in range(6):

print("Training with the", graph_names_list[graph_index], "graph")

########################################################################################################################
# Evaluate simple classification pipeline on a specific task
########################################################################################################################

task = meta_dataloader.TCGA.TCGATask(('_EVENT', 'LUNG'))  # ('PAM50Call_RNAseq', 'BRCA'))

########################################################################################################################
# Load a graph, get laplacian and data and get a consistent column ordering
########################################################################################################################

graph = graph_initializer_list[graph_index](
    datastore="/network/home/bertinpa/Documents/gene-graph-conv/genegraphconv/data")
# /Users/paul/Desktop/user1/PycharmProjects/gene-graph-conv/genegraphconv/data
# /network/home/bertinpa/Documents/gene-graph-conv/genegraphconv/data
graph_name = graph_names_list[graph_index]

# Sets of nodes
graph_genes = list(graph.nx_graph.nodes)
dataset_genes = task.gene_ids
intersection_genes = list(set(graph_genes).intersection(dataset_genes))

# Get subggraph
subgraph = graph.nx_graph.subgraph(intersection_genes)
subgraph_genes = list(subgraph.nodes)

# Get Adjacency Matrix of the subgraph
M = torch.Tensor(np.array(nx.adjacency_matrix(subgraph).todense()))
if torch.cuda.is_available():
    M = M.cuda()

# Get matrix with the columns in the same order as Laplacian
X = pd.DataFrame(task._samples, columns=dataset_genes)[subgraph_genes].to_numpy()
y = task._labels

########################################################################################################################
# Prepare data
########################################################################################################################

# Turn it into a binary classification (all against type 2)
# y = [int(i == 2) for i in y]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,
                                                                            y,
                                                                            stratify=y,
                                                                            train_size=0.8,
                                                                            shuffle=True,
                                                                            random_state=0)


########################################################################################################################
# Define model
########################################################################################################################

# Should work in theory but does not du to instability of computation
# class MaskedLinear(torch.nn.Module):
#     def __init__(self, in_features, out_features, mask):
#         super(MaskedLinear, self).__init__()
#         self.linear = torch.nn.Linear(in_features, out_features)
#         masked_weight = self.linear.weight * mask
#         self.linear.weight = Parameter(masked_weight)  # to zero it out first
#         # self.mask = mask
#
#         # def hook(module, grad_input, grad_output):
#         #     return grad_input * mask
#         #
#         # self.register_backward_hook(hook)  # to make sure gradients won't propagate
#     #
#     # def hook(self, grad_input, grad_output):
#     #     return grad_input * self.mask
#     #
#     def forward(self, x):
#         return self.linear(x).sum(axis=1)[:, None]
#


class CustomModel(torch.nn.Module):
    """
    One fully connected masked by adj matrix and then scalar product with the vector having 1 in every component
    """
    def __init__(self, input_dim, output_dim, adjacency_matrix):
        super(CustomModel, self).__init__()
        self.weight = Parameter(torch.Tensor(input_dim, input_dim))
        self.bias = Parameter(torch.Tensor(input_dim))
        self.adj = adjacency_matrix
        self.sig = torch.nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.sig(F.linear(x, self.weight * self.adj, self.bias)).sum(axis=1)[:, None]


batch_size = 32
epochs = 10
# Lambda = 0.01
learning_rate = 0.0001

model = CustomModel(X.shape[1], X.shape[1], M)
if torch.cuda.is_available():
    model = model.cuda()
criterion = torch.nn.BCEWithLogitsLoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_set = TensorDataset(Tensor(X_train), Tensor(y_train))
test_set = TensorDataset(Tensor(X_test), Tensor(y_test))

train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=True)

########################################################################################################################
# Train
########################################################################################################################

train_loss_list = []
test_loss_list = []
test_acc_list = []
cpt = 0

for epoch in range(int(epochs)):
    print("Epoch", epoch, "over", epochs)
    # train
    for i, (data, label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        outputs = model(data)
        loss = criterion(outputs[:, 0], label)  # + Lambda * ((1 - M) * model.linear.weight)
        loss.backward()
        train_loss_list.append((cpt, loss.item()))
        cpt += 1
        optimizer.step()

    # test
    for i, (data, label) in enumerate(test_dataloader):
        # Only one batch
        print(data.shape)
        optimizer.zero_grad()
        if torch.cuda.is_available():
            data = data.cuda()
            label = label.cuda()
        outputs = model(data)
        loss = criterion(outputs[:, 0], label)
        test_loss_list.append((cpt, loss.item()))
        test_acc_list.append((cpt, (accuracy_score(label.cpu(), (outputs.detach().cpu() > 0).int()))))

train_loss_list = np.array(train_loss_list)
test_loss_list = np.array(test_loss_list)
# Save
np.save("/network/home/bertinpa/Documents/TCGA_Benchmark/results/prediction_pipeline_losses4/train_loss_list_"
        + graph_name,
        train_loss_list)
np.save("/network/home/bertinpa/Documents/TCGA_Benchmark/results/prediction_pipeline_losses4/test_loss_list_"
        + graph_name,
        test_loss_list)
np.save("/network/home/bertinpa/Documents/TCGA_Benchmark/results/prediction_pipeline_losses4/test_acc_list_"
        + graph_name,
        test_acc_list)
# /Users/paul/Desktop/user1/PycharmProjects/
# /network/home/bertinpa/Documents/


# Plot
# plt.ylim(0, 2)
# plt.plot(train_loss_list[:, 0], train_loss_list[:, 1], label="train")
# plt.plot(test_loss_list[:, 0], test_loss_list[:, 1], label="test")
# plt.legend()
# plt.show()
