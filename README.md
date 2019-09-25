# TCGA_Benchmark
TCGA Benchmark Tasks for Clinical Attribute Prediction based on Gene expression data 

This repo is providing Clincal Benchmark tasks drived from TCGA (The Cancer Genome Atlas Program Dataset). 

## Installation

## Starter files

To run the experiments you can run the Majority, Logisitic Regression and MLP function in `tcga_benchmark` notebook. 

## Task Definition 

Tasks are a combination of clinical features and cancer study. An example of a task would be predicting gender, age, alcohol document history, family history of stomach cancer, the code of the disease and other clinical attributes for different types of patients based on their gene expressions. There are 39 types of cancers in total but we only used 25 ones of them.  A simple example of a task is (‘gender’, BRCA) where we’re predicting gender for breast cancer patients. The list of all valid tasks (considering that the minimum number of samples is 10) is provided under result directory named `list_of_valid_tasks`

There are 121 clinical tasks. (44  clinical attributes and 25  cancer studies)

## Experiments 

We studied the performance of three supervised learning model on  all 121 clinical tasks, we evaluated the performance of each task by using the following models:

1- Majority Class <br/>
2- Logistic Regression (LR) <br/>
3- Multi-Layer Percptron (MLP) <br/>

- Experiments setting:

We split each dataset to a train set of size 50 and a test size equal to 100 samples.

We performed a hyperparameter search for Neural networks. A Neural Network with 2 hidden layers, 128 and 64 hidden nodes in each respectively seems to have the highest overall performance in compare to LR and Majority. We run the network for 250 epochs for each task and over 10 different trials. After hyperparameter search we  set the learning rate to 0.0001, batch_size to 32 and weight decay to 0.0 (no regularization effect) as there is no overfitting from the beginning of the training.  We use Adam for optimization in NN and LBFGS in LR.  

While Neural Network is dealing better with clinical tasks classification but Logistic Regression has a higher performance among Gender tasks. 
This could be an interesting hypothesis that this difference in LR and MLP can actually be used to judge whether few or many genes influence the task.

We run the baselines over all tasks for 10 seeds and measured the average of accuracy, the following figure illustrate the result. <br/>

<p align="center">
  <img src="/results/results_all_model.png" width="1200" hight="400" title="hover text">
</p>

## How to use?

To see how you can load all the tasks, please check the tcga_benchmark.ipynb. <br/> 
 `tasks = meta_dataloader.TCGA.TCGAMeta(download=True, 
                                      min_samples_per_class=10)`

There are 121 tasks given the fact that we’re limiting the minimum number of samples per class to 10. 

The following plots show the performance of each model over all tasks and for 10 trials:


<p align="center">
  <img src="/results/model_comparison.png" width="500" title="hover text">
</p>

Here is the link to the [Poster](https://drive.google.com/file/d/1DeOSek1SF38KIzDzSqTSBSiBsE2AK-GN/view?usp=sharing).

