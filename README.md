# TCGA_Benchmark
TCGA Benchmark Tasks for Clinical Attribute Prediction based on Genetics 

This repo is providing Clincal Benchmark tasks drived from TCGA (The Cancer Genome Atlas Program Dataset). 

## Installation

## Starter files

To run the experiments you can run the Majority, Logisitic Regression and MLP function in `tcga_benchmark` notebook. 

## Task Definition 

Tasks are a combination of clinical features and cancer study. An example of a task would be predicting gender, age, alcohol document history, family history of stomach cancer, the code of the disease and other clinical attributes for different types of patients based on their gene expressions. There are 39 types of cancers in total but we only used 25 ones of them.  A simple example of a task is (‘gender’, BRCA) where we’re predicting gender for breast cancer patients. The list of all valid tasks (considering that the minimum number of samples is 10) is provided under result directory named `list_of_valid_tasks`

There are 121 clinical tasks. (44  clinical attributes and 25  cancer studies)
## Cancer Studies
* **ACC (TCGA Adrenocortical Cancer)**: This is the case when cancer cells form in the outer layer of the adrenal gland.
* **BLCA (TCGA Bladder Cancer)**: Bladder cancer typically affects men rather than women and it often occurs in older adults. This type of cancer most frequently begins in the urothelial cells that line the inside of the bladder.
* **BRCA (TCGA Breast Cancer)**: Breast cancer starts when a group of cancer cells grow into breast tissues and invade nearby ones.
* **CESC (TCGA Cervical Cancer)**: This type of cancer occurs in the cervix when cervical cells grow abnormally and destroy other tissues and organs in the body. However, the progress of cervical cancer is usually slow and it allows for early detection and treatment. The average age of women who are diagnosed with cervical cancer is in the mid-50s. 
* **CHOL (TCGA Bile Duct Cancer)**: For a better understanding of this cancer, should one know the role of bile ducts. They move a fluid called bile from the liver and gallbladder to the small intestine where it allows a better digestion of the fats in foods. Sometimes the cells in the bile ducts change and grow abnormally. These changes can cause benign conditions or lead to bile duct cancer. Most often, bile duct cancer begins in the cells of the inner layer of the bile duct.
* **COAD (TCGA Colon Cancer)**:
    Colon cancer often occurs in older adults. It usually starts as a small, noncancerous mass of cells called polyps that form on the inside of the colon. Some of these polyps can become colon cancers over time. 
* **COADREAD (TCGA Colon and Rectal Cancer)**:
    Colon and Rectum are parts of the digestive system and the large intestine. This type of cancer is also known as colorectal cancer as these organs have the same type of tissues and there is no clear border between them. 
* **DLBC (TCGA Large B-cell Lymphoma Cancer)**: Lymphoma is a complicated cancer that begins in lymphocyte cells of the immune system. These cells are white blood cells that are responsible for fighting with infections.
* **ESCA (TCGA Esophageal Cancer)**: This cancer occurs in the esophagus which is a long, hollow tube that moves the food from the back of the throat to the stomach to be digested.
* **GBM  (TCGA Glioblastoma Cancer)**: Glioblastoma is an aggressive type of cancer that usually affects the brain cells or spinal cord. Glioblastoma is made of cells called astrocytes that support nerve cells.
* **GBMLGG (TCGA Lower-Grade Glioma \& Glioblastoma)**: Glioma is a type of cancer that develops in the glial cells of the brain. Glial cells support the brain nerve cells and provide them with the required nutrients. Tumors are classified into grades I, II, III, and IV based on standards established by the World Health Organization. TCGA studied lower-grade glioma, which consists of grades II and III. GBM or glioblastoma is classified as grade IV, which is the most aggressive one. 
* **HNSC (TCGA Head and Neck Cancer)**: Cancers of the head and neck are categorized by the area of the head or neck in which they occur. They usually start in the squamous cells that have moist and mucosal surfaces inside the head and neck such as inside the mouth, the nose, and the throat.
* **KICH (TCGA Kidney Chromophobe)**:
    Chromophobe renal cell carcinoma is a rare type of kidney cancer that forms in the cells lining the small tubules in the kidney. These small tubules help filter waste from the blood to make urine.
* **KIRP (TCGA Kidney Papillary Cell Carcinoma)**: This is the second most common type of kidney cancer and usually develops from inside the kidney's tubules. 
* **LAML (TCGA Acute Myeloid Leukemia)**: LAML is one of the most common acute leukemia cancer in North America. The average age of LAML patients is 67. LAML is a cancer of the blood and bone marrow. It's quite dangerous and should be treated quickly as it can result in death within months. 
* **LGG  (TCGA Lower Grade Glioma)**: As I have already explained in the GBMLGG type of cancer, lower-grade glioma is of grade II and III types of tumors in glial cells that support the brain nerve cells. 
* **LIHC (TCGA Liver Cancer)**: Liver cancer occurs in the cells of the liver. The liver plays an important role in the body by cleaning the blood and discarding harmful materials.
    
* **LUNG (TCGA Lung Cancer)**: There are two types of lung cancers. Non-small cell lung cancer and small cell lung cancer are based on the type of cell in which cancer starts.
    
* **LUAD (TCGA Lung Adenocarcinoma)**:
    Non-small cell lung cancer usually starts in glandular cells on the outer part of the lung. This type of cancer is called adenocarcinoma.
    
* **LUSC (TCGA Lung Squamous Cell Carcinoma)**: Non-small cell lung cancer can also start in flat, thin cells called squamous cells. This type of cancer is called squamous cell carcinoma of the lung.
    
* **MESO (TCGA Mesothelioma)**:
    Malignant mesothelioma is a type of cancer that starts in the thin layer of tissue that covers the majority of the internal organs (mesothelium).
* **OV   (TCGA Ovarian Cancer)**:
    Ovarian cancer is a type of cancer that occurs in the ovaries. The female reproductive system contains two ovaries, one on each side of the uterus. The ovaries make hormones estrogen and progesterone as well as ova.
* **PAAD (TCGA Pancreatic Cancer)**: Pancreatic cancer begins in the tissues of your pancreas. The pancreas produces enzymes that help digestion and makes hormones that help manage blood sugar.
* **PCPG (TCGA Pheochromocytoma \& Paraganglioma)**: These are rare tumors that come from the same type of tissue. Paraganglioma starts in nerve tissues in the adrenal glands and near certain blood vessels and nerves. Paragangliomas that form in the adrenal glands are called pheochromocytomas.
* **PRAD (TCGA Prostate Cancer)**: Prostate cancer is a type of cancer that occurs in the prostate. This is one of the most common cancers among men.
* **READ (TCGA Rectal Cancer)**: This cancer occurs in the rectum that is the last part of the large intestine.
* **SARC (TCGA Sarcoma)**: Soft tissue sarcoma is a rare type of cancer that occurs in the tissues that connect, support and surround other body structures. This can contain nerves, muscle, blood vessels, fat, tendons and the lining of the joints.
* **SKCM (TCGA Melanoma)**: Melanoma, the most serious type of skin cancer, develops in the cells called melanocytes that make melanin which is the pigment gives the skin its color.
* **STAD (TCGA Stomach Cancer)**: Stomach cancer usually begins in the mucus-producing cells that line the stomach.
* **TCGT (TCGA Testicular Cancer)**: Testicular cancer occurs in the testicles. The testicles release male sex hormones and sperm for reproduction.
* **THCA (TCGA Thyroid Cancer)**: Thyroid cancer occurs in the cells of the thyroid. The thyroid makes hormones that regulate blood pressure, heart rate, body temperature, and weight.
* **THYM (TCGA Thymoma)**: Thymoma and thymic carcinoma are diseases in which cancer cells form on the outside surface of the thymus. The thymus is an organ in the neck that makes T-cells for the immune system. 
* **UCEC (TCGA Endometrioid Cancer)**: Endometrial cancer occurs in the cells that form the inner layer of the uterus (endometrium) and is sometimes called uterine cancer. Other types of cancer can form in the uterus, including uterine sarcoma, but they are much less common than endometrial cancer.
* **UCS (TCGA Uterine Carcinosarcoma)**:  This type of cancer starts in the inner layer of the tissue lining the uterus, while sarcoma begins in the outer layer of muscle of the uterus.
    
* **UVM (TCGA Ocular melanomas)**: 
As explained in TCGA Melanoma, this is a type of cancer that develops in the cells that produce melanin. The eyes also have cells that make melanin and can develop melanoma. Eye melanoma is also called ocular melanoma.
## Experiments 

We studied the performance of three supervised learning model on  all 121 clinical tasks, we evaluated the performance of each task by using the following models:

1- Majority Class <br/>
2- Logistic Regression (LR) <br/>
3- Multi-Layer Percptron (MLP) <br/>

**Experiment setting**

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

Here is the link to the [Poster](https://docs.google.com/drawings/d/1xfuK7bDSC6enxo0984zZ8i2aywpw_cUqQ8NBIoFywOg/edit?usp=sharing).

