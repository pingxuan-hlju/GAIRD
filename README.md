# GAIRD

## Introduction
This project is an implementation of Graph reasoning method based on affinity identification and representation decoupling for predicting lncRNA-disease associations (GAIRD).

GAIRD designed homogeneous and heterogeneous distribution learning modules to combine information from different neighborhood scopes, and a representation decoupling strategy was established to distinguish the contributions of node attributes and network topology to the lncRNA-disease association prediction task.

## Citation
If you find this repository useful, please consider citing our paper:
```
@inproceedings{
xuan2022prcs,
title={Graph reasoning method based on affinity identification and representation decoupling for predicting lncRNA-disease associations},
author={Shuai Wang, Hui Cui, Tiangang Zhang, Peiliang Wu, Toshiya Nakaguchi, Ping Xuan},
booktitle={Journal of Chemical Information and Modeling(under review)},
year={2023}
}
```

# catalogs

* /config: the initialization parameters for GAIRD.
* /utils: tool used, e.g. dataset splitting, etc.
* /data: dataset used in our study.
* /model: code implementation of the GAIRD algorithm.
* /output: output directory storing preprocessed features, segmented dataset, trained model, and prediction result.
* main.py: scripts for model training and testing.
* preprocessing.py: scripts for data preprocessing.

## Environment
The codes of GAIRD are implemented and tested under the following development environment:

* python == 3.6 <br>
* networkx == 2.5 <br>
* torch == 1.9.0 <br>
* numpy == 1.19.2 <br>
* scikit-learn == 1.0.2 <br>
* matplotlib == 2.2.2 <br>

## Dataset
* disease_name.txt: disease names.


* lncRNA_name.txt: lncRNA names.


* disease_similarity.txt: disease similarity matrix computed from directed acyclic graphs(DAG) between diseases.


* miRNA_similarity.txt miRNA similarity matrix obtained by similarity calculation for a set of diseases associated with two miRNAs.


* lncRNA_disease_association.txt: lncRNA-disease associations extracted from the LncRNADisease database.


* miRNA_disease_association.txt: miRNA-disease associations extracted from the HMDD database.


* lncRNA_miRNA_interaction.txt: lncRNA-miRNA interactions extracted from the starbase v2.0 database.


## How to Run the Code
1. Data Preprocessing: generating training set, test set, adjacency matrix, attribute matrix, shortest path distance matrix
    ```
    preprocessing.py
    ```
2. Train and test the model.
    ```
    main.py
    ```

The other details can be seen in the paper and the codes.
