# GAIRD
Graph reasoning method based on affinity identification and representation decoupling for predicting lncRNA-disease associations

# Requirements
python == 3.6 <br>
networkx == 2.5 <br>
torch == 1.9.0 <br>
numpy == 1.19.2 <br>
scikit-learn == 1.0.2 <br>
matplotlib == 2.2.2 <br>

# Dataset
>disease_name.txt
>
>lncRNA_name.txt
>
>disease_similarity.txt
>
>miRNA_similarity.txt
>
>lncRNA_disease_association.txt
> 
>miRNA_disease_association.txt
>
>lncRNA_miRNA_interaction.txt

# Usage
1. Data Preprocessing: run `preprocessing.py`
    ```
    preprocessing.py  - generating training set, test set, adjacency matrix, attribute matrix, shortest path distance matrix
    ```
2. Then, run `main.py` to train and evaluate the model.

The other details can be seen in the paper and the codes.
