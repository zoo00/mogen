# GEN
Gene Embedding based feedforward Neural networks

# Datasets
https://zenodo.org/record/6972738#.YvDMenZBxaQ


## The overview
GEN is a prediction model for cancer drug responses and achieves SOTA performance in cancer drug response tasks. 
GEN uses gene embedding vectors as input data, so it could increase the representative power of genes.

![New_Figure1-100](https://user-images.githubusercontent.com/31497898/183336664-5dabd29c-9b24-444b-bd3e-c9ed254df5b2.jpg)
The comparison of using gene expression data to predict values for given samples in conventional and our methods, where the workflow is divided into two main stages (set up and prediction stages), and S and g mean samples and genes, respectively. (a) is the workflow of the conventional method, where a database (e.g., COSMIC) is used to select commonly important genes for all samples in the given task, and vector based encoding models (e.g., an autoencoder) are often used for feature reduction because the input data type is limited to vectors. (b) is the workflow of our method, where individually important (over or under expressed) genes are selected as input gene sets for each sample, gene embedding vectors are used to represent input genes, gene expression values scale their gene embedding vectors, and it is possible to choose from a variety of models (e.g., a transformer encoder) because the input data type is a matrix.

## Requirements

pytorch >= 1.8.0

conda install pyg -c pyg

pip install scipy

conda install -c anaconda scikit-learn

conda install hickle

## The results
![tSNE](https://user-images.githubusercontent.com/31497898/183337464-d2933a2b-dcfa-4d3b-b186-88d91c0e2cd8.PNG)
(a), (b), (c), and (d) are the t-SNE plots of encoding vectors of GENs trained on the GDSC. In (a) and (c), all methods are more clustered than (b) and (d); especially, the cases of FC in the same gene set are most clustered.

![table1](https://user-images.githubusercontent.com/31497898/169188961-95831aca-c075-404e-a99a-eb2454cc5706.PNG)
