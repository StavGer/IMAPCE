# IMAPCE
This Repository contains the Source code for the Python implementation of Informative MAnfiold Projections for Cluster Exploration. IMAPCE is an iterative dimensionality reduction method that computes informative low-dimensional data projections which highlight data separability and remove background information.
# Required packages
Before running the code, two packages need to be downloaded and imported, namely pymanopt https://github.com/pymanopt/pymanopt and contrastive PCA https://github.com/abidlabs/contrastive . Replace pymanopt's steepest_descent.py with the one provided in this repo before running IMAPCE. 
# Datasets 
BNC-2007 and UCI image segmentation datasets in rds formats can be found and downloaded from https://github.com/edahelsinki/corand .
# Baselines
Implementations of two baselines are also provided for comparing their performance with IMAPCE.
# Running the algorithms
main.py can be appropriately edited to run IMAPCE and the baselines on different datasets and scenarios.
