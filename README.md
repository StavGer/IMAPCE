# IMAPCE
This repo contains the source code of work "Cluster Exploration using Informative Manifold Projections", accepted at European Conference on Artificial Intelligence (ECAI) 2024. <br>
Full Paper can be found in: https://arxiv.org/pdf/2309.14857.pdf <br>
IMAPCE is a Dimensionality Reduction method which computes low-dimensional embeddings that filter-out any unwanted prior knowledge a practitioner may have regarding the high-dimensional data structure. The computed embeddings tend to form clusters that reveal any underlying and/or previously unknown high-dimensional data structure.
# Code
`main.py` contains the code to reproduce results and plots for the datasets mentioned in the paper. <br> <br>
Assuming that you have some `input_data` you wish to compute embeddings for and some `prior_data` whose structure is unwanted and should be removed from the embeddings. Then, at first, you define the framework as: <br> <br>
`framework = Explore(model='IMAPCE',
                    dataset_name=dataset_name,
                    input_data=input_data,
                    data_labels=data_labels,
                    clustering_algorithm="DPGMM",
                    background_samples=prior_data,
                    alpha=1,
                    mu=mu,
                    seed=0)` <br>
, where `mu` should be chosen as advised in our work and `data_labels` can be provided to numerically evaluate the method. <br> <br>
If you want to compute a single set of embeddings which considers the `prior data`, after defining `framework`, run the function: `framework.run_exploration()` <br> <br>
If you want to perform the Iterative Visual Exploration of a dataset, then after defining `framework`, adjust the hypers accordingly and run the function:
`framework.Iterative_exploration(min_cluster_size=75, max_clusters_num=5, max_exploration_iterations=25)`, where `min_cluster_size` is the minimum size of acceptable cluster (clusters with less points will be considered outliers), `max_clusters_num` is the maximum number of clusters the DPGMM can use and `max_exploration_iterations` is the maximum number of iteration for the iterative exploration process. 

## Baselines
cPCA baseline can be found in https://github.com/abidlabs/contrastive <br>
ct-SNE baseline can be found in in https://bitbucket.org/ghentdatascience/ct-sne/src/master/ <br>
Fair-NeRV baseline can be found in https://github.com/wenxu-fi/Fair-NeRV
## Contact
If you have any questions, please feel free to reach out me at s.gerolymatos@liverpool.ac.uk
## Citation
If you use this repo or IMAPCE as part of your research, please cite our work with : <br>
```bibtex
@incollection{gerolymatos2024cluster, 
  title = {Cluster Exploration Using Informative Manifold Projections},
  booktitle = {ECAI 2024},
  author = {Gerolymatos, Stavros and Evangelopoulos, Xenophon and Gusev, Vladimir V. and Goulermas, John Y.},
  year = {2024},
  pages = {2011--2018},
  publisher = {IOS Press},
  doi = {10.3233/FAIA240717}
}
