# edge2evc
The open source code for our paper "edge2vec: Learning Node Representation Using Edge Semantics".

## How to use the code

### Dataset
The dataset we offer for test is data.csv. The data contains four columns, which refer to Source ID, Target ID, Edge Type, Edge ID. And columns are seperated by space ' '.

For unweighted graph, please see unweighted_graph.txt. The four columns are Source ID, Target ID, Edge Type, Edge ID. And columns are seperated by space ' '. For weighted graph, please see weighted_graph.txt. The five columns are Source ID, Target ID, Edge Type, Edge Weight, Edge ID. And columns are seperated by space ' '.

### Run the code
There are two steps for running the code. 
First, to calculate transition matrix in heterogeneous networks. run transition.py from bash:

```
$ transition.py --input data.csv --output matrix.txt --type_size 3 --em_iteration 5 --e_step 3 --walk-length 3 --num-walks 2
```

The output is matrix.txt which stores edge transition matrix.
Second, run edge2vec.py to the node embeddings via biased random walk. To use it from bash:

```
$ edge2vec.py --input data.csv --matrix matrix.txt --output vector.txt --dimensions 128 --walk-length 3 --num-walks 2 --p 1 --q 1
```

The output is the node embedding file vector.txt.
Data repository for medical dataset in the link: http://ella.ils.indiana.edu/~gao27/data_repo/edge2vec%20vector.zip or https://figshare.com/articles/edge2vec_vector_zip/8097539 (It is a re-computed version so the evaluation output may be a little bit different with the paper reported results.)
## Citations

if you use the code, please cite:

- Gao, Zheng, Gang Fu, Chunping Ouyang, Satoshi Tsutsui, Xiaozhong Liu, Jeremy Yang, Christopher Gessner et al. "edge2vec: Representation learning using edge semantics for biomedical knowledge discovery." BMC bioinformatics 20, no. 1 (2019): 306.
## License
The code is released under BSD 3-Clause License. 


## Contributor

* **Zheng Gao** - [gao27@indiana.edu](gao27@indiana.edu) <br />


