# edge2evc
The open source code for our edge2vec paper.

## How to use the code

### Dataset
The dataset we offer for test is data.csv. The data contains four columns, which refer to Source ID, Target ID, Edge Type, Edge ID. And columns are seperated by comma ','.

### Run the code
There are two steps for running the code. 
First, to calculate transition matrix in heterogeneous networks. run transition.py from bash:

```
$ transition.py --input data.csv --output matrix --type_size 3 --em_iteration 5 --e_step 3 --walk-length 3 --num-walks 2
```

The output is matrix.txt which stores edge transition matrix.
Second, run edge2vec.py to the node embeddings via biased random walk. To use it from bash:

```
$ edge2vec.py --input data.csv --matrix matrix.txt --output vector.txt --dimensions 128 --walk-length 3 --num-walks 2 --p 1 --q 1
```

The output is the node embedding file vector.txt.

## Citations

if you use the code, please cite:

- Gao, Zheng, Gang Fu, Chunping Ouyang, Satoshi Tsutsui, Xiaozhong Liu, and Ying Ding. "edge2vec: Learning Node Representation Using Edge Semantics." arXiv preprint arXiv:1809.02269 (2018).

## License
The code is released under GNU license. 


## Contributor

* **Zheng Gao** - [gao27@indiana.edu](gao27@indiana.edu) <br />


