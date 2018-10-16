# edge2evc
The open source code for our edge2vec paper.

## How to use the code

### Statistical Validation
In the folder example there is the simple_example.py for the python usage. To use it from bash:

```
$ bipnet/Validate.py -i examples/bipartite_input.net -o proj.net --side 0 --thr 0.01
```

the option --side can assume value 0 or 1. If it is zero it will projected with respect to the first column, otherwise it will projected with respect the second column. The option --thr is the significance level of the statistical validation. It is suggested to use 0.01.

The code will produce three networks proj_full.net proj_bonf.net proj_fdr.net.

#### Adjusted Wallace Index

In the folder example there is the simple_example.py for the python usage. To use it from bash:

```
$ bipnet/metric.py examples/awi_input.cfg 
```
The code will check the first column respect to the second.

### Benchmark

In the folder example there is the simple_example.py for the python usage. To use it from bash:

```
$ bipnet/Benchmark.py -i examples/config_benchmark.cfg -o bipartite.net --noise 0.7 
```

## Citations

if you use the code, please cite:

- Gao, Zheng, Gang Fu, Chunping Ouyang, Satoshi Tsutsui, Xiaozhong Liu, and Ying Ding. "edge2vec: Learning Node Representation Using Edge Semantics." arXiv preprint arXiv:1809.02269 (2018).

## License
The code is released under GNU license. 


## Contributor

* **Zheng Gao** - [gao27@indiana.edu](gao27@indiana.edu) <br />


