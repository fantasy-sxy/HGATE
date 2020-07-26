# HGATE: Heterogeneous Graph Attention Auto-Encoders

# Paper
HGATE: Heterogeneous Graph Attention Auto-Encoders

# Citation
...

## Requirements
* tensorflow (1.14)
* sklearn
* pandas
* numpy

## Run the demo

# inductive learning
```bash
python3 inductive_weibo.py
```

# transductive learning
```bash
python3 transductive_weibo.py
```


## Data

In order to use your own data, you have to provide
* an N by N adjacency matrix (N is the number of nodes),
* an N by F node attribute feature matrix (F is the number of attributes features per node), and
* an N by E binary label matrix (E is the number of classes).

