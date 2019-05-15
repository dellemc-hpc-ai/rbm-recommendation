# Recommendation System - RBM

Restricted Boltzmann Machine (RBM) is a generative learning model that is useful for collaborative filtering in recommendation system. RBM is much robust and makes accurate predictions compared to other models such Singular Value Decomposition (SVD).
In this implementation we show the parallelization of RBM model which will be helpful in large datasets as the Markov Chain Monte Carlo (MCMC) steps in the learning algorithm is computationally expensive.

## Getting Started

Clone the files to a lcoal  directory or mounted directory in case of nauta.  

  * TODO - Add requirements.txt for bare metal


## General Instructions

Ensure that the rbm-recommendation directory is appended to the PYTHONPATH correctly.

This can be done by updating the sys.path in train.py and eval.py.

```
sys.path.append('/path/to/rbm-recommendation')
```

Using Nauta:

```
sys.path.append('/mnt/output/home/rbm-recommendation')
```


## Data

The [MovieLens datasets](https://grouplens.org/datasets/movielens/) is used for training and evaluation. The full dataset consist of 27,000,000 ratings and 1,100,000 tag applications applied to 58,000 movies by 280,000 users.


## Train

Train using 4 nodes and 4 processes each.

Export the required environment variables. This needs to be be udpated in templates values.yaml in case of nauta.

```
export OMP_NUM_THREADS=9
export KMP_BLOCKTIME=0
```

Update train.py, data_dir and output_dir paths accordingly.

```
mpiexec --hostfile /path/to/hostfile --map-by ppr:4:node --oversubscribe -n 16 -x OMP_NUM_THREADS -x KMP_BLOCKTIME python /path/to/train.py --hidden=100 --epochs=1 --gbz=512 --data_dir="/path/to/data/movielens_full.csv" --output_dir="/path/to/output_dir"
```

Using Nauta:

```
nctl exp submit --name rec-sys-test -t multinode-tf-training-horovod /path/to/train.py -- --hidden 100 --epochs 1 --gbz 512 --data_dir "/path/to/data/movielens_full.csv" --output_dir "/mnt/output/experiment"
```



### Eval

Evaluate the model by loading the weights, bias hidden and bias visible files obtained from training.

```
mpiexec --hostfile /path/to/hostfile --map-by ppr:4:node --oversubscribe -n 16 -x OMP_NUM_THREADS -x KMP_BLOCKTIME python /path/to/eval.py --data_dir="/path/to/data/movielens_full.csv" --weights_file="/path/to/rbm_w_file.txt" --bias_hidden "/path/to/rbm_bh_file.txt" --bias_visible "/path/to/rbm_bv_file.txt" --output_dir="/path/to/output_dir"
```

Using Nauta:

```
nctl exp submit --name rec-sys-eval-test -t multinode-tf-training-horovod /path/to/eval.py -- --data_dir "/path/to/data/movielens_full.csv" --weights_file "/path/to/rbm_w_file.txt" --bias_hidden "/path/to/rbm_bh_file.txt" --bias_visible "/path/to/rbm_bv_file.txt" --output_dir "/mnt/output/experiment"

```


## Related articles


  * TODO - Add links to blogs once posted 

