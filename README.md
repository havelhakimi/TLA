# TLA

## Requirements
- Python >= 3.6
- torch >= 1.6.0
- transformers >= 4.2.1
- fairseq == 0.10.0
- torch-geometric == 2.4.0
- torch-sparse == 0.6.17

## Data
- The repository contains tokenized versions of the WOS dataset in `data/wos` folder, processed using BERT tokenzier. This is obtained following the same way as in [contrastive-htc](https://github.com/wzh9969/contrastive-htc#preprocess).
- Specific details on how to obtain the original datasets (WOS, RCV1-V2 and NYT) and the corresponding scripts to preprocess them are mentioned in [contrastive-htc](https://github.com/wzh9969/contrastive-htc#preprocess). They will be added here later on.

## Train
The `train.py` can be used to train all the models by setting different arguments.  

### For BERT (does flat multi-label classification) 
`python train.py --name='ckp_bert' --batch 10 --data='wos' --graph 0` </br> </br>
Some Important arguments: </br>
- `--name` name of directory in which your model will be saved. For e.g. the above model will be saved in `./HTLA/data/wos/ckp_bert`
- `--data` name of dataset directory which contains your data and related files
- `--graph` whether to use graph encoder

###  For HTLA (does Hierarchical Text Classification)
`python train.py --name='ckp_htla' --batch 10 --data='wos' --graph 1 --graph_type='GPA' --edge_dim 30 --tla 1 --tl_temp 0.07` </br>
</br>
Some Important arguments: </br>
- `--graph_type` type of graph encoder. Possible choices are 'GCN,'GAT', 'graphormer', 'GPA'. HTLA uses GPA as the graph encoder
- `--edge_dim` edge feature size for GPA (We use 30 as edge feature size for each dataset )
- `--tla` whether Text-Label Alignment Loss required or not. If set to 0, the model will be optimized only on BCE loss, which we refer to as BERT-GPTrans in the paper.
- `--tl_temp` Tempertaure value for the TLA loss (We use 0.07 as the temp value for all datasets)
- The node feature is fixed as 768 to match the text feature size and is not included as run time argument   
### For multiple  random runs
In `train.py` set the `--seed=None` for multiple random runs
### Other arguments for TLA in train.py:
Argumnets of `train.py` which are `--norm`, `--proj`, and `--hsize` are part of TLA but have not been used in this work and can be ignored. 



## Test
To run the trained model on test set run the script `test.py` </br> 
`python test.py --name ckp_htla --data wos --extra _macro` </br> </br>
Some Important arguments
- `--name` name of the directory which contains the saved checkpoint. The checkpoint is saved in `../HTLA/data/wos/` when working with WOS dataset
- `--data` name of dataset directory which contains your data and related files
- `--extra` two checkpoints are kept based on macro-F1 and micro-F1 respectively. The possible choices are  `_macro` and `_micro` to choose from the two checkpoints

