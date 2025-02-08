# TLA (Text-Label Alignment)
Official implementation for the ECML-PKDD 2024 accepted paper "Modeling Text-Label Alignment for Hierarchical Text Classification" [arxiv](https://arxiv.org/abs/2409.00788) [ecml](https://link.springer.com/chapter/10.1007/978-3-031-70365-2_10) 
## Demo notebook
The notebook, [run_htla.ipynb](https://github.com/havelhakimi/TLA/blob/main/run_HTLA.ipynb), demonstrates how to run the scripts provided in this repository to train models using the WOS dataset.
## Requirements
- Python >= 3.6
- torch >= 1.6.0
- transformers >= 4.2.1
- Below libraries only if you want to run on GAT/GCN as the graph encoder
  - torch-geometric == 2.4.0
  - torch-sparse == 0.6.17
  - torch-scatter == 2.1.1

## Data
- All datasets are publically available and can be accessed at [WOS](https://github.com/kk7nc/HDLTex), [RCV1-V2](https://trec.nist.gov/data/reuters/reuters.html) and [NYT](https://catalog.ldc.upenn.edu/LDC2008T19). 
- We followed the specific details mentioned in the  [contrastive-htc](https://github.com/wzh9969/contrastive-htc#preprocess) repository to obtain and preprocess the original datasets (WOS, RCV1-V2, and NYT).
- After accessing the dataset, run the scripts in the folder `preprocess` for each dataset separately to obtain tokenized version of dataset and the related files. These will be added in the `data/x` folder where x is the name of dataset with possible choices as: wos, rcv and nyt.
- For reference we have added tokenized versions of the WOS dataset along with its related files in the `data/wos` folder. Similarly do for the other two datasets.

## Train
The `train.py` can be used to train all the models by setting different arguments.  

### For BERT (does flat multi-label classification) 
`python train.py --name='ckp_bert' --batch 10 --data='wos' --graph 0` </br> </br>
Some Important arguments: </br>
- `--name` name of directory in which your model will be saved. For e.g. the above model will be saved in `./TLA/data/wos/ckp_bert`
- `--data` name of dataset directory which contains your data and related files
- `--graph` whether to use graph encoder

###  For HTLA (BERT+GPTrans+TLA; does Hierarchical Text Classification)
`python train.py --name='ckp_htla' --batch 10 --data='wos' --graph 1 --graph_type='GPTrans' --edge_dim 30 --tla 1 --tl_temp 0.07` </br>
</br>
Some Important arguments: </br>
- `--graph_type` type of graph encoder. Possible choices are 'GCN, 'GAT', 'graphormer' and 'GPTrans'. HTLA uses GPTrans as the graph encoder
- `--edge_dim` edge feature size for GPTrans (We use 30 as edge feature size for each dataset )
- `--tla` whether Text-Label Alignment (TLA) Loss required or not. If set to 0, the model will be optimized only on BCE loss, which we refer to as BERT-GPTrans in the paper.
- `--tl_temp` Temperature value for the TLA loss (We use 0.07 as the temp. value for all datasets)
- The node feature is fixed as 768 to match the text feature size and is not included as run time argument   

### Other arguments for TLA in train.py:
Arguments of train.py, namely `--norm`, `--proj`, and `--hsize`, are part of TLA but have not been used in this work and can be ignored.



## Test
To run the trained model on test set run the script `test.py` </br> 
`python test.py --name ckp_htla --data wos --extra _macro` </br> </br>
Some Important arguments
- `--name` name of the directory which contains the saved checkpoint. The checkpoint is saved in `../TLA/data/wos/` when working with WOS dataset
- `--data` name of dataset directory which contains your data and related files
- `--extra` two checkpoints are kept based on macro-F1 and micro-F1 respectively. The possible choices are  `_macro` and `_micro` to choose from the two checkpoints

## Citation
```bibtex
@InProceedings{10.1007/978-3-031-70365-2_10,
author="Kumar, Ashish
and Toshniwal, Durga",
editor="Bifet, Albert
and Davis, Jesse
and Krilavi{\v{c}}ius, Tomas
and Kull, Meelis
and Ntoutsi, Eirini
and {\v{Z}}liobait{\.{e}}, Indr{\.{e}}",
title="Modeling Text-Label Alignment for Hierarchical Text Classification",
booktitle="Machine Learning and Knowledge Discovery in Databases. Research Track",
year="2024",
publisher="Springer Nature Switzerland",
address="Cham",
pages="163--179",
isbn="978-3-031-70365-2"
}

