# Official code of SGCN-CHIP-DSIR 
## _Trajectory Prediction with Contrastive Pre-training and Social Rank Fine-Tuning_ [[Paper]](https://link.springer.com/chapter/10.1007/978-981-99-8141-0_40) 
### Published ICONIP 2023
### Chenyou Fan, Haiqi Jiang, Aimin Huang, and Junjie Hu
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](./LICENSE) 

## This is the official code of implementing SGCN-CHIP-DSIR model which is CPU/GPU compatible


## ETH-UCY pre-processed datasets
```sh
Download from https://share.weiyun.com/CVQhbRBl
```

## A quick demo -- test our pretrained model


```sh
# to test our pretrained SGCN+CHIP+DSIR on ``eth" data, i.e., (last line) (first column) of Table 1 and Table 2
python test.py -p pretrained/eth_uc1_up1

# to test our pretrained SGCN+CHIP on ``eth" data, i.e., (second to last line) (first column) of Table 1 and Table 2
python test.py -p pretrained/eth_uc1_up0
```


### Code details
```
root
 - train.py : main entry of training model, implement CHIP loss and DSIR loss
 - test.py : evaluate model with ADE/FDE and IPScore
 - model.py : implement CHIP feature product
 - tools/metrics.py : implement our proposed IPScore
 - sinkhorn_sort.py : implementing sorting for ranking
 - result/ : storing all results
 - datasets/ : ETH and UCY datasets
 - dataset_processed/ : pre-processed data
```


### Software requirement
```
pytorch 1.8+, CPU/GPU compatible
CUDA-11.8 if GPU is available
```


### To run Stage-1 pre-training
```sh
# the dataset options are eth/hotel/univ/zara1/zara2
# the best validating model is auto-saved in dir {DIR_STAGE_1}
python train.py -d eth -uc 1
```


### To run Stage-2 fine-tuning
```sh
# fill in the model dir {DIR_STAGE_1} pre-trained in stage-1
# fine-tuned model will be saved in {DIR_STAGE_2}
python train.py -d eth -uc 1 -up 1 -p ${DIR_STAGE_1}
```

### To run Stage-1/2 testing
```sh
python test.py -p ${DIR_STAGE_1} -d eth
python test.py -p ${DIR_STAGE_2} -d eth
```



