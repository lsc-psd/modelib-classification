---   
<div align="center">    

# Handy Pytoch - **Classification**     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
  
</div>

## Prerequesties:
- pytorch v1.4

- [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) [[doc]](https://pytorch-lightning.readthedocs.io/en/latest/)
- opencv

## Usage:
```bash
python train.py -m VGG16 -f PATH_TO_TRAINDATA
python test.py -config PATH_TO_CONFIG -ckpt PATH_TO_CHECKPOINT -tag_csv PATH_TO_TAGCSV -f PATH_TO_TESTDATA

```


#### TODO:
- Operation check for re-organized files `utils.py`, `test.py`, `train.py`



