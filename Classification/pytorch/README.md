---   
<div align="center">    

# Handy Pytoch - **Classification**     

![Test](https://img.shields.io/badge/LSC-PSD-red?style=flat-square&logo=python)
  
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



