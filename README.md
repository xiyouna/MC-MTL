<div align="center">
  <h1>MC-MTL</h1>
  <p><strong>This code repository is associated with our manuscript currently under review at a computer vision journal:</strong></p>
  <p><em>“Enhancing Few-shot Image Classification via Multi-constraint Meta-Transfer Learning with Bidirectional Calibration”</em></p>
</div>

## Prerequisites
* Ubuntu 16.04
* Python 3.7
* PyTorch 1.8




## Datasets
```bash
cd datasets
bash download_miniimagenet.sh
bash download_tieredimagenet.sh
bash download_cub.sh
bash download_cifar_fs.sh

```

Code Structures
    
    MC-MTL/
    ├── datasets/
    ├── model/
    ├── scripts/
    │   ├── cifar_fs/
    │   ├── cub/
    │   ├── miniimagenet/
    │   └── tieredimagenet/
    ├── pre_train/
    │   ├── dataloader/
    │   ├── pre.py/
    │   └── pre_net.py/
    │
    train.py
    test.py
    README.md
    environment.yml
    
    
## pre-training
```bash
run pre.py
```
   
## testing
For example, we test on the miniImageNet dataset in the 5-way 1-shot setting.
```bash
bash scripts/test/miniimagenet_5wKs.sh
```

```bash
bash scripts/test/miniimagenet_5w1s.sh
```

## Training
For example, we test on the CUB dataset in the 5-way 1-shot setting:
```bash
bash scripts/train/cub_5wKs.sh
```
```bash
bash scripts/train/cub_5w1s.sh
```

## Citation

If you find this work helpful in your research, please consider citing our manuscript currently under review:

```bibtex
@article{song2025mcmtl,
  title={Enhancing Few-shot Image Classification via Multi-constraint Meta-Transfer Learning with Bidirectional Calibration},
  author={Wei Song and Xiuna Zhang},
  journal={Under review},
  year={2025},
  note={Manuscript submitted to a computer vision journal}
}
```

## acknowledgement
Our project references the codes in the following repos.
* [DeepEMD](https://github.com/icoz69/DeepEMD)
* [meta-transfer-learning](https://github.com/yaoyao-liu/meta-transfer-learning.git)
