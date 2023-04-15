# CTformer

### Environments

The operating environment is shown as follows:

```python
python == 3.7
torch == 1.7.0 cpu/gpu
pytorch-lightning == 1.5.3
numpy == 1.19.2
networkx == 2.5
scipy == 1.5.2
```

You also need to install the [pytorch-lightning](https://www.pytorchlightning.ai/):

```python
pip install pytorch-lightning
```

### File tree

```python
-CTformer
    -dataset    # storing datasets
        -weibo_
        -aps_
    -load_data  # processing data
        -data_params.py
        -datamodule.py
        -gen_cascade.py
        -gen_dataset.py
    -model.py   # CTformer and parameters
    -train.py   # training
```



### Dataset

You can obtain the Weibo datasets from the link of DeepHawkes (https://github.com/CaoQi92/DeepHawkes), we also provide a small sample for testing (weibo_/dataset.txt).

Please put the dataset.txt under our dataset/weibo_ file and then run the gen_cascade.py and gen_dataset.py in load_data file in turn. You can also switch datasets or modify data parameters in data_params.py

```python
run gen_cascade.py
run gen_dataset.py
```

An example of a cascade: 

**cascade id** \t **origin node** \t **public time** \t **reposts** \t **repost paths**

1	1	1464710400	41	1:0 1/2:22032 1/3:30685 1/4:32169 1/5:34580 1/6:29372 1/7:16459 1/8:11292 1/9:22293 1/10:6970 1/11:5530 1/12:2822 1/13:12772 1/14:1019 1/15:3360 1/16:21422 1/17:1333 1/18:1643 1/19:1518 1/20:669 1/21:2191 1/22:207 1/23:2880 1/24:445 1/25:23626 1/26:2514 1/27:681 1/28:2038 1/29:4815 1/30:99 1/31:2329 1/32:884 1/33:243 1/34:1931 1/35:236 1/36:908 1/37:7108 1/38:1501 1/39:1287 1/40:549 1/41:376

### Train our model

You can get more parameter details in model.py. Our model depends on pytorch-lightning, and you can simply run train.py to train the model.

```python
run train.py
or python train.py --gpu_lst=[1] or not setting
```

### Cite our paper 
Xigang Sun, Jingya Zhou, Ling Liu and Zhen Wu, CasTformer: A Novel Cascade Transformer Towards Predicting Information Diffusion, 2023.
