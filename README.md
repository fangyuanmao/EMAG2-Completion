# EMAG2 completion

**Recovering Missing Regions of Earth Magnetic Anomaly Grid data (EMAG2) Using RePaint based on Diffusion Model**

![1728467390246](image/README/1728467390246.png)

## Set up

### 1. Environment

```
pip install numpy torch blobfile tqdm pyYaml pillow    # e.g. torch 1.7.1+cu110.
```

### 2. Download pretrained model and EMAG2 data.

|        Name        | Link                                       |
| :-----------------: | ------------------------------------------ |
|  Completion results  | https://1drv.ms/f/c/2e4d56a3d20d5c20/EmXvznPFpKRMu9dRpMWMkf8BtLq1n7uEUsy8QO8FBjaR1Q?e=UoTLF8      |
| Original EMAG2 data | https://www.ncei.noaa.gov/emag-survey-page |
|  Pretrained model  | https://1drv.ms/f/c/2e4d56a3d20d5c20/Etyk9PQRhqlEmF1s1v8L6rQBURaacy-HfRvMCk7QxZBhrA?e=pOIINp  |

Place the pretrained model under `./pretrain` and original EMAG2 data under `./EMAG2`.

### 3. Run example

```
bash shell/easy_test.sh
```

## Completion method

### 1. Stage 0: Data preprocess

Download EMAG2_V3. Run the below command, you can obtain the slice of EMAG2_V3.

```shell
python scripts/preprocess.py
```

### 2. Stage 1: Global completion

```shell
bash shell/stage1.sh
```

### 3. Stage 2: Local completion

```shell
bash shell/stage2.sh
```

## FAQ

**How to apply it for other datasets?**

If you want train new model on new dataset, it is recommended to follow [guided-diffusion](https://github.com/openai/guided-diffusion) repository.

## Acknowledgements

Our code is built upon [RePaint](https://github.com/andreas128/RePaint) and [guided-diffuion](https://github.com/openai/guided-diffusion.git). We thank the authors for their excellent work.

If you have any question, feel free to contact with fymao@zju.edu.cn or fangyuanmaocs@gmail.com .
