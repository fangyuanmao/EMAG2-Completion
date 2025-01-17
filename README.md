# EMAG2 completion

**Recovering Missing Regions of Earth Magnetic Anomaly Grid data (EMAG2) Using RePaint based on Diffusion Model**

![1728467390246](image/README/1728467390246.png)

## Set up

### 1. Environment

```
pip install numpy torch blobfile tqdm pyYaml pillow pandas    # e.g. torch 1.7.1+cu110.
```

### 2. Download pretrained model and EMAG2 data.

|        Name        | Note                                       |
| :-----------------: | :------------------------------------------: |
|  [Completion results](https://1drv.ms/f/c/2e4d56a3d20d5c20/EmXvznPFpKRMu9dRpMWMkf8BtLq1n7uEUsy8QO8FBjaR1Q?e=UoTLF8)  |  Our completion results including csv and pdf |
| [Public EMAG2 data](https://www.ncei.noaa.gov/emag-survey-page) | EMAG2 data |
|  [Pretrained model](https://1drv.ms/f/c/2e4d56a3d20d5c20/Etyk9PQRhqlEmF1s1v8L6rQBURaacy-HfRvMCk7QxZBhrA?e=pOIINp)  |  Pretrained guided-diffusion model |

Place the pretrained model under `./pretrain` and original EMAG2 data under `./EMAG2`.

### 3. Run example

We prepare an easy test for quick evaluation. The input images and masks are in `./data`. 

```
bash shell/easy_test.sh
```

## Completion method

### 1. Step 0: Data preprocess

Download EMAG2_V3 and place it in `./EMAG2`. Run the below command, you can preprocess the EMAG2_V3.

```shell
python scripts/preprocess.py
```

### 2. Step 1: Global completion

```shell
bash shell/step1.sh
```

### 3. Step 2: Local completion

```shell
bash shell/step2.sh
```

## FAQ

**How to apply it for other datasets?**

If you want train new completion model on a new dataset, it is recommended to follow [guided-diffusion](https://github.com/openai/guided-diffusion) repository to obtain guided-diffusion model, then follow our completion method.

## Acknowledgements

Our code is built upon [RePaint](https://github.com/andreas128/RePaint) and [guided-diffuion](https://github.com/openai/guided-diffusion.git). We thank the authors for their excellent work.

If you have any question, feel free to contact fymao@zju.edu.cn or fangyuanmaocs@gmail.com .
