# PEGANs: 
# PEGANs
![framework](img/fig1.png?raw=true "framework")
This is the official implementation of PE-GANs

## Requirements
- Python 3.9.0
- Python packages
    ```sh
    # update `pip` for installing tensorboard.
    pip install -U pip setuptools
    pip install -r requirements.txt
    ```

## Datasets
- CIFAR-10

    Pytorch build-in CIFAR-10 will be downloaded automatically.

- STL-10

    Pytorch build-in STL-10 will be downloaded automatically.

## Preprocessing Datasets for FID
Pre-calculated statistics for FID can be downloaded [here](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC?usp=sharing):
- cifar10.train.npz - Training set of CIFAR10
- cifar10.test.npz - Testing set of CIFAR10
- stl10.unlabeled.48.npz - Unlabeled set of STL10 in resolution 48x48

Folder structure:
```
./stats
├── cifar10.test.npz
├── cifar10.train.npz
└── stl10.unlabeled.48.npz
```

**NOTE**

All the reported values (Inception Score and FID) in our paper are calculated by official implementation instead of our implementation. 


## Training
- Configuration files
    - We use `absl-py` to parse, save and reload the command line arguments.
    - All the configuration files can be found in `./config`. 
    - The compatible configuration list is shown in the following table:

        | Script          |Configurations|
        |----------|----------|
        | `trainPEGAN.py` |`PEGAN_P5_CIFAR10_CNN.txt`<br>`PEGAN_P5_STL10_CNN.txt`<br>`PEGAN_P10_CIFAR10_CNN.txt`<br>`PEGAN_P10_STL10_CNN .txt`|
        | `trainSNGAN.py` |`SNGAN_CNN_CIFAR10.txt`|
        | `trainWGAN.py`  |`WGAN_CNN_CIFAR10.txt`|
        | `trainWGANGP.py`  |`WGAN_GP_CNN_CIFAR10.txt`|
- Run the training script with the compatible configuration, e.g.,
    - `trainPEGAN.py` supports training gan on `CIFAR10` and `STL10`, e.g.,
        ```sh
        python trainPEGAN.py \
            --flagfile ./config/PEGAN_P10_CIFAR10_CNN.txt
        ```
    
- Generate images from checkpoints, e.g.,

    `--eval`: evaluate best checkpoint.

    `--save PATH`: save the generated images to `PATH`
    ```
    python train.py \
        --flagfile ./logs/PEGAN_P10_CIFAR10_CNN/flagfile.txt \
        --eval \
        --save path/to/generated/images
    ```
## Acknowledgments
Pytorch framework from [GNGAN](https://github.com/basiclab/GNGAN-PyTorch).

