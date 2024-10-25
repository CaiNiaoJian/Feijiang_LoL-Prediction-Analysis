# Feijiang_LoL-Prediction-Analysis

This project is a processing of the predictions made by League of Legends masters.

## Table of Contents

- [Environment Setup](#environment-setup)
  - [Check Your Environment](#check-your-environment)
  - [Install PaddlePaddle](#install-paddlepaddle)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Usage](#usage)
- [Validation](#validation)
- [Uninstallation](#uninstallation)
- [License](#license)

## Environment Setup

### Check Your Environment

Before installing PaddlePaddle, ensure your environment meets the following requirements:

1. **Python Version**: Ensure Python version is 3.8/3.9/3.10/3.11/3.12.
   ```bash
   python --version
   ```

2. **pip Version**: Ensure pip version is 20.2.2 or higher.
   ```bash
   python -m ensurepip
   python -m pip --version
   ```

3. **Architecture**: Ensure Python and pip are 64-bit, and the processor architecture is x86_64 (or x64, Intel 64, AMD64).
   ```bash
   python -c "import platform;print(platform.architecture()[0]);print(platform.machine())"
   ```

4. **MKL Support**: The default installation package requires MKL support.

### Install PaddlePaddle

#### CPU Version

If your computer does not have an NVIDIA® GPU, install the CPU version of PaddlePaddle:
```bash
python -m pip install paddlepaddle==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

#### GPU Version

If your computer has an NVIDIA® GPU, ensure it meets the following requirements:

- CUDA Toolkit 11.8 with cuDNN v8.6.0 (TensorRT 8.5.1.7 for PaddleTensorRT inference).
- CUDA Toolkit 12.3 with cuDNN v9.0.0 (TensorRT 8.6.1.6 for PaddleTensorRT inference).
- GPU compute capability greater than 6.0.

Install the GPU version of PaddlePaddle:

- **CUDA 11.8**:
  ```bash
  python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
  ```

- **CUDA 12.3**:
  ```bash
  python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
  ```

**Note**:
- Ensure the Python environment for PaddlePaddle installation is the expected one.
- The default installation package supports AVX and MKL. To check if your machine supports AVX, use the CPU-Z tool.
- If you need an AVX and OpenBLAS version, download the wheel package and install it locally:
  ```bash
  python -m pip install https://paddle-wheel.bj.bcebos.com/3.0.0-beta0/windows/windows-cpu-avx-openblas-vs2017/paddlepaddle-3.0.0b1-cp38-cp38-win_amd64.whl
  ```

## Project Overview

This project aims to build a classification model using the `PaddlePaddle` framework to predict whether a League of Legends player will win or lose a game. The dataset includes in-game statistics such as kills, deaths, assists, and damage dealt.

## Dataset

### Data Description

Each row in the dataset represents a player's game data with the following fields:

- `id`: Player record ID
- `win`: Whether the player won (label variable)
- `kills`: Number of kills
- `deaths`: Number of deaths
- `assists`: Number of assists
- `largestkillingspree`: Largest killing spree
- `largestmultikill`: Largest multi-kill
- `longesttimespentliving`: Longest time spent alive
- `doublekills`: Number of double kills
- `triplekills`: Number of triple kills
- `quadrakills`: Number of quadra kills
- `pentakills`: Number of penta kills
- `totdmgdealt`: Total damage dealt
- `magicdmgdealt`: Magic damage dealt
- `physicaldmgdealt`: Physical damage dealt
- `truedmgdealt`: True damage dealt
- `largestcrit`: Largest critical hit
- `totdmgtochamp`: Damage to opponent players
- `magicdmgtochamp`: Magic damage to opponent players
- `physdmgtochamp`: Physical damage to opponent players
- `truedmgtochamp`: True damage to opponent players
- `totheal`: Total healing
- `totunitshealed`: Total units healed
- `dmgtoturrets`: Damage to turrets
- `timecc`: Crowd control time
- `totdmgtaken`: Total damage taken
- `magicdmgtaken`: Magic damage taken
- `physdmgtaken`: Physical damage taken
- `truedmgtaken`: True damage taken
- `wardsplaced`: Number of wards placed
- `wardskilled`: Number of wards killed
- `firstblood`: Whether the player got first blood

### Dataset Files

- `train.csv`: Training dataset with 8 million records.
- `test.csv`: Test dataset with samples to predict.

## Usage

### Data Preprocessing

In `lol_prediction.py`, the data preprocessing steps include:

- Loading the training and test data.
- Standardizing the feature data.
- Converting the data to `float32` type.

### Model Training

The model uses a simple fully connected neural network for training. The training process includes:

- Defining the model structure.
- Defining the loss function and optimizer.
- Training the model using the training data.

### Model Prediction

After training, use the test data for prediction and save the results to `submission.csv`.

### Running the Code

Run the following command in the terminal:
```bash
python lol_prediction.py
```

## Validation

After installation, you can verify the installation by running:
```bash
python -c "import paddle; paddle.utils.run_check()"
```
If you see `PaddlePaddle is installed successfully!`, the installation is successful.

## Uninstallation

To uninstall PaddlePaddle:

- **CPU Version**:
  ```bash
  python -m pip uninstall paddlepaddle
  ```

- **GPU Version**:
  ```bash
  python -m pip uninstall paddlepaddle-gpu
  ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

For more information, refer to the [PaddlePaddle official documentation](https://www.paddlepaddle.org.cn/).
