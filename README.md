# CycleGAN Image Translation
This repository contains the code for implementing CycleGAN, a deep learning model for image-to-image translation. CycleGAN can learn to translate images from one domain to another without paired training data.

## Prerequisites
- Python 3.x
- PyTorch
- torchvision
- numpy
- tqdm
- matplotlib

## Usage
Clone the repository:

```
git clone https://github.com/your_username/repository_name.git
Install the required dependencies:
```

```
pip install -r requirements.txt
Run the code:
```

```
python code.py
```

## Command-line Arguments
- run_path: The path to save the output images and metrics. Default: 100.
- num_epochs: The number of training epochs. Default: 1000.
- batch_size: The batch size for training. Default: 32.
- samples_per_class: The maximum number of samples per class. Default: 1000.
- learning_rate_g: The learning rate for the generator. Default: 3e-4.
- learning_rate_d: The learning rate for the discriminator. Default: 1e-4.
- z_size: The size of the input noise vector. Default: 128.
- coef_c: The coefficient for the cycle consistency loss. Default: 10.
- coef_i: The coefficient for the identity loss. Default: 5.0.
- is_debug: Set to True for debugging mode. Default: False.

Note: The code supports both CPU and GPU (CUDA) execution. By default, it uses the GPU if available, otherwise falls back to CPU.

## Dataset
The code expects the dataset to be organized as follows:

```
data
├── trainA
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── testA
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── trainB
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── testB
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```
The trainA and trainB folders contain the training images from domain A and domain B, respectively. The testA and testB folders contain the test images from domain A and domain B, respectively.

## Results
During training, the code will display and save the following information:

- Losses:

  - train_loss_gan: Adversarial loss of the generator.
  - train_loss_identity: Identity loss.
  - train_loss_d: Discriminator loss.
  - train_loss_cycle: Cycle consistency loss.
    
- Sample Images:

  - Real winter scenes.
  - Transformed summer scenes.
  - Recovered winter scenes.
  - Real summer scenes.
  - Transformed winter scenes.
  - Recovered summer scenes.
    
The metrics and sample images will be saved in the specified run_path directory.
