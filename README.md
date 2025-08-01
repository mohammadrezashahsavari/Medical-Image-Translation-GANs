# Medical Image Modality Transformation using Generative Adversarial Networks

This repository provides implementations of several deep learning models for medical image modality transformation. The primary use case demonstrated is the conversion of 2D and 3D Ultrasound (US) images into Magnetic Resonance (MR) images. The project includes various Generative Adversarial Network (GAN) architectures, allowing for both paired and unpaired image-to-image translation.

---

## Models Included

This repository contains the following models, each located in its own directory:

* **CycleGAN (2D & 3D)**: For unpaired image-to-image translation, ideal for datasets where a direct correspondence between source (e.g., US) and target (e.g., MRI) images is not available.
* **Pix2Pix (2D)**: For paired image-to-image translation, which requires datasets where each source image has a corresponding target image.
* **UNet & Encoder-Decoder (3D)**: Architectures commonly used for image segmentation and translation tasks in 3D medical imaging.

---

## Directory Structure

The repository is organized by model. Each model's directory contains the necessary scripts to run experiments.

```bash
.
│   README.md
│   requirements.txt
│
├───CycleGAN 2D/
│   │   Experiment.py           # Main script to run the 2D CycleGAN experiment
│   │   preprocess.py           # Script for data preparation
│   │   Models/CycleGAN.py      # The CycleGAN model definition
│   │   ...
│
├───CycleGAN 3D/
│   │   Experiment.py           # Main script to run the 3D CycleGAN experiment
│   │   ...
│
├───Pix2Pix 2D/
│   │   Experiment.py           # Main script to run the 2D Pix2Pix experiment
│   │   Models/Pix2Pix2D.py     # The Pix2Pix model definition
│   │   ...
│
└───UNet & EncoderDecoder 3D/
│   Experiment.py           # Main script to run the 3D UNet/Encoder-Decoder experiment
│   Models/UNet.py          # The UNet model definition
│   Models/EncoderDecoder.py # The Encoder-Decoder model definition
│   ...
```

* **`[Model_Name]/`**: Each main directory contains a standalone implementation of a model.
* **`Experiment.py`**: The main entry point for training and evaluating a model.
* **`preprocess.py`**: Scripts for preparing and augmenting your dataset.
* **`Models/`**: Contains the Python scripts defining the neural network architectures.
* **`Tools/`**: A collection of utility scripts for callbacks, custom loss functions, and metrics.
* **`output/`** (created on run): Stores the results of the experiments, including saved model weights and generated images.

---

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.x**
* **TensorFlow**
* **NumPy**
* **SciPy**
* **Matplotlib**
* **SimpleITK** or **NiBabel** (for handling medical image formats)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mohammadrezashahsavari/Ultrasound-to-MRI-Transformation.git](https://github.com/mohammadrezashahsavari/Ultrasound-to-MRI-Transformation.git)
    cd Ultrasound-to-MRI-Transformation
    ```

2.  **Install Python dependencies:**
    It's highly recommended to use a virtual environment. The root `requirements.txt` contains common packages. Some models may have specific dependencies listed in their respective directories.

    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

The process for running an experiment is similar for each model. The following example uses the **CycleGAN 2D** model.

### 1. Data Preparation

1.  Organize your dataset. For unpaired models like CycleGAN, you will need two separate directories of images (e.g., one for Ultrasound, one for MRI). For paired models like Pix2Pix, you'll need corresponding image pairs.
    ```
    data/
    ├───trainA/ (e.g., Ultrasound images)
    │   └─── img1.png
    │   └─── ...
    └───trainB/ (e.g., MR images)
        └─── img1.png
        └─── ...
    ```

2.  Run the preprocessing script within the model's directory to format your data correctly. You may need to adjust paths within the script.
    ```bash
    cd "CycleGAN 2D"
    python preprocess.py --dataroot ../data
    ```

### 2. Running an Experiment

1.  Navigate to the directory of the model you wish to use.
    ```bash
    cd "CycleGAN 2D"
    ```

2.  Configure the experiment parameters inside `Experiment.py`. This includes setting the `dataset_path`, learning rate, epochs, and other hyperparameters.

3.  Run the main experiment script.
    ```bash
    python Experiment.py
    ```

The training progress will be displayed in the console. Generated images, model checkpoints, and logs will be saved to an `output/` directory created within the model's folder.

---

## Results

Here are some sample results from the **CycleGAN 2D** model, transforming ultrasound slices to MR images.

**Training Progress (First 100 Epochs):**
![Training_First_100_Epochs](https://github.com/mohammadrezashahsavari/Ultrasound-to-MRI-Transformation/assets/76266892/d2dce706-9bbd-4a9b-bd05-619ea2f6f0b1)

**CycleGAN Output Sample on the Test Set:**
![US2MRI](https://github.com/mohammadrezashahsavari/Ultrasound-to-MRI-Slice-Transformation/assets/76266892/e18fed31-1bc1-46ec-8b41-9d1a239191f3)

---

## Contribution & Citation

This repository is an open-source project. If you find this code useful in your research, please consider starring ⭐ this repository. You can also cite it using the "Cite this repository" feature on the GitHub sidebar.

Contributions are welcome! Please feel free to open an issue or submit a pull request.
