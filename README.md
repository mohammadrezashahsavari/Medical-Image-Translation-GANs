# Medical Image Modality Transformation using Generative Adversarial Networks

This repository provides implementations of several deep learning models for medical image modality transformation. The primary use case demonstrated is the conversion of 2D and 3D Ultrasound (US) images into Magnetic Resonance (MR) images. The project includes various Generative Adversarial Network (GAN) architectures, allowing for both paired and unpaired image-to-image translation.

---

## Models Included

This repository contains the following models, each located in its own directory:

* **CycleGAN (2D & 3D)**: For unpaired image-to-image translation, ideal for datasets where a direct correspondence between source (e.g., US) and target (e.g., MRI) images is not available. This model is implemented using a ResNet-based generator and a PatchGAN discriminator.
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
│   │   Experiment.py           # Main script to configure and run the experiment
│   │   preprocess.py           # Script for data preparation (NIfTI to NPZ)
│   │   utils.py                # Helper functions for data loading and splitting
│   │   dataset.py              # Defines the TensorFlow dataset creation pipeline
│   │
│   ├───Models/
│   │   └───CycleGAN.py         # The CycleGAN model definition
│   │
│   ├───Experiments/
│   │   └───Transformation.py   # Core logic for training and generation
│   │
│   └───Data/
│       ├───US_Images/          # Directory for raw Ultrasound NIfTI files
│       ├───MRI_Images/         # Directory for raw MRI NIfTI files
│       └───Preprocessed/       # Output of preprocess.py
│
│
│───CycleGAN 3D/
│   │
│   │____ ...
│   │
│   │____ ...
│
│
│───Pix2Pix 2D/
│   │
│   │____ ...
│   │
│   │____ ...
│
│───UNet & EncoderDecoder 3D/
│   │
│   │____ ...
│   │
│   │____ ...
```


* **`[Model_Name]/`**: Each main directory contains a standalone implementation of a model.
* **`Experiment.py`**: The main entry point for training and evaluating a model.
* **`preprocess.py`**: Scripts for preparing and augmenting your dataset from raw `.nii.gz` files.
* **`Models/`**: Contains the Python scripts defining the neural network architectures.
* **`Tools/`**: A collection of utility scripts for callbacks, custom loss functions, and metrics.
* **`Output/`** (created on run): Stores the results, including saved model weights, logs, and generated images.

---

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

* **Python 3.x**
* **TensorFlow**
* **TensorFlow-Addons** (for Instance Normalization)
* **NumPy**
* **NiBabel** (for reading medical image files)
* **Matplotlib**
* **tqdm**

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mohammadrezashahsavari/Ultrasound-to-MRI-Transformation.git](https://github.com/mohammadrezashahsavari/Ultrasound-to-MRI-Transformation.git)
    cd Ultrasound-to-MRI-Transformation
    ```

2.  **Install Python dependencies:**
    It's highly recommended to use a virtual environment. The root `requirements.txt` contains common packages.

    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

The process involves two main steps: preparing the data and running an experiment. The following example uses the **CycleGAN 2D** model.

### 1. Data Preparation

The `preprocess.py` script converts 3D NIfTI (`.nii.gz`) images into 2D `.npz` slices that the model can consume.

1.  Navigate to the model's directory.
    ```bash
    cd "CycleGAN 2D"
    ```

2.  Place your raw 3D Ultrasound and MRI volumes (in `.nii.gz` format) into the respective data folders:
    * `Data/US_Images/`
    * `Data/MRI_Images/`

    *Note: The script expects corresponding US and MRI files to have names that can be matched (e.g., `ID_001_US.nii.gz` and `ID_001_MRI.nii.gz`).*

3.  Run the preprocessing script. You can modify the `resize_to` parameter within the script if needed.
    ```bash
    python preprocess.py
    ```
    This will populate the `Data/Preprocessed/` directory with subfolders for each subject, containing the 2D image slices in `.npz` format.

### 2. Running an Experiment

Once the data is preprocessed, you can train a new model or generate images from an existing one using `Experiment.py`.

1.  **Configure the experiment:**
    Open `CycleGAN 2D/Experiment.py` in a text editor. You can set key parameters at the top of the file:
    * `mode`: Set to `'train'` to train a new model, `'generate_outputs_2D'` to create 2D image results, or `'generate_outputs_3D'` to reconstruct and save 3D volumes from the 2D outputs.
    * `max_epochs`: The number of epochs to train for.
    * `lr_gen_G`, `lr_disc_X`, etc.: Learning rates for the generators and discriminators.
    * `random_seed`: Ensures the train/validation/test split is reproducible.

2.  **Run the script:**
    After saving your configuration, run the script from the `CycleGAN 2D` directory:
    ```bash
    python Experiment.py
    ```

    * If `mode = 'train'`, the script will start training and save model weights to the `Output/CycleGan/TrainedModels/` directory. You will also see sample generated images saved periodically in `Output/CycleGan/SampleGeneratedImages/`.
    * If `mode = 'generate_outputs_2D'`, the script will load the latest trained model and generate translated images, differential maps, and a `.csv` file with performance metrics (MAE, RMSE, SSI) in the `Output/CycleGan/Results/2D/` directory.

---

## Results

Here are some sample results from the **CycleGAN 2D** model, transforming ultrasound slices to MR images.

**Training Progress (Sampled Output):**
![Training_First_100_Epochs](https://github.com/mohammadrezashahsavari/Ultrasound-to-MRI-Transformation/assets/76266892/d2dce706-9bbd-4a9b-bd05-619ea2f6f0b1)

**CycleGAN Output Sample on the Test Set:**
![US2MRI](https://github.com/mohammadrezashahsavari/Ultrasound-to-MRI-Slice-Transformation/assets/76266892/e18fed31-1bc1-46ec-8b41-9d1a239191f3)

---

## Contribution & Citation

This repository is an open-source project. If you find this code useful in your research, please consider starring ⭐ this repository. You can also cite it using the "Cite this repository" feature on the GitHub sidebar.

Contributions are welcome! Please feel free to open an issue or submit a pull request.
