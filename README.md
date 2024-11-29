
# License Plate Character Recognition

This repository contains a pipeline for generating synthetic character datasets, preprocessing images, and training a Convolutional Neural Network (CNN) model to recognize alphanumeric characters from license plates. The implementation uses PyTorch and leverages synthetic data generation with the `TextRecognitionDataGenerator`.

---

## Features

1. **Synthetic Dataset Generation**: Generate images for alphanumeric characters (`0-9`, `A-Z`) using a specified font.
2. **Preprocessing**: Normalize and resize images to a uniform size and save them as tensors.
3. **Custom Dataset Class**: Load the preprocessed data for training and evaluation.
4. **CNN Model**: A convolutional neural network for character classification with batch normalization and dropout for regularization.
5. **Training and Evaluation**: Train the model on the generated dataset and evaluate its accuracy.

---

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/oianmol/license-plate-recognition.git
    cd license-plate-recognition
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have the following additional packages installed:
    - `Pillow`
    - `torch`
    - `torchvision`
    - `trdg` (TextRecognitionDataGenerator)

4. Download a suitable TTF font (e.g., Arial Bold) and specify its path in the script.

---

## Usage

### 1. Generate Synthetic Dataset
Run the script to generate synthetic images for characters (`0-9`, `A-Z`):

```python
generate_characters(output_dir="dataset/characters", font_path="/path/to/font.ttf", count_per_char=100)
```

### 2. Preprocess Images
Preprocess the generated images by resizing, normalizing, and converting them to tensors:

```python
preprocess_characters(input_dir="dataset/characters", output_dir="dataset/preprocessed")
```

### 3. Train the Model
Train the character recognition model on the preprocessed dataset:

```bash
python main.py
```

The model will be saved as `trained_model.pth` in the dataset directory upon completion.

---

## Model Architecture

The CNN model includes:

1. **Convolutional Layers**: Extract spatial features from the input images with ReLU activation.
2. **Batch Normalization**: Normalize activations for stable training.
3. **Pooling Layers**: Reduce the spatial dimensions while retaining key features.
4. **Fully Connected Layers**: Perform classification on the extracted features.

---

## File Structure

```plaintext
.
├── dataset/
│   ├── characters/          # Synthetic character images
│   └── preprocessed/        # Preprocessed tensors
├── fonts/                   # Directory to store fonts
├── main.py                  # Main script for training and evaluation
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- torchvision 0.10+
- Pillow
- TextRecognitionDataGenerator

---

## Future Enhancements

- Integration with real-world datasets for transfer learning.
- Extend the model to support multi-line text recognition.
- Add functionality for license plate segmentation and character extraction.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
