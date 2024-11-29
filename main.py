import os
from pathlib import Path

from PIL import Image
from trdg.generators import GeneratorFromStrings
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


# Step 1: Generate Synthetic Character Dataset
def generate_characters(output_dir, font_path, count_per_char=1):
    """
    Generate images for characters (0-9, A-Z) using TextRecognitionDataGenerator.
    """
    os.makedirs(output_dir, exist_ok=True)
    characters = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    generator = GeneratorFromStrings(
        characters,
        count=count_per_char * len(characters),
        fonts=[font_path],
        size=64,
        random_blur=False,
        skewing_angle=0,
        background_type=0,
        distorsion_type=0,
    )

    for idx, (image, label) in enumerate(generator):
        char_output_dir = os.path.join(output_dir, label)
        os.makedirs(char_output_dir, exist_ok=True)
        image.save(os.path.join(char_output_dir, f"{label}_{idx}.png"))
        print(f"Generated: {label}_{idx}.png")


# Step 2: Preprocess the Dataset
def preprocess_characters(input_dir, output_dir):
    """
    Preprocess character images: Resize, normalize, and save as tensors.
    """
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    os.makedirs(output_dir, exist_ok=True)

    for char_label in os.listdir(input_dir):
        char_input_dir = os.path.join(input_dir, char_label)
        char_output_dir = os.path.join(output_dir, char_label)
        os.makedirs(char_output_dir, exist_ok=True)

        for image_file in os.listdir(char_input_dir):
            image_path = os.path.join(char_input_dir, image_file)
            image = Image.open(image_path)
            processed_image = transform(image)
            torch.save(processed_image, os.path.join(char_output_dir, f"{image_file}.pt"))
            print(f"Preprocessed: {image_file} -> {char_label}")


# Step 3: Define the Dataset Class
class CharacterDataset(Dataset):
    def __init__(self, data_dir):
        """
        Initialize the dataset with preprocessed character data.
        """
        self.samples = []
        self.data_dir = data_dir

        # Define a mapping from characters to integers
        self.char_to_int = {char: idx for idx, char in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")}

        for char_label in os.listdir(data_dir):
            if char_label not in self.char_to_int:
                print(f"Skipping unknown label: {char_label}")
                continue

            char_dir = os.path.join(data_dir, char_label)
            for file in os.listdir(char_dir):
                self.samples.append((os.path.join(char_dir, file), self.char_to_int[char_label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        tensor = torch.load(file_path)
        return tensor, label


# Step 4: Define the Model
class LicensePlateRecognizer(nn.Module):
    def __init__(self, num_classes):
        super(LicensePlateRecognizer, self).__init__()

        # Convolutional layers
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),  # Batch normalization
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),  # Batch normalization
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Additional layer
            nn.ReLU(),
            nn.BatchNorm2d(128),  # Batch normalization
            nn.MaxPool2d(2, 2)
        )

        # Fully connected layers
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),  # Adjust based on input size
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(128, num_classes)  # Output layer for classification
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x


# Step 5: Train and Evaluate the Model
def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")


def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")


# Main Pipeline
if __name__ == "__main__":
    # Paths
    font_path = f"{Path.home()}/hellotorch/fonts/arialbd.ttf"
    character_output_dir =  f"{Path.home()}/hellotorch/dataset/characters"
    preprocessed_output_dir =  f"{Path.home()}/hellotorch/dataset/preprocessed"

    # Step 1: Generate synthetic character images
    generate_characters(output_dir=character_output_dir, font_path=font_path)

    # Step 2: Preprocess character images
    preprocess_characters(input_dir=character_output_dir, output_dir=preprocessed_output_dir)

    # Step 3: Load Dataset
    dataset = CharacterDataset(preprocessed_output_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Step 4: Define Model, Loss, Optimizer
    num_classes = 36  # 0-9, A-Z
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LicensePlateRecognizer(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Step 5: Train and Evaluate
    train_model(model, dataloader, criterion, optimizer, epochs=10)
    evaluate_model(model, dataloader)

    # Save the trained model
    torch.save(model.state_dict(),  f"{Path.home()}/hellotorch/dataset/trained_model.pth")
    print("Model saved as trained_model.pth")