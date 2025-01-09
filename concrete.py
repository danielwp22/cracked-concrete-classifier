import urllib.request
import zipfile
import os

# URLs for the files
# positive_url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Positive_tensors.zip"
# negative_url = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Negative_tensors.zip"

# Output file names
# positive_zip = "Positive_tensors.zip"
# negative_zip = "Negative_tensors.zip"

# Function to download and unzip a file
#def download_and_unzip(url, output_zip):
    # Download the file
    #urllib.request.urlretrieve(url, output_zip)
    #print(f"Downloaded: {output_zip}")
    
    # Unzip the file
    #with zipfile.ZipFile(output_zip, 'r') as zip_ref:
     #   zip_ref.extractall(".")
    #print(f"Extracted: {output_zip}")

    # Optionally remove the ZIP file after extraction
    #os.remove(output_zip)
    #print(f"Removed: {output_zip}")

# Download and extract both files
#download_and_unzip(positive_url, positive_zip)
#download_and_unzip(negative_url, negative_zip)

#zip_path = "Negative_tensors.zip"
#extract_dir = "./Negative_tensors"

#try:
 #   with zipfile.ZipFile(zip_path, 'r') as zip_ref:
  #      zip_ref.extractall(extract_dir)
   # print("Extraction successful.")
#except zipfile.BadZipFile:
 #   print("Error: The file is corrupted or not a valid zip file.")
#except Exception as e:
 #   print(f"An unexpected error occurred: {e}")


import torchvision.models as models
from PIL import Image
import pandas
from torchvision import transforms
import torch.nn as nn
import time
import torch 
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import glob
torch.manual_seed(0)
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd
import os


class Dataset(Dataset):
    # Constructor
    def __init__(self, transform=None, train=True, train_size=30000):
        directory = "/Users/danielpalin/Pythonstuff"
        positive = "Positive_tensors"
        negative = "Negative_tensors"

        positive_file_path = os.path.join(directory, positive)
        negative_file_path = os.path.join(directory, negative)

        # List .pt files in positive and negative directories
        positive_files = [os.path.join(positive_file_path, file) for file in os.listdir(positive_file_path) if file.endswith(".pt")]
        negative_files = [os.path.join(negative_file_path, file) for file in os.listdir(negative_file_path) if file.endswith(".pt")]

        # Count total samples
        total_positive = len(positive_files)
        total_negative = len(negative_files)

        # Print counts
        print(f"Number of positive tensors: {total_positive}")
        print(f"Number of negative tensors: {total_negative}")

        # Calculate number_of_samples based on the smaller count
        number_of_samples = min(total_positive, total_negative) * 2  # *2 to account for both classes

        # Initialize the file list
        self.all_files = [None] * number_of_samples
        self.all_files[::2] = positive_files[:number_of_samples // 2]  # Fill even indices with positive files
        self.all_files[1::2] = negative_files[:number_of_samples // 2]  # Fill odd indices with negative files

        # Transform for preprocessing
        self.transform = transform

        # Create target labels
        self.Y = torch.zeros(number_of_samples).type(torch.LongTensor)
        self.Y[::2] = 1  # Positive class label
        self.Y[1::2] = 0  # Negative class label

        # Remove specific corrupted files
        self._remove_specific_corrupted_files(corrupted_files=[
            "/Users/danielpalin/Negative_tensors/16260.pt",
            "/Users/danielpalin/Positive_tensors/5114.pt"
        ])

        # Split into train and validation datasets
        if train:
            self.len = min(train_size, len(self.all_files))  # Limit to train_size
            self.all_files = self.all_files[:self.len]
            self.Y = self.Y[:self.len]
        else:
            start_index = number_of_samples  
            if total_positive > total_negative:
                # Add remaining positive samples if negative samples are exhausted
                remaining_positive_files = positive_files[number_of_samples // 2:total_positive]
                self.all_files += remaining_positive_files
                self.Y = torch.cat((self.Y, torch.ones(len(remaining_positive_files), dtype=torch.long)))
            else:
                # Add remaining negative samples if positive samples are exhausted
                remaining_negative_files = negative_files[number_of_samples // 2:total_negative]
                self.all_files += remaining_negative_files
                self.Y = torch.cat((self.Y, torch.zeros(len(remaining_negative_files), dtype=torch.long)))

            self.len = len(self.all_files)

    def _remove_specific_corrupted_files(self, corrupted_files):
        """Remove specific corrupted .pt files from the dataset."""
        # Keep only valid files and corresponding labels
        valid_files = []
        valid_labels = []

        for file, label in zip(self.all_files, self.Y):
            if file not in corrupted_files:
                valid_files.append(file)
                valid_labels.append(label)

        self.all_files = valid_files
        self.Y = torch.tensor(valid_labels)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        image = torch.load(self.all_files[idx])
        y = self.Y[idx]
                  
        if self.transform:
            image = self.transform(image)

        return image, y

train_dataset = Dataset(train=True)
validation_dataset = Dataset(train=False)
print("done")

print(len(train_dataset))
print(len(validation_dataset))

model = models.resnet18(weights = True)


for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512,2)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
validation_loader = DataLoader(validation_dataset, batch_size=100, shuffle=False)

optimizer = torch.optim.Adam([parameters  for parameters in model.parameters() if parameters.requires_grad],lr=0.001)

len(validation_dataset)

def validate_files(dataset):
    for idx in range(len(dataset)):
        try:
            _ = torch.load(dataset.all_files[idx])
        except EOFError:
            print(f"Corrupted file found at index {idx}: {dataset.all_files[idx]}")
        except Exception as e:
            print(f"Error encountered while validating file at index {idx}: {e}")

validate_files(train_dataset)



n_epochs = 1
loss_list = []
accuracy_list = []
N_test = len(validation_dataset)
N_train = len(train_dataset)
start_time = time.time()

print("starting")
for epoch in range(n_epochs):
    model.train()  # Set the model to training mode
    for x, y in train_loader:
        # Clear gradients
        optimizer.zero_grad()

        try:
            # Make a prediction
            yhat = model(x)

            # Calculate loss
            loss = criterion(yhat, y)

            # Calculate gradients of parameters
            loss.backward()

            # Update parameters
            optimizer.step()

            # Store the loss value
            loss_list.append(loss.item())  # Use .item() to get the scalar value
        except EOFError as e:
            print(f"EOFError encountered: {e}. Skipping this batch.")
            continue  # Skip this batch and continue with the next

    correct = 0
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for validation
        for x_test, y_test in validation_loader:
            try:
                # Make a prediction
                yhat = model(x_test)

                # Find max
                _, predicted = torch.max(yhat, 1)

                # Calculate correct predictions
                correct += (predicted == y_test).sum().item()
            except EOFError as e:
                print(f"EOFError encountered during validation: {e}. Skipping this batch.")
                continue  # Skip this batch and continue with the next

    # Calculate accuracy
    accuracy = correct / N_test if N_test > 0 else 0
    accuracy_list.append(accuracy)  # Store the accuracy for this epoch

    # Print metrics for the current epoch
    print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {np.mean(loss_list[-len(train_loader):]):.4f}, Accuracy: {accuracy:.4f}')

# Total training time
end_time = time.time()
print(f'Training completed in {end_time - start_time:.2f} seconds.')

print(accuracy)

plt.plot(loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
print("Plotting...")
plt.show()
print("Done plotting.")
