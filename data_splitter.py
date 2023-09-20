import os
import shutil
import random
import zipfile

# Function to extract contents of zip files in source folder to target folder
def extract_zips(source_folder, target_folder):
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_folder)

# Create .gc files and write file names to them
def create_and_write_gc_files(data_list, gc_file_path):
    with open(gc_file_path, 'w') as gc_file:
        for file_name in data_list:
            gc_file.write(file_name + '\n')

# Define paths for source, target, and rus folders
source_folder = os.getcwd()
target_folder = os.path.realpath(os.path.join(os.getcwd(), 'data'))
rus_folder = os.path.realpath(os.path.join(os.getcwd(), 'rus'))

# Extract contents of zip files to target folder
extract_zips(source_folder, target_folder)

# Get all image file names in the Data folder
image_files = [file for file in os.listdir(target_folder) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

# Shuffle the image file names randomly
random.seed(0)
random.shuffle(image_files)

# Calculate the split point for 80% train and 20% test
split_point = int(0.2 * len(image_files))

# Separate into test and train sets
test_files = image_files[:split_point]
train_files = image_files[split_point:]

# Define paths for test.gc and train.gc files
test_gc_path = os.path.join(rus_folder, "test.gc")
train_gc_path = os.path.join(rus_folder, "train.gc")

# Create .gc files and write file names
create_and_write_gc_files(test_files, test_gc_path)
create_and_write_gc_files(train_files, train_gc_path)