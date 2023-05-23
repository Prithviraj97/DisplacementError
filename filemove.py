import os
import shutil

# set the directory path and target folder path
directory_path = 'O:\\New folder'
target_folder = 'lables'

# iterate over all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):  # check if file ends with .txt
        file_path = os.path.join(directory_path, filename)
        # move the file to the target folder
        shutil.move(file_path, target_folder)
