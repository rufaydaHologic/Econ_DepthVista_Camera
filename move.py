import os
import shutil
 
# Use current working directory
current_dir = os.getcwd()
 
# Define the extensions to look for
extensions = ('.png', '.bmp', '.ply', '.txt', '.raw', '.yml')
 
# Destination folder
destination_folder = os.path.join(current_dir, 'RUFAYDA')
os.makedirs(destination_folder, exist_ok=True)
 
# Loop through files in the current directory only (no subdirectories)
for file in os.listdir(current_dir):
    file_path = os.path.join(current_dir, file)
 
    if os.path.isfile(file_path) and file.lower().endswith(extensions):
        dest_path = os.path.join(destination_folder, file)
 
        # Avoid moving the file if it’s already in the destination folder
        if os.path.abspath(file_path) != os.path.abspath(dest_path):
            print(f"Moving: {file_path} -> {dest_path}")
            shutil.move(file_path, dest_path)
 
print("✅ Done. Files have been moved.")