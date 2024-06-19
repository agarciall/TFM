import os
from PIL import Image

input_dir = './'  
output_dir = '../newHRD'  

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for dirpath, dirnames, filenames in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".bmp"):  
            img = Image.open(os.path.join(dirpath, filename))
            img_resized = img.resize((478, 319))  # Change the image resolution

            # Create the same subdirectory structure in the output directory
            relative_path = os.path.relpath(dirpath, input_dir)
            new_dir = os.path.join(output_dir, relative_path)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            img_resized.save(os.path.join(new_dir, filename))  