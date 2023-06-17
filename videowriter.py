import cv2
import os

# Set the directory containing the input images
input_directory = 'E:\img3400-3500\png'

# Set the output video file name
output_file = 'E:\img3400.mp4'

# Set the desired frames per second (fps)
output_fps = 60

# Get the list of image filenames in the input directory
image_files = sorted([file for file in os.listdir(input_directory) if file.endswith(('.jpg', '.png'))])

# Read the first image to get the width, height, and channel information
first_image = cv2.imread(os.path.join(input_directory, image_files[0]))
height, width, channels = first_image.shape

# Create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, output_fps, (width, height))

# Loop over the image files and write them to the output video
for image_file in image_files:
    # Read the image
    image = cv2.imread(os.path.join(input_directory, image_file))
    
    # Write the image to the output video file
    out.write(image)
    
    # Display the image (optional)
    cv2.imshow('Image', image)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video writer object
out.release()

# Destroy any open windows
cv2.destroyAllWindows()
