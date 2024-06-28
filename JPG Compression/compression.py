import os
import cv2
import sys

def compress_images(input_folder, quality=10):
    # Create output folder in the same parent directory as the input folder
    parent_dir = os.path.dirname(input_folder)
    output_folder = os.path.join(parent_dir, "Output Images")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all files in the input folder
    files = os.listdir(input_folder)

    for file in files:
        input_path = os.path.join(input_folder, file)
        
        # Check if the file is an image (assuming JPEG for simplicity)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Read the image using OpenCV
            img = cv2.imread(input_path)

            # Compress the image
            # The third parameter in imwrite is the quality (0-100)
            output_path = os.path.join(output_folder, file)
            cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])

            print(f'Compressed {file} and saved to {output_path}')
        else:
            print(f'Skipped {file} as it is not a supported image format')

if __name__ == "__main__":
    # Take the input folder as a script input
    input_folder = sys.argv[1]
    compress_images(input_folder)
