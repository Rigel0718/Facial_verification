from PIL import Image
import os

def check_image_file_integrity(directory_path):
    fail_list = []
    for directory, _, filename in os.walk(directory_path) :
        for file in filename:
            file_path = os.path.join(directory, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except (IOError, SyntaxError) as e:
                print("Corrupted image file:", file_path)
                fail_list.append(file_path)

    return fail_list