from PIL import Image
import os

# 이미지 파일을 정상적으로 쓸 수 있는지 확인
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

# 다른 파일을 JPEG 형태로 바꿈
def convert_to_jpg(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.bmp', '.gif', '.tiff', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.jpg')
            image.save(output_path, 'JPEG')

    