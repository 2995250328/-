import cv2
import os

def process_images_from_txt(txt_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 读取txt文件
    with open(txt_path, "r") as file:
        lines = file.readlines()

    # 遍历路径并处理
    for idx, line in enumerate(lines):
        # 去除换行符，提取路径
        image_path = line.strip().strip("()").strip("''").strip(',').strip("'")
        print(image_path)
        
        # 使用 OpenCV 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image at {image_path}")
            continue

        # 构造输出路径
        output_path = f"{output_folder}/test{idx + 1}.png"

        # 保存图片
        cv2.imwrite(output_path, image)
        print(f"Saved: {output_path}")

# 设置文件路径和输出目录
txt_file = "7scenes_stairs_feat4_conf2_supp_16m_4head_big_error.txt"  # 替换为你的txt文件路径
output_dir = "7scenes_stairs_feat4_conf2_supp_16m_4head_big_errorr"  # 替换为你的输出目录

process_images_from_txt(txt_file, output_dir)
