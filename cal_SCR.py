import numpy as np
import cv2
import os
import json
from glob import glob
from tqdm import tqdm


def calculate_scr(img, box):
    image = img.copy()
    # 获取目标区域
    tg_area = image[box[1]: box[3] + 1, box[0]: box[2] + 1].copy()
    tg_center = [box[0] + (box[2] - box[0]) // 2, box[1] + (box[3] - box[1]) // 2]
    tg_size = [box[2] - box[0] + 1, box[3] - box[1] + 1]
    image[box[1]: box[3] + 1, box[0]: box[2] + 1] = 0

    # 计算背景区域的范围
    bg_width = 2 * tg_size[0]  # 修改为只计算目标区域的宽度两倍
    bg_height = 2 * tg_size[1]  # 修改为只计算目标区域的高度两倍
    bg_x_min = max(0, tg_center[0] - bg_width // 2)
    bg_x_max = min(image.shape[1], tg_center[0] + bg_width // 2)
    bg_y_min = max(0, tg_center[1] - bg_height // 2)
    bg_y_max = min(image.shape[0], tg_center[1] + bg_height // 2)

    # 提取背景区域的像素值，确保不包含目标区域
    bg_area = image[bg_y_min: bg_y_max, bg_x_min: bg_x_max].flatten()

    # 生成目标区域的掩码
    target_mask = np.zeros((bg_height, bg_width), dtype=np.uint8)

    # 计算目标区域在背景区域中的坐标
    target_x_min = box[0] - bg_x_min
    target_x_max = target_x_min + (box[2] - box[0])
    target_y_min = box[1] - bg_y_min
    target_y_max = target_y_min + (box[3] - box[1])

    target_mask[target_y_min:target_y_max + 1, target_x_min:target_x_max + 1] = 1

    # 排除目标区域的像素
    bg_area = bg_area[target_mask.flatten() == 0]

    # 计算目标局部信杂比
    if bg_area.size == 0:  # 如果背景区域为空，返回0
        return 0, image
    scr = abs(np.mean(tg_area) - np.mean(bg_area)) / (np.std(bg_area, ddof=1) + 1e-5)

    # 绘制背景区域和目标区域的边界框
    cv2.rectangle(image, (bg_x_min, bg_y_min), (bg_x_max, bg_y_max), (0, 255, 0), 1)  # 绿色框
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)  # 红色框

    return scr, image  # 返回 SCR 和标注后的图像


def main():
    data_path = './SCR_imgs/images/'
    image_files = glob(data_path + '*.png')

    scr_list = []
    total_scr = 0
    total_count = 0

    for image_file in image_files:
        file_name = image_file.split('\\')[-1].split('.')[0]
        print(file_name)

        # 读取原始图像
        image = cv2.imread(image_file).astype(np.float32)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image[np.where(image == 0)] = 1e-5

        # 读取目标真实值
        label_file = image_file.replace('png', 'txt')
        f = open(label_file, 'r')
        label = f.readlines()
        f.close()

        one_image_scr = []
        # 对图像中每一个目标, 计算其SCR
        for box in label:
            box = list(map(int, box.split(',')))
            # 计算SCR
            scr, annotated_image = calculate_scr(image, box)  # 计算原图像的SCR
            one_image_scr.append(scr)
            total_scr += scr
            total_count += 1

        scr_list.append(one_image_scr)

        # 保存标注后的图像
        annotated_image_path = f'./SCR_imgs/annotated_{file_name}.png'
        cv2.imwrite(annotated_image_path, annotated_image)
        print(f"Saving annotated image to: {annotated_image_path}")

        save_file = open('./SCR_imgs/scr_0312.txt', 'a')
        save_file.write('{}:{}\n'.format(file_name, one_image_scr))
        save_file.close()

if __name__ == '__main__':
    main()