import os
import random
import string

def generate_random_name():
    # 生成 8 位数字形式：xxxx-xxxx
    part1 = ''.join(random.choices(string.digits, k=4))
    part2 = ''.join(random.choices(string.digits, k=4))
    return f"{part1}-{part2}"

def rename_files_in_folder(folder_path):
    used_names = set()

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 跳过子文件夹
        if not os.path.isfile(file_path):
            continue

        # 获取扩展名
        name, ext = os.path.splitext(filename)

        # 生成唯一新文件名
        new_name = generate_random_name()
        while new_name in used_names or os.path.exists(os.path.join(folder_path, new_name + ext)):
            new_name = generate_random_name()
        used_names.add(new_name)

        # 重命名
        new_path = os.path.join(folder_path, new_name + ext)
        os.rename(file_path, new_path)
        print(f"✔ {filename} → {new_name + ext}")

    print("✅ 所有文件重命名完成。")

# 示例用法（替换成你自己的路径）
folder = "/Users/hendrick/Desktop/MemeVerse/Meme Warehouse/addtional"  # 替换为你的文件夹路径
rename_files_in_folder(folder)
