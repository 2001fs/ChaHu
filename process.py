import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import glob
import os
from PIL import Image
import io

def apply_mask(image, mask):
    """
    将mask应用到图片上，只保留mask区域内的内容
    其他区域用白色填充
    """
    # 确保图片和mask大小一致
    if image.size != mask.size:
        mask = mask.resize(image.size)
    
    # 将图片和mask转换为numpy数组
    image_np = np.array(image)
    mask_np = np.array(mask)
    
    # 创建一个白色背景
    result = np.ones_like(image_np) * 255
    
    # 只保留mask区域内的内容（mask为白色的区域）
    mask_binary = (mask_np > 127)  # 二值化
    result[mask_binary] = image_np[mask_binary]
    
    # 转换回PIL图片
    return Image.fromarray(result.astype(np.uint8))

def process_dataset(input_file, output_file):
    """处理单个数据集文件"""
    print(f'Processing {input_file}...')
    
    # 读取原始数据集
    table = pq.read_table(input_file)
    df = table.to_pandas()
    
    # 处理每张图片
    processed_images = []
    for idx, row in df.iterrows():
        # 读取图片和mask
        img_bytes = row['image']['bytes']
        mask_bytes = row['mask']['bytes']
        
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        mask = Image.open(io.BytesIO(mask_bytes)).convert('L')
        
        # 应用mask
        processed_image = apply_mask(image, mask)
        
        # 将处理后的图片转换为bytes
        buffer = io.BytesIO()
        processed_image.save(buffer, format='JPEG')
        processed_bytes = buffer.getvalue()
        
        # 存储处理后的图片数据
        processed_images.append({
            'bytes': processed_bytes,
            'path': row['image']['path']  # 保留原始路径信息
        })
        
        # 打印进度
        if (idx + 1) % 100 == 0:
            print(f'  Processed {idx + 1}/{len(df)} images')
    
    # 创建新的DataFrame，只保留需要的列
    new_df = pd.DataFrame({
        'id': df['id'],
        'image': processed_images,
        'geometric shape type': df['geometric shape type'],
        'natural shape type': df['natural shape type'],
        'flower type': df['flower type'],
        'handle type': df['handle type'],
        'innovative': df['innovative'],
        'caption': df['caption'],
        'time': df['time']
    })
    
    # 转换为PyArrow Table并保存
    table = pa.Table.from_pandas(new_df)
    pq.write_to_dataset(table, root_path=output_file)
    print(f'  Saved to {output_file}')
    
    return len(df)

def main():
    # 查找所有CN开头的parquet文件
    cn_files = []
    for file in glob.glob('*.parquet'):
        # 检查是否是CN开头且不是已经处理过的（不含-new）
        if file.lower().startswith('cn') and '-new' not in file.lower():
            cn_files.append(file)
    
    print(f'找到 {len(cn_files)} 个CN开头的数据集文件:')
    for f in cn_files:
        print(f'  - {f}')
    
    # 处理每个文件
    total_processed = 0
    for input_file in cn_files:
        # 生成输出文件名（添加-new后缀）
        output_file = input_file.replace('.parquet', '-new.parquet')
        count = process_dataset(input_file, output_file)
        total_processed += count
    
    print(f'\n预处理完成！共处理 {total_processed} 张图片')

if __name__ == '__main__':
    main()