import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import glob
import os
from PIL import Image
import io
import matplotlib.pyplot as plt
import random

def apply_mask(image, mask):
    """
    将mask应用到图片上，只保留mask区域内的内容
    其他区域用白色填充
    """
    if image.size != mask.size:
        mask = mask.resize(image.size)
    
    image_np = np.array(image)
    mask_np = np.array(mask)
    
    result = np.ones_like(image_np) * 255
    mask_binary = (mask_np > 127)
    result[mask_binary] = image_np[mask_binary]
    
    return Image.fromarray(result.astype(np.uint8))

def main():
    # 查找parquet文件
    parquet_files = glob.glob('*.parquet')
    if not parquet_files:
        print('No parquet files found')
        return
    
    # 随机选择一个parquet文件
    input_file = random.choice(parquet_files)
    print(f'Using file: {input_file}')
    
    # 读取数据
    table = pq.read_table(input_file)
    df = table.to_pandas()
    print(f'Total images: {len(df)}')
    
    # 随机选择10张图片
    num_samples = min(10, len(df))
    sample_indices = random.sample(range(len(df)), num_samples)
    
    # 创建一个大图，包含10行3列
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(sample_indices):
        row = df.iloc[idx]
        
        # 读取图片和mask
        img_bytes = row['image']['bytes']
        mask_bytes = row['mask']['bytes']
        
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        mask = Image.open(io.BytesIO(mask_bytes)).convert('L')
        fused = apply_mask(image, mask)
        
        # 第一列：原始图片
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'Image {idx}')
        axes[i, 0].axis('off')
        
        # 第二列：Mask
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'Mask {idx}')
        axes[i, 1].axis('off')
        
        # 第三列：融合结果
        axes[i, 2].imshow(fused)
        axes[i, 2].set_title(f'Fused {idx}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    output_path = 'test_output.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {output_path}')
    plt.show()

if __name__ == '__main__':
    main()