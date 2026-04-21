import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import io
import os
import glob
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import time

# ==================== 参数配置 ====================
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
NUM_EPOCHS = 50
IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_SIZE = 0.2

# ID筛选范围
ID_START = 'JN000001'
ID_END = 'JN009354'

LR_STEP_SIZE = 15
LR_GAMMA = 0.5
WEIGHT_DECAY = 1e-4

# ==================================================

# 全局变量存储四个类别的信息
GEOMETRIC_CLASS_NAMES = None
NATURAL_CLASS_NAMES = None
FLOWER_CLASS_NAMES = None
HANDLE_CLASS_NAMES = None

geometric_name_to_idx = None
natural_name_to_idx = None
flower_name_to_idx = None
handle_name_to_idx = None

NUM_GEOMETRIC_CLASSES = None
NUM_NATURAL_CLASSES = None
NUM_FLOWER_CLASSES = None
NUM_HANDLE_CLASSES = None

# 自定义数据集类（多任务）
class TeapotDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        
        img_bytes = row['image']['bytes']
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # 获取四个类型的标签
        geo_label = geometric_name_to_idx.get(row['geometric shape type'], 0)
        nat_label = natural_name_to_idx.get(row['natural shape type'], 0)
        flo_label = flower_name_to_idx.get(row['flower type'], 0)
        han_label = handle_name_to_idx.get(row['handle type'], 0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, (geo_label, nat_label, flo_label, han_label)

# SE模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# SE-ResNet Block
class SEBasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# 多任务SE-ResNet-34模型
class MultiTaskSEResNet34(nn.Module):
    def __init__(self, num_geometric, num_natural, num_flower, num_handle):
        super(MultiTaskSEResNet34, self).__init__()
        self.inplanes = 64
        
        # 共享的卷积主干
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(SEBasicBlock, 64, 3)
        self.layer2 = self._make_layer(SEBasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(SEBasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(SEBasicBlock, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 四个任务的分类头
        self.fc_geometric = nn.Linear(512, num_geometric)
        self.fc_natural = nn.Linear(512, num_natural)
        self.fc_flower = nn.Linear(512, num_flower)
        self.fc_handle = nn.Linear(512, num_handle)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 共享特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # 四个任务的输出
        geo_out = self.fc_geometric(x)
        nat_out = self.fc_natural(x)
        flo_out = self.fc_flower(x)
        han_out = self.fc_handle(x)
        
        return geo_out, nat_out, flo_out, han_out

# 根据验证准确率动态调整任务权重
def dynamic_task_weight(val_accs, base_weights=[0.25, 0.25, 0.25, 0.25]):
    # 表现差的任务分配更高权重
    inv_accs = [1 - acc for acc in val_accs]
    inv_accs = [w / sum(inv_accs) for w in inv_accs]
    # 混合权重
    weights = [0.7 * base + 0.3 * inv for base, inv in zip(base_weights, inv_accs)]
    return weights

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience=5, 
               class_weights=None, use_dynamic_weights=True, weight_adjust_method='hybrid'):
    """
    Args:
        class_weights: dict，每个任务的类别权重
        use_dynamic_weights: 是否使用动态权重调整
        weight_adjust_method: 权重调整方法
            - 'accuracy': 根据验证准确率调整
            - 'hybrid': 包含历史状态的混合权重调整
    """
    train_losses = []
    val_losses = []
    train_accs = {'geometric': [], 'natural': [], 'flower': [], 'handle': []}
    val_accs = {'geometric': [], 'natural': [], 'flower': [], 'handle': []}
    
    best_acc = 0.0
    best_model_path = 'multitask_best.pth'
    
    # 初始化任务权重（基于样本数量的权重已通过class_weights传入）
    task_weights = [0.25, 0.25, 0.25, 0.25]
    
    # 创建带类别权重的损失函数（如果提供了类别权重）
    if class_weights is not None:
        geo_weight_tensor = torch.tensor(class_weights['geometric'], dtype=torch.float32).to(DEVICE)
        nat_weight_tensor = torch.tensor(class_weights['natural'], dtype=torch.float32).to(DEVICE)
        flo_weight_tensor = torch.tensor(class_weights['flower'], dtype=torch.float32).to(DEVICE)
        han_weight_tensor = torch.tensor(class_weights['handle'], dtype=torch.float32).to(DEVICE)
        
        criterion_geo = nn.CrossEntropyLoss(weight=geo_weight_tensor)
        criterion_nat = nn.CrossEntropyLoss(weight=nat_weight_tensor)
        criterion_flo = nn.CrossEntropyLoss(weight=flo_weight_tensor)
        criterion_han = nn.CrossEntropyLoss(weight=han_weight_tensor)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        
        # 准确率统计
        correct = {'geometric': 0, 'natural': 0, 'flower': 0, 'handle': 0}
        total = {'geometric': 0, 'natural': 0, 'flower': 0, 'handle': 0}
        
        for images, labels in train_loader:
            images = images.to(DEVICE)
            geo_labels, nat_labels, flo_labels, han_labels = [l.to(DEVICE) for l in labels]
            
            optimizer.zero_grad()
            
            geo_out, nat_out, flo_out, han_out = model(images)
            
            # 计算四个任务的损失（使用带类别权重的损失函数）
            if class_weights is not None:
                loss_geo = criterion_geo(geo_out, geo_labels)
                loss_nat = criterion_nat(nat_out, nat_labels)
                loss_flo = criterion_flo(flo_out, flo_labels)
                loss_han = criterion_han(han_out, han_labels)
            else:
                loss_geo = criterion(geo_out, geo_labels)
                loss_nat = criterion(nat_out, nat_labels)
                loss_flo = criterion(flo_out, flo_labels)
                loss_han = criterion(han_out, han_labels)
            
            # 使用动态任务权重计算总损失
            loss = (task_weights[0] * loss_geo + 
                    task_weights[1] * loss_nat + 
                    task_weights[2] * loss_flo + 
                    task_weights[3] * loss_han)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            
            # 计算准确率
            for name, out, lbl in zip(['geometric', 'natural', 'flower', 'handle'],
                                     [geo_out, nat_out, flo_out, han_out],
                                     [geo_labels, nat_labels, flo_labels, han_labels]):
                _, pred = torch.max(out.data, 1)
                total[name] += lbl.size(0)
                correct[name] += (pred == lbl).sum().item()
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(train_loader.dataset)
        
        for name in ['geometric', 'natural', 'flower', 'handle']:
            train_accs[name].append(correct[name] / total[name])
        
        scheduler.step()
        
        # 验证
        model.eval()
        val_total_loss = 0.0
        val_correct = {'geometric': 0, 'natural': 0, 'flower': 0, 'handle': 0}
        val_total = {'geometric': 0, 'natural': 0, 'flower': 0, 'handle': 0}
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                geo_labels, nat_labels, flo_labels, han_labels = [l.to(DEVICE) for l in labels]
                
                geo_out, nat_out, flo_out, han_out = model(images)
                
                # 使用带类别权重的损失函数
                if class_weights is not None:
                    loss_geo = criterion_geo(geo_out, geo_labels)
                    loss_nat = criterion_nat(nat_out, nat_labels)
                    loss_flo = criterion_flo(flo_out, flo_labels)
                    loss_han = criterion_han(han_out, han_labels)
                else:
                    loss_geo = criterion(geo_out, geo_labels)
                    loss_nat = criterion(nat_out, nat_labels)
                    loss_flo = criterion(flo_out, flo_labels)
                    loss_han = criterion(han_out, han_labels)
                
                val_total_loss += (task_weights[0] * loss_geo + 
                                   task_weights[1] * loss_nat + 
                                   task_weights[2] * loss_flo + 
                                   task_weights[3] * loss_han).item() * images.size(0)
                
                for name, out, lbl in zip(['geometric', 'natural', 'flower', 'handle'],
                                         [geo_out, nat_out, flo_out, han_out],
                                         [geo_labels, nat_labels, flo_labels, han_labels]):
                    _, pred = torch.max(out.data, 1)
                    val_total[name] += lbl.size(0)
                    val_correct[name] += (pred == lbl).sum().item()
        
        avg_val_loss = val_total_loss / len(val_loader.dataset)
        
        current_val_accs = []
        for name in ['geometric', 'natural', 'flower', 'handle']:
            acc = val_correct[name] / val_total[name]
            val_accs[name].append(acc)
            current_val_accs.append(acc)
        
        # 根据验证准确率动态调整任务权重（从第二个epoch开始）
        if epoch > 0 and use_dynamic_weights:
            if weight_adjust_method == 'accuracy':
                task_weights = dynamic_task_weight(current_val_accs)
            elif weight_adjust_method == 'hybrid':
                # acc_weights为动态任务权重，task_weights为包含acc_weights的历史状态，实现平滑过渡
                acc_weights = dynamic_task_weight(current_val_accs)
                task_weights = [0.9 * a + 0.1 * s for a, s in zip(task_weights, acc_weights)]
        
        epoch_time = time.time() - start_time
        
        # 打印日志
        print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s')
        print(f'  Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'  Task Weights: geo={task_weights[0]:.4f}, nat={task_weights[1]:.4f}, flo={task_weights[2]:.4f}, han={task_weights[3]:.4f}')
        for name in ['geometric', 'natural', 'flower', 'handle']:
            print(f'  {name}: Train Acc={train_accs[name][-1]:.4f}, Val Acc={val_accs[name][-1]:.4f}')
        
        # 计算平均验证准确率作为早停指标
        avg_val_acc = sum(current_val_accs) / 4
        
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'  Best model saved with avg val acc: {best_acc:.4f}')
        
        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)
    
    return model, train_losses, val_losses, train_accs, val_accs

# 测试函数
def test_model(model, test_loader):
    model.eval()
    
    # 存储四个任务的预测结果
    all_preds = {'geometric': [], 'natural': [], 'flower': [], 'handle': []}
    all_labels = {'geometric': [], 'natural': [], 'flower': [], 'handle': []}
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            geo_labels, nat_labels, flo_labels, han_labels = [l.to(DEVICE) for l in labels]
            
            geo_out, nat_out, flo_out, han_out = model(images)
            
            # 获取预测结果
            for name, out, lbl in zip(['geometric', 'natural', 'flower', 'handle'],
                                     [geo_out, nat_out, flo_out, han_out],
                                     [geo_labels, nat_labels, flo_labels, han_labels]):
                _, pred = torch.max(out.data, 1)
                
                all_preds[name].extend(pred.cpu().numpy())
                all_labels[name].extend(lbl.cpu().numpy())
    
    # 打印每个任务的准确率
    print('\n=== 各任务单独准确率 ===')
    task_accuracies = {}
    for name in ['geometric', 'natural', 'flower', 'handle']:
        acc = accuracy_score(all_labels[name], all_preds[name])
        task_accuracies[name] = acc
        print(f'{name}: {acc:.4f}')
    
    return task_accuracies

# 绘制训练曲线
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    from datetime import datetime  # 加上时间模块
    # 生成当前时间字符串
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 拼接文件名
    save_path = f'multitask_training_curves_resnet_{current_time}.png'
    
    plt.figure(figsize=(16, 8))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    for name in ['geometric', 'natural', 'flower', 'handle']:
        plt.plot(train_accs[name], label=f'{name} Train')
        plt.plot(val_accs[name], label=f'{name} Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)  # 使用带时间的文件名
    print(f"训练曲线已保存：{save_path}")
    plt.show()

def load_processed_datasets():
    process_dir = 'process'
    
    if not os.path.exists(process_dir):
        raise ValueError(f'process文件夹不存在: {os.path.abspath(process_dir)}')
    
    cn_files = []
    for file in glob.glob(os.path.join(process_dir, '*.parquet')):
        filename = os.path.basename(file)
        if filename.lower().startswith('cn') and '-new' in filename.lower():
            cn_files.append(file)
    
    print(f'找到 {len(cn_files)} 个处理后的CN数据集文件:')
    for f in cn_files:
        print(f'  - {f}')
    
    dfs = []
    for file in cn_files:
        df = pq.read_table(file).to_pandas()
        dfs.append(df)
        print(f'  已加载 {os.path.basename(file)}: {len(df)} 条记录')
    
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f'\n合并后总记录数: {len(combined_df)}')
    
    return combined_df

def filter_by_id_range(df, start_id, end_id):
    def extract_id_number(id_str):
        match = re.match(r'JN(\d+)', str(id_str))
        if match:
            return int(match.group(1))
        return None
    
    start_num = extract_id_number(start_id)
    end_num = extract_id_number(end_id)
    
    print(f'\n筛选ID范围: {start_id} ({start_num}) - {end_id} ({end_num})')
    
    df['id_num'] = df['id'].apply(extract_id_number)
    filtered_df = df[(df['id_num'] >= start_num) & (df['id_num'] <= end_num)]
    filtered_df = filtered_df.drop('id_num', axis=1)
    
    print(f'筛选前记录数: {len(df)}')
    print(f'筛选后记录数: {len(filtered_df)}')
    
    return filtered_df

def main():
    global GEOMETRIC_CLASS_NAMES, NATURAL_CLASS_NAMES, FLOWER_CLASS_NAMES, HANDLE_CLASS_NAMES
    global geometric_name_to_idx, natural_name_to_idx, flower_name_to_idx, handle_name_to_idx
    global NUM_GEOMETRIC_CLASSES, NUM_NATURAL_CLASSES, NUM_FLOWER_CLASSES, NUM_HANDLE_CLASSES
    
    print('Loading processed CN datasets...')
    combined_df = load_processed_datasets()
    
    # ID范围筛选
    combined_df = filter_by_id_range(combined_df, ID_START, ID_END)
    
    # 清理各类型字段的制表符
    print('\n清理类型字段中的制表符...')
    for col in ['geometric shape type', 'natural shape type', 'flower type', 'handle type']:
        combined_df[col] = combined_df[col].str.rstrip('\t')
    
    # 获取四个类型的类别信息
    print('\n=== 各类型类别分布 ===')
    
    # geometric shape type（去除样本数≤3的类别）
    geo_dist = combined_df['geometric shape type'].value_counts()
    rare_geo = geo_dist[geo_dist <= 3].index.tolist()
    if rare_geo:
        print(f'\ngeometric shape type 稀有类别（样本数≤3）: {rare_geo}')
        combined_df = combined_df[~combined_df['geometric shape type'].isin(rare_geo)]
        geo_dist = combined_df['geometric shape type'].value_counts()
    GEOMETRIC_CLASS_NAMES = geo_dist.index.tolist()
    NUM_GEOMETRIC_CLASSES = len(GEOMETRIC_CLASS_NAMES)
    geometric_name_to_idx = {name: idx for idx, name in enumerate(GEOMETRIC_CLASS_NAMES)}
    print(f'\ngeometric shape type ({NUM_GEOMETRIC_CLASSES}类):')
    print(geo_dist)
    
    # natural shape type
    nat_dist = combined_df['natural shape type'].value_counts()
    NATURAL_CLASS_NAMES = nat_dist.index.tolist()
    NUM_NATURAL_CLASSES = len(NATURAL_CLASS_NAMES)
    natural_name_to_idx = {name: idx for idx, name in enumerate(NATURAL_CLASS_NAMES)}
    print(f'\nnatural shape type ({NUM_NATURAL_CLASSES}类):')
    print(nat_dist)
    
    # flower type
    flo_dist = combined_df['flower type'].value_counts()
    FLOWER_CLASS_NAMES = flo_dist.index.tolist()
    NUM_FLOWER_CLASSES = len(FLOWER_CLASS_NAMES)
    flower_name_to_idx = {name: idx for idx, name in enumerate(FLOWER_CLASS_NAMES)}
    print(f'\nflower type ({NUM_FLOWER_CLASSES}类):')
    print(flo_dist)
    
    # handle type
    han_dist = combined_df['handle type'].value_counts()
    HANDLE_CLASS_NAMES = han_dist.index.tolist()
    NUM_HANDLE_CLASSES = len(HANDLE_CLASS_NAMES)
    handle_name_to_idx = {name: idx for idx, name in enumerate(HANDLE_CLASS_NAMES)}
    print(f'\nhandle type ({NUM_HANDLE_CLASSES}类):')
    print(han_dist)
    
    print(f'\n剩余样本数: {len(combined_df)}')
    
    # 划分数据集
    try:
        train_df, test_df = train_test_split(combined_df, test_size=TEST_SIZE, random_state=42, 
                                            stratify=combined_df['geometric shape type'])
        print('\n使用分层抽样划分训练集和测试集')
    except ValueError:
        train_df, test_df = train_test_split(combined_df, test_size=TEST_SIZE, random_state=42)
        print('\n分层抽样失败，使用随机划分')
    
    try:
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, 
                                           stratify=train_df['geometric shape type'])
        print('使用分层抽样划分验证集')
    except ValueError:
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
        print('随机划分验证集')
    
    print(f'\n训练集: {len(train_df)} 条')
    print(f'验证集: {len(val_df)} 条')
    print(f'测试集: {len(test_df)} 条')
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集和数据加载器
    train_dataset = TeapotDataset(train_df, transform=train_transform)
    val_dataset = TeapotDataset(val_df, transform=val_test_transform)
    test_dataset = TeapotDataset(test_df, transform=val_test_transform)
    
    num_workers = min(4, os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    # 创建多任务模型
    print(f'\nCreating Multi-Task SE-ResNet-34 model...')
    model = MultiTaskSEResNet34(
        num_geometric=NUM_GEOMETRIC_CLASSES,
        num_natural=NUM_NATURAL_CLASSES,
        num_flower=NUM_FLOWER_CLASSES,
        num_handle=NUM_HANDLE_CLASSES
    ).to(DEVICE)
    print(f'Model initialized on {DEVICE}')
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    
    # 训练模型
    print('\nStarting training...')
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS
    )
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    # 加载最佳模型进行测试
    print('\nLoading best model for testing...')
    model.load_state_dict(torch.load('multitask_best.pth'))
    task_accuracies = test_model(model, test_loader)
    
    # 保存最终模型
    torch.save(model.state_dict(), 'multitask_final.pth')
    print('\nFinal model saved as: multitask_final.pth')

if __name__ == '__main__':
    main()