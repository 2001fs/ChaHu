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
import matplotlib.pyplot as plt
import time
import pickle

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

# 模型层配置
LAYER1_BLOCK = 'se'
LAYER2_BLOCK = 'se'
LAYER3_BLOCK = 'inception'
LAYER4_BLOCK = 'se'
# ==================================================

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

# 原始 SE-ResNet BasicBlock
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

class LightInceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(LightInceptionModule, self).__init__()
        branch_channels = out_channels // 4

        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=5, padding=2, stride=stride, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.branch1x1(x)
        out2 = self.branch3x3(x)
        out3 = self.branch5x5(x)
        out4 = self.branch_pool(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

# SE-Inception-ResNet Block
class SEInceptionBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEInceptionBasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride

        self.inception1 = LightInceptionModule(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.inception2 = LightInceptionModule(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)

    def forward(self, x):
        residual = x

        out = self.inception1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.inception2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# 块类型选择辅助函数
def get_block_type(block_type):
    if block_type == 'inception':
        return SEInceptionBasicBlock
    elif block_type == 'se':
        return SEBasicBlock
    else:
        raise ValueError(f"Invalid block_type: {block_type}. Must be 'se' or 'inception'")

# 多任务可配置 SE-Inception-ResNet-34 模型
class MultiTaskConfigurableResNet34(nn.Module):
    def __init__(self, num_geometric, num_natural, num_flower, num_handle,
                 layer1_block='se', layer2_block='se', layer3_block='se', layer4_block='se'):
        super(MultiTaskConfigurableResNet34, self).__init__()
        self.inplanes = 64

        self.layer_config = {
            'layer1': layer1_block,
            'layer2': layer2_block,
            'layer3': layer3_block,
            'layer4': layer4_block
        }

        # 共享的卷积主干
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(get_block_type(layer1_block), 64, 3)
        self.layer2 = self._make_layer(get_block_type(layer2_block), 128, 4, stride=2)
        self.layer3 = self._make_layer(get_block_type(layer3_block), 256, 6, stride=2)
        self.layer4 = self._make_layer(get_block_type(layer4_block), 512, 3, stride=2)

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

        geo_out = self.fc_geometric(x)
        nat_out = self.fc_natural(x)
        flo_out = self.fc_flower(x)
        han_out = self.fc_handle(x)

        return geo_out, nat_out, flo_out, han_out

    def print_config(self):
        print("Model Configuration:")
        print(f"  layer1 (conv2_x): {self.layer_config['layer1']}")
        print(f"  layer2 (conv3_x): {self.layer_config['layer2']}")
        print(f"  layer3 (conv4_x): {self.layer_config['layer3']}")
        print(f"  layer4 (conv5_x): {self.layer_config['layer4']}")

# 根据验证准确率动态调整任务权重
def dynamic_task_weight(val_accs, base_weights=[0.25, 0.25, 0.25, 0.25]):
    inv_accs = [1 - acc for acc in val_accs]
    inv_accs = [w / sum(inv_accs) for w in inv_accs]
    weights = [0.7 * base + 0.3 * inv for base, inv in zip(base_weights, inv_accs)]
    return weights

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience=5,
               class_weights=None, use_dynamic_weights=True, weight_adjust_method='hybrid'):
    train_losses = []
    val_losses = []
    train_accs = {'geometric': [], 'natural': [], 'flower': [], 'handle': []}
    val_accs = {'geometric': [], 'natural': [], 'flower': [], 'handle': []}

    best_acc = 0.0
    best_model_path = 'model_save/multitask_best.pth'
    os.makedirs('model_save', exist_ok=True)

    task_weights = [0.25, 0.25, 0.25, 0.25]

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

        correct = {'geometric': 0, 'natural': 0, 'flower': 0, 'handle': 0}
        total = {'geometric': 0, 'natural': 0, 'flower': 0, 'handle': 0}

        for images, labels in train_loader:
            images = images.to(DEVICE)
            geo_labels, nat_labels, flo_labels, han_labels = [l.to(DEVICE) for l in labels]

            optimizer.zero_grad()

            geo_out, nat_out, flo_out, han_out = model(images)

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

            loss = (task_weights[0] * loss_geo +
                    task_weights[1] * loss_nat +
                    task_weights[2] * loss_flo +
                    task_weights[3] * loss_han)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

            for name, out, lbl in zip(['geometric', 'natural', 'flower', 'handle'],
                                     [geo_out, nat_out, flo_out, han_out],
                                     [geo_labels, nat_labels, flo_labels, han_labels]):
                _, pred = torch.max(out.data, 1)
                total[name] += lbl.size(0)
                correct[name] += (pred == lbl).sum().item()

        avg_loss = total_loss / len(train_loader.dataset)

        for name in ['geometric', 'natural', 'flower', 'handle']:
            train_accs[name].append(correct[name] / total[name])

        scheduler.step()

        model.eval()
        val_total_loss = 0.0
        val_correct = {'geometric': 0, 'natural': 0, 'flower': 0, 'handle': 0}
        val_total = {'geometric': 0, 'natural': 0, 'flower': 0, 'handle': 0}

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                geo_labels, nat_labels, flo_labels, han_labels = [l.to(DEVICE) for l in labels]

                geo_out, nat_out, flo_out, han_out = model(images)

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

        if epoch > 0 and use_dynamic_weights:
            if weight_adjust_method == 'accuracy':
                task_weights = dynamic_task_weight(current_val_accs)
            elif weight_adjust_method == 'hybrid':
                acc_weights = dynamic_task_weight(current_val_accs)
                task_weights = [0.9 * a + 0.1 * s for a, s in zip(task_weights, acc_weights)]

        epoch_time = time.time() - start_time

        print(f'Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s')
        print(f'  Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'  Task Weights: geo={task_weights[0]:.4f}, nat={task_weights[1]:.4f}, flo={task_weights[2]:.4f}, han={task_weights[3]:.4f}')
        for name in ['geometric', 'natural', 'flower', 'handle']:
            print(f'  {name}: Train Acc={train_accs[name][-1]:.4f}, Val Acc={val_accs[name][-1]:.4f}')

        avg_val_acc = sum(current_val_accs) / 4

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'  Best model saved with avg val acc: {best_acc:.4f}')

        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)

    return model, train_losses, val_losses, train_accs, val_accs

# 绘制训练曲线
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    from datetime import datetime
    os.makedirs('image_save', exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'image_save/multitask_training_curves_resnet_{current_time}.png'

    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    for name in ['geometric', 'natural', 'flower', 'handle']:
        plt.plot(train_accs[name], label=f'{name} Train')
        plt.plot(val_accs[name], label=f'{name} Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"训练曲线已保存：{save_path}")
    # plt.show()  # 服务器环境可注释

# 数据加载函数
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
    dfs = []
    for f in cn_files:
        df = pq.read_table(f).to_pandas()
        dfs.append(df)
        print(f'  已加载 {os.path.basename(f)}: {len(df)} 条记录')

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f'\n合并后总记录数: {len(combined_df)}')
    return combined_df

# ID筛选函数
def filter_by_id_range(df, start_id, end_id):
    def extract_id_number(id_str):
        match = re.match(r'JN(\d+)', str(id_str))
        return int(match.group(1)) if match else None

    start_num = extract_id_number(start_id)
    end_num = extract_id_number(end_id)
    print(f'\n筛选ID范围: {start_id} ({start_num}) - {end_id} ({end_num})')

    df['id_num'] = df['id'].apply(extract_id_number)
    filtered_df = df[(df['id_num'] >= start_num) & (df['id_num'] <= end_num)]
    filtered_df = filtered_df.drop('id_num', axis=1)
    print(f'筛选前记录数: {len(df)}')
    print(f'筛选后记录数: {len(filtered_df)}')
    return filtered_df

# 主训练流程
def main():
    global geometric_name_to_idx, natural_name_to_idx, flower_name_to_idx, handle_name_to_idx
    global NUM_GEOMETRIC_CLASSES, NUM_NATURAL_CLASSES, NUM_FLOWER_CLASSES, NUM_HANDLE_CLASSES

    print('Loading processed CN datasets...')
    combined_df = load_processed_datasets()
    combined_df = filter_by_id_range(combined_df, ID_START, ID_END)

    # 清理制表符
    for col in ['geometric shape type', 'natural shape type', 'flower type', 'handle type']:
        combined_df[col] = combined_df[col].str.rstrip('\t')

    # 构建类别映射
    geo_dist = combined_df['geometric shape type'].value_counts()
    rare_geo = geo_dist[geo_dist <= 3].index.tolist()
    if rare_geo:
        combined_df = combined_df[~combined_df['geometric shape type'].isin(rare_geo)]
        geo_dist = combined_df['geometric shape type'].value_counts()
    GEOMETRIC_CLASS_NAMES = geo_dist.index.tolist()
    NUM_GEOMETRIC_CLASSES = len(GEOMETRIC_CLASS_NAMES)
    geometric_name_to_idx = {name: idx for idx, name in enumerate(GEOMETRIC_CLASS_NAMES)}

    nat_dist = combined_df['natural shape type'].value_counts()
    NATURAL_CLASS_NAMES = nat_dist.index.tolist()
    NUM_NATURAL_CLASSES = len(NATURAL_CLASS_NAMES)
    natural_name_to_idx = {name: idx for idx, name in enumerate(NATURAL_CLASS_NAMES)}

    flo_dist = combined_df['flower type'].value_counts()
    FLOWER_CLASS_NAMES = flo_dist.index.tolist()
    NUM_FLOWER_CLASSES = len(FLOWER_CLASS_NAMES)
    flower_name_to_idx = {name: idx for idx, name in enumerate(FLOWER_CLASS_NAMES)}

    han_dist = combined_df['handle type'].value_counts()
    HANDLE_CLASS_NAMES = han_dist.index.tolist()
    NUM_HANDLE_CLASSES = len(HANDLE_CLASS_NAMES)
    handle_name_to_idx = {name: idx for idx, name in enumerate(HANDLE_CLASS_NAMES)}

    print(f'\n剩余样本数: {len(combined_df)}')

    # 数据集划分（完全保留原抽样逻辑）
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

    # 保存测试集（供测试脚本使用）
    os.makedirs('data_split', exist_ok=True)
    test_df.to_parquet('data_split/test_dataset.parquet', index=False)
    print("测试集已保存至: data_split/test_dataset.parquet")

    # 保存类别映射
    label_mapping = {
        'geometric': geometric_name_to_idx,
        'natural': natural_name_to_idx,
        'flower': flower_name_to_idx,
        'handle': handle_name_to_idx,
        'num_classes': (NUM_GEOMETRIC_CLASSES, NUM_NATURAL_CLASSES, NUM_FLOWER_CLASSES, NUM_HANDLE_CLASSES)
    }
    with open('model_save/label_mapping.pkl', 'wb') as f:
        pickle.dump(label_mapping, f)
    print("标签映射已保存至: model_save/label_mapping.pkl")

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

    # 数据集加载器
    train_dataset = TeapotDataset(train_df, transform=train_transform)
    val_dataset = TeapotDataset(val_df, transform=val_test_transform)

    num_workers = min(4, os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 模型初始化
    model = MultiTaskConfigurableResNet34(
        num_geometric=NUM_GEOMETRIC_CLASSES,
        num_natural=NUM_NATURAL_CLASSES,
        num_flower=NUM_FLOWER_CLASSES,
        num_handle=NUM_HANDLE_CLASSES,
        layer1_block=LAYER1_BLOCK,
        layer2_block=LAYER2_BLOCK,
        layer3_block=LAYER3_BLOCK,
        layer4_block=LAYER4_BLOCK
    ).to(DEVICE)

    model.print_config()
    print(f'Model initialized on {DEVICE}')

    # 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 训练
    print('\nStarting training...')
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS
    )

    # 绘图
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # 保存最终模型
    torch.save(model.state_dict(), 'model_save/multitask_final.pth')
    print('\n训练完成！最终模型已保存: model_save/multitask_final.pth')

if __name__ == '__main__':
    main()