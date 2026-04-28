import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import io
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib.font_manager import FontProperties
import urllib.request

# -------------------------- 1. 自动下载中文字体，保证中文显示 --------------------------
font_path = "./SimHei.ttf"
if not os.path.exists(font_path):
    print("正在下载中文字体（黑体）...")
    url = "https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220716-mmclassification/dataset/SimHei.ttf"
    urllib.request.urlretrieve(url, font_path)
    print("字体下载完成")
chinese_font = FontProperties(fname=font_path)

# 屏蔽警告
import warnings
warnings.filterwarnings("ignore")

# ==================== 检查文件 ====================
if not os.path.exists('model_save/label_mapping.pkl'):
    raise FileNotFoundError("请先运行 train.py")
if not os.path.exists('data_split/test_dataset.parquet'):
    raise FileNotFoundError("请先运行 train.py")
if not os.path.exists('model_save/multitask_best.pth'):
    raise FileNotFoundError("请先运行 train.py")

# ==================== 全局参数 ====================
BATCH_SIZE = 64
IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LAYER1_BLOCK = 'se'
LAYER2_BLOCK = 'se'
LAYER3_BLOCK = 'inception'
LAYER4_BLOCK = 'se'

# ==================== 加载标签映射 ====================
with open('model_save/label_mapping.pkl', 'rb') as f:
    label_mapping = pickle.load(f)

geometric_name_to_idx = label_mapping['geometric']
natural_name_to_idx   = label_mapping['natural']
flower_name_to_idx    = label_mapping['flower']
handle_name_to_idx    = label_mapping['handle']

geometric_idx_to_name = {v: k for k, v in geometric_name_to_idx.items()}
natural_idx_to_name   = {v: k for k, v in natural_name_to_idx.items()}
flower_idx_to_name    = {v: k for k, v in flower_name_to_idx.items()}
handle_idx_to_name    = {v: k for k, v in handle_name_to_idx.items()}

# 获取「其他」类别ID
other_geo_id = geometric_name_to_idx["其他"]
other_nat_id = natural_name_to_idx["其他"]
other_flw_id = flower_name_to_idx["其他"]
other_hdl_id = handle_name_to_idx["其他"]

NUM_GEOMETRIC_CLASSES, NUM_NATURAL_CLASSES, NUM_FLOWER_CLASSES, NUM_HANDLE_CLASSES = label_mapping['num_classes']

# ==================== 数据集 ====================
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

        geo_label = geometric_name_to_idx.get(row['geometric shape type'], 0)
        nat_label = natural_name_to_idx.get(row['natural shape type'], 0)
        flo_label = flower_name_to_idx.get(row['flower type'], 0)
        han_label = handle_name_to_idx.get(row['handle type'], 0)

        if self.transform:
            image = self.transform(image)
        return image, (geo_label, nat_label, flo_label, han_label), idx

# ==================== 模型结构 ====================
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

class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class LightInceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        branch_channels = out_channels // 4
        self.branch1x1 = nn.Sequential(nn.Conv2d(in_channels, branch_channels,1,stride,bias=False),nn.BatchNorm2d(branch_channels),nn.ReLU(True))
        self.branch3x3 = nn.Sequential(nn.Conv2d(in_channels, branch_channels,3,stride,1,bias=False),nn.BatchNorm2d(branch_channels),nn.ReLU(True))
        self.branch5x5 = nn.Sequential(nn.Conv2d(in_channels, branch_channels,5,stride,2,bias=False),nn.BatchNorm2d(branch_channels),nn.ReLU(True))
        self.branch_pool = nn.Sequential(nn.AvgPool2d(3,stride,1),nn.Conv2d(in_channels, branch_channels,1,bias=False),nn.BatchNorm2d(branch_channels),nn.ReLU(True))
    def forward(self, x):
        return torch.cat([self.branch1x1(x),self.branch3x3(x),self.branch5x5(x),self.branch_pool(x)],1)

class SEInceptionBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.downsample = downsample
        self.inception1 = LightInceptionModule(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(True)
        self.inception2 = LightInceptionModule(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.inception1(x)))
        out = self.se(self.bn2(self.inception2(out)))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

def get_block_type(block_type):
    return SEInceptionBasicBlock if block_type == 'inception' else SEBasicBlock

class MultiTaskConfigurableResNet34(nn.Module):
    def __init__(self, num_geometric, num_natural, num_flower, num_handle,
                 layer1_block='se', layer2_block='se', layer3_block='se', layer4_block='se'):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(get_block_type(layer1_block), 64, 3)
        self.layer2 = self._make_layer(get_block_type(layer2_block), 128, 4, 2)
        self.layer3 = self._make_layer(get_block_type(layer3_block), 256, 6, 2)
        self.layer4 = self._make_layer(get_block_type(layer4_block), 512, 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_geometric = nn.Linear(512, num_geometric)
        self.fc_natural   = nn.Linear(512, num_natural)
        self.fc_flower    = nn.Linear(512, num_flower)
        self.fc_handle    = nn.Linear(512, num_handle)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*block.expansion,1,stride,bias=False),nn.BatchNorm2d(planes*block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc_geometric(x), self.fc_natural(x), self.fc_flower(x), self.fc_handle(x)

# ==================== 绘图 & 反归一化 ====================
def denormalize(tensor):
    tensor = tensor.cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img = tensor * std + mean
    img = img.permute(1,2,0).numpy()
    return np.clip(img, 0, 1)

def plot_best_image_with_probs(img_tensor, probs, true_label, pred_label, idx_to_name, task_name):
    img = denormalize(img_tensor)
    probs = probs.cpu().numpy()
    class_names = [idx_to_name[i] for i in range(len(probs))]
    true_cls = idx_to_name[true_label]
    pred_cls = idx_to_name[pred_label]

    # -------------------------- 核心调整 --------------------------
    plt.figure(figsize=(20, 20))  

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'真实标签：{true_cls}\n预测标签：{pred_cls}',
              fontproperties=chinese_font, fontsize=14)

    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(class_names))

    plt.barh(y_pos, probs, color='#4285F4', height=0.4)

    # 调整 y 轴间距
    plt.yticks(y_pos, class_names, fontproperties=chinese_font, fontsize=10)
    plt.ylim(-0.5, len(class_names) - 0.5)  # 强制上下边界，让间距均匀

    # 加大整体间距
    plt.tight_layout(pad=8.0)

    plt.xlabel('类别概率', fontproperties=chinese_font, fontsize=12)
    plt.title('各类别预测概率分布', fontproperties=chinese_font, fontsize=14)

    os.makedirs('vis_results', exist_ok=True)
    save_path = f'vis_results/best_correct_{task_name}.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'✅ 【{task_name}】已保存（间距已拉大）')

# ==================== 筛选：真实=预测 && 不是其他类 ====================
def find_target_best_images(model, test_loader):
    model.eval()
    best = {
        "geometric": {"max_p": 0.0, "img": None, "true": -1, "pred": -1, "probs": None},
        "natural":   {"max_p": 0.0, "img": None, "true": -1, "pred": -1, "probs": None},
        "flower":    {"max_p": 0.0, "img": None, "true": -1, "pred": -1, "probs": None},
        "handle":    {"max_p": 0.0, "img": None, "true": -1, "pred": -1, "probs": None},
    }

    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs = imgs.to(DEVICE)
            gt, nt, ft, ht = [x.to(DEVICE) for x in labels]
            go, no, fo, ho = model(imgs)

            gp = torch.softmax(go, dim=1)
            np_ = torch.softmax(no, dim=1)
            fp = torch.softmax(fo, dim=1)
            hp = torch.softmax(ho, dim=1)

            for i in range(imgs.size(0)):
                # 1. geometric：预测正确 + 非其他
                t_geo = gt[i].item()
                p_geo = gp[i].argmax().item()
                if t_geo == p_geo and t_geo != other_geo_id:
                    cur_p = gp[i, p_geo].item()
                    if cur_p > best["geometric"]["max_p"]:
                        best["geometric"]["max_p"] = cur_p
                        best["geometric"]["img"]   = imgs[i]
                        best["geometric"]["true"]  = t_geo
                        best["geometric"]["pred"]  = p_geo
                        best["geometric"]["probs"] = gp[i]

                # 2. natural：预测正确 + 非其他
                t_nat = nt[i].item()
                p_nat = np_[i].argmax().item()
                if t_nat == p_nat and t_nat != other_nat_id:
                    cur_p = np_[i, p_nat].item()
                    if cur_p > best["natural"]["max_p"]:
                        best["natural"]["max_p"] = cur_p
                        best["natural"]["img"]   = imgs[i]
                        best["natural"]["true"]  = t_nat
                        best["natural"]["pred"]  = p_nat
                        best["natural"]["probs"] = np_[i]

                # 3. flower：预测正确 + 非其他
                t_flw = ft[i].item()
                p_flw = fp[i].argmax().item()
                if t_flw == p_flw and t_flw != other_flw_id:
                    cur_p = fp[i, p_flw].item()
                    if cur_p > best["flower"]["max_p"]:
                        best["flower"]["max_p"] = cur_p
                        best["flower"]["img"]   = imgs[i]
                        best["flower"]["true"]  = t_flw
                        best["flower"]["pred"]  = p_flw
                        best["flower"]["probs"] = fp[i]

                # 4. handle：预测正确 + 非其他
                t_hdl = ht[i].item()
                p_hdl = hp[i].argmax().item()
                if t_hdl == p_hdl and t_hdl != other_hdl_id:
                    cur_p = hp[i, p_hdl].item()
                    if cur_p > best["handle"]["max_p"]:
                        best["handle"]["max_p"] = cur_p
                        best["handle"]["img"]   = imgs[i]
                        best["handle"]["true"]  = t_hdl
                        best["handle"]["pred"]  = p_hdl
                        best["handle"]["probs"] = hp[i]

    # 绘图
    task_list = [
        ("geometric", geometric_idx_to_name),
        ("natural",   natural_idx_to_name),
        ("flower",    flower_idx_to_name),
        ("handle",    handle_idx_to_name)
    ]
    for task, idx_map in task_list:
        info = best[task]
        if info["img"] is not None:
            plot_best_image_with_probs(
                info["img"], info["probs"],
                info["true"], info["pred"],
                idx_map, task
            )
        else:
            print(f"❌ 【{task}】未找到满足条件样本")

# ==================== 测试准确率 ====================
def test_model(model, loader):
    model.eval()
    preds = {'g':[],'n':[],'f':[],'h':[]}
    gts   = {'g':[],'n':[],'f':[],'h':[]}
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(DEVICE)
            g,n,f,h = [x.to(DEVICE) for x in labels]
            go,no,fo,ho = model(imgs)
            preds['g'].extend(go.argmax(1).cpu().numpy())
            preds['n'].extend(no.argmax(1).cpu().numpy())
            preds['f'].extend(fo.argmax(1).cpu().numpy())
            preds['h'].extend(ho.argmax(1).cpu().numpy())
            gts['g'].extend(g.cpu().numpy())
            gts['n'].extend(n.cpu().numpy())
            gts['f'].extend(f.cpu().numpy())
            gts['h'].extend(h.cpu().numpy())
    print("\n 测试集各任务准确率：")
    print(f"几何形状：{accuracy_score(gts['g'],preds['g']):.3f}")
    print(f"自然形状：{accuracy_score(gts['n'],preds['n']):.3f}")
    print(f"花卉类别：{accuracy_score(gts['f'],preds['f']):.3f}")
    print(f"把手类型：{accuracy_score(gts['h'],preds['h']):.3f}")

# ==================== 主函数 ====================
def main():
    test_df = pd.read_parquet('data_split/test_dataset.parquet')
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = TeapotDataset(test_df, transform)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = MultiTaskConfigurableResNet34(
        NUM_GEOMETRIC_CLASSES, NUM_NATURAL_CLASSES, NUM_FLOWER_CLASSES, NUM_HANDLE_CLASSES,
        LAYER1_BLOCK, LAYER2_BLOCK, LAYER3_BLOCK, LAYER4_BLOCK
    ).to(DEVICE)
    model.load_state_dict(torch.load('model_save/multitask_best.pth', map_location=DEVICE))
    print("✅ 模型加载完成")

    test_model(model, dl)
    print("\n 筛选条件：真实标签 == 预测标签 且 标签不为【其他】")
    find_target_best_images(model, dl)
    print("\n 执行完毕，图片保存至 vis_results 文件夹")

if __name__ == '__main__':
    main()