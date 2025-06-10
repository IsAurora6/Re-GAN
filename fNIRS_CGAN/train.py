import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from generator import Generator
from discriminator import Discriminator
from regan import Regan_training
from dataset import FNIRSDataset
import os
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import Counter

# 配置参数
DATA_ROOT = '../your_fnirs_data_dir'  # 修改为你的数据路径
BATCH_SIZE = 16
NOISE_DIM = 100
LABEL_DIM = 3  # ADHD/HC/ASD
EPOCHS_GAN = 100  # WGAN-GP训练轮数
EPOCHS_CLS = 250   # 分类模型训练轮数
LR = 2e-4
SPARSITY = 0.3  # 稀疏化比例
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LAMBDA_GP = 10  # 梯度惩罚系数

# 分类模型（简单CNN）
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*20*2, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = torch.ones(d_interpolates.size(), device=real_samples.device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# 1. 加载原始数据
full_dataset = FNIRSDataset(DATA_ROOT)
X = full_dataset.X.numpy()
y = full_dataset.y.numpy()

# 2. 4:1 划分训练集和测试集
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# 3. 训练集10%为验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=42
)

# 4. 五折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"=== Fold {fold+1}/5 ===")
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    # 统计每类数量
    counter = Counter(y_fold_train)
    max_count = max(counter.values())
    print("各类别样本数：", counter, "目标增强到：", max_count)
    # 5. 用WCGAN-GP增强每类
    X_aug, y_aug = [X_fold_train], [y_fold_train]
    for label in range(LABEL_DIM):
        n_to_generate = max_count - counter.get(label, 0)
        if n_to_generate > 0:
            print(f"增强类别{label}，生成{n_to_generate}个样本...")
            # 训练WGAN-GP生成器/判别器
            # 只用该类别样本训练
            real_data = X_fold_train[y_fold_train == label]
            real_data = torch.tensor(real_data, dtype=torch.float32).unsqueeze(1).to(DEVICE)  # (N,1,160,10)
            real_labels = torch.full((real_data.size(0),), label, dtype=torch.long, device=DEVICE)
            G = Generator(NOISE_DIM, LABEL_DIM).to(DEVICE)
            D = Discriminator(LABEL_DIM).to(DEVICE)
            G_regan = Regan_training(G, SPARSITY)
            D_regan = Regan_training(D, SPARSITY)
            optimizer_G = optim.Adam(G_regan.parameters(), lr=LR, betas=(0.5, 0.9))
            optimizer_D = optim.Adam(D_regan.parameters(), lr=LR, betas=(0.5, 0.9))
            for epoch in range(EPOCHS_GAN):
                idx = torch.randperm(real_data.size(0))[:BATCH_SIZE]
                real_batch = real_data[idx]
                label_batch = real_labels[idx]
                # 1. 训练判别器
                noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=DEVICE)
                fake_data = G_regan(noise, label_batch)
                D_regan.turn_training_mode('sparse')
                optimizer_D.zero_grad()
                real_validity = D_regan(real_batch, label_batch)
                fake_validity = D_regan(fake_data.detach(), label_batch)
                gp = compute_gradient_penalty(D_regan, real_batch, fake_data.detach(), label_batch)
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gp
                d_loss.backward()
                D_regan.apply_masks()
                optimizer_D.step()
                # 2. 训练生成器
                if epoch % 5 == 0:
                    G_regan.turn_training_mode('sparse')
                    optimizer_G.zero_grad()
                    noise = torch.randn(BATCH_SIZE, NOISE_DIM, device=DEVICE)
                    gen_data = G_regan(noise, label_batch)
                    g_loss = -torch.mean(D_regan(gen_data, label_batch))
                    g_loss.backward()
                    G_regan.apply_masks()
                    optimizer_G.step()
            # 生成增强样本
            G_regan.eval()
            with torch.no_grad():
                n_gen = n_to_generate
                gen_noise = torch.randn(n_gen, NOISE_DIM, device=DEVICE)
                gen_labels = torch.full((n_gen,), label, dtype=torch.long, device=DEVICE)
                gen_samples = G_regan(gen_noise, gen_labels).cpu().numpy().squeeze(1)
                X_aug.append(gen_samples)
                y_aug.append(np.full(n_gen, label, dtype=np.int64))
    # 合并增强数据
    X_train_final = np.concatenate(X_aug, axis=0)
    y_train_final = np.concatenate(y_aug, axis=0)
    print(f"增强后训练集样本数：{len(y_train_final)}")
    # 6. 训练分类模型
    train_dataset = TensorDataset(torch.tensor(X_train_final, dtype=torch.float32).unsqueeze(1), torch.tensor(y_train_final, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_fold_val, dtype=torch.float32).unsqueeze(1), torch.tensor(y_fold_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32).unsqueeze(1), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = SimpleClassifier(num_classes=LABEL_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    for epoch in range(EPOCHS_CLS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_cls_fold{fold+1}.pth')
        if epoch % 10 == 0:
            print(f"[Fold {fold+1}] Epoch {epoch} 分类验证准确率: {val_acc:.4f}")
    # 测试集评估
    model.load_state_dict(torch.load(f'best_cls_fold{fold+1}.pth'))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    test_acc = correct / total
    print(f"[Fold {fold+1}] 测试集准确率: {test_acc:.4f}")
    # 只做一折可去掉break
    # break
print('全部交叉验证完成！') 