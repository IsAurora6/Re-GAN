import torch
import torch.nn as nn

class Regan_training(nn.Module):
    """
    Re-GAN稀疏化训练模块
    用于包裹生成器或判别器，实现动态剪枝（稀疏化）训练
    """
    def __init__(self, model, sparsity, train_on_sparse=False):
        super(Regan_training, self).__init__()
        self.model = model  # 需要稀疏化的网络（生成器或判别器）
        self.sparsity = sparsity  # 稀疏比例，比如0.3表示剪掉30%的参数
        self.train_on_sparse = train_on_sparse
        self.layers = []  # 存储所有需要剪枝的参数
        self.masks = []   # 存储每个参数的"剪枝掩码"
        layers = list(self.model.named_parameters())
        for i in range(0, len(layers)):
            w = layers[i]
            self.layers.append(w[1])
        self.reset_masks()  # 初始化掩码，全为1（不剪枝）

    def reset_masks(self):
        """
        重置所有掩码为1（全部参数保留）
        """
        self.masks = []
        for w in self.layers:
            mask_w = torch.ones_like(w, dtype=bool)
            self.masks.append(mask_w)
        return self.masks

    def update_masks(self):
        """
        动态更新掩码：将绝对值最小的sparsity比例参数标记为True（剪掉）
        """
        for i, w in enumerate(self.layers):
            q_w = torch.quantile(torch.abs(w), q=self.sparsity)
            mask_w = torch.where(torch.abs(w) < q_w, True, False)
            self.masks[i] = mask_w

    def turn_training_mode(self, mode):
        """
        切换稀疏/稠密训练模式
        mode='dense'：不剪枝
        mode='sparse'：剪枝并更新掩码
        """
        if mode == 'dense':
            self.train_on_sparse = False
        else:
            self.train_on_sparse = True
            self.update_masks()

    def apply_masks(self):
        """
        应用掩码：将被剪枝的参数和梯度都置为0
        """
        for w, mask_w in zip(self.layers, self.masks):
            w.data[mask_w] = 0
            if w.grad is not None:
                w.grad.data[mask_w] = 0

    def forward(self, *args, **kwargs):
        """
        前向传播，直接调用原始网络
        """
        return self.model(*args, **kwargs) 