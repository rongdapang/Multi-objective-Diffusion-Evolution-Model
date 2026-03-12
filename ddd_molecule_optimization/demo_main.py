"""
DDD算法演示版本 - 不依赖RDKit

本演示版本使用简化的分子模型来展示DDD算法的工作原理。
实际使用时需要安装RDKit来处理真实的分子数据。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)


class SimplifiedMoleculeModel:
    """
    简化的分子模型，用于演示
    实际应用中应使用RDKit处理真实分子
    """
    
    def __init__(self, latent_dim: int = 32):
        self.latent_dim = latent_dim
        # 预定义一些"分子模板"
        self.templates = self._generate_templates(100)
        
    def _generate_templates(self, n: int) -> List[Dict]:
        """生成模拟的分子模板"""
        templates = []
        for i in range(n):
            z = np.random.randn(self.latent_dim)
            # 模拟分子性质
            qed = np.random.beta(2, 2)  # QED在0-1之间
            logp = np.random.normal(2, 1.5)  # logP
            mw = np.random.uniform(100, 500)  # 分子量
            templates.append({
                'z': z,
                'qed': qed,
                'logp': logp,
                'mw': mw,
                'id': f'MOL_{i:03d}'
            })
        return templates
    
    def decode(self, z: np.ndarray) -> Dict:
        """
        解码潜在向量到分子
        实际应用中应使用RDKit解码SMILES
        """
        # 找到最近的模板
        distances = [np.linalg.norm(z - t['z']) for t in self.templates]
        nearest_idx = np.argmin(distances)
        template = self.templates[nearest_idx]
        
        # 添加一些噪声来模拟解码的不确定性
        noise_scale = min(0.1 * distances[nearest_idx], 1.0)
        
        return {
            'id': template['id'],
            'qed': np.clip(template['qed'] + np.random.randn() * noise_scale * 0.1, 0, 1),
            'logp': template['logp'] + np.random.randn() * noise_scale * 0.5,
            'mw': np.clip(template['mw'] + np.random.randn() * noise_scale * 10, 50, 1000),
            'valid': True  # 简化版本：所有都有效
        }
    
    def encode(self, molecule_id: str) -> np.ndarray:
        """编码分子到潜在向量"""
        for template in self.templates:
            if template['id'] == molecule_id:
                return template['z']
        return np.random.randn(self.latent_dim)


class ConditionalDiffusionModel(nn.Module):
    """条件扩散模型"""
    
    def __init__(self, z_dim: int = 32, cond_dim: int = 3, 
                 num_steps: int = 20, device: str = 'cpu'):
        super().__init__()
        self.z_dim = z_dim
        self.cond_dim = cond_dim
        self.num_steps = num_steps
        self.device = device
        
        # 时间嵌入
        self.time_embed = nn.Embedding(num_steps, 64)
        
        # 条件投影
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # 噪声预测网络
        self.net = nn.Sequential(
            nn.Linear(z_dim + 64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim)
        )
        
        # 扩散参数
        self.betas = torch.linspace(0.0001, 0.02, num_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.is_trained = False
        
    def forward(self, z_t, cond, t):
        """预测噪声"""
        t_emb = self.time_embed(t)
        c_emb = self.cond_proj(cond)
        x = torch.cat([z_t, t_emb, c_emb], dim=1)
        return self.net(x)
    
    def train_model(self, z_train, cond_train, epochs=50, lr=1e-3):
        """训练模型"""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        z_tensor = torch.FloatTensor(z_train).to(self.device)
        cond_tensor = torch.FloatTensor(cond_train).to(self.device)
        
        # 归一化
        self.z_mean = z_tensor.mean(dim=0)
        self.z_std = z_tensor.std(dim=0) + 1e-8
        z_norm = (z_tensor - self.z_mean) / self.z_std
        
        self.cond_mean = cond_tensor.mean(dim=0)
        self.cond_std = cond_tensor.std(dim=0) + 1e-8
        cond_norm = (cond_tensor - self.cond_mean) / self.cond_std
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(10):  # 每个epoch迭代10次
                # 随机时间步
                t = torch.randint(0, self.num_steps, (len(z_norm),)).to(self.device)
                
                # 前向扩散
                noise = torch.randn_like(z_norm)
                z_t = self.sqrt_alphas_cumprod[t].view(-1, 1) * z_norm + \
                      self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1) * noise
                
                # 预测噪声
                noise_pred = self(z_t, cond_norm, t)
                loss = F.mse_loss(noise_pred, noise)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / 10)
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {losses[-1]:.6f}")
        
        self.is_trained = True
        return losses
    
    def sample(self, conditions, num_samples=None):
        """采样"""
        if not self.is_trained:
            if conditions.ndim == 1:
                return np.random.randn(num_samples if num_samples else 1, self.z_dim)
            return np.random.randn(len(conditions), self.z_dim)
        
        if conditions.ndim == 1:
            conditions = np.tile(conditions, (num_samples if num_samples else 1, 1))
        
        cond = torch.FloatTensor(conditions).to(self.device)
        cond_norm = (cond - self.cond_mean.to(self.device)) / self.cond_std.to(self.device)
        
        z = torch.randn(len(conditions), self.z_dim).to(self.device)
        
        self.eval()
        with torch.no_grad():
            for t in reversed(range(self.num_steps)):
                t_batch = torch.full((len(conditions),), t, device=self.device, dtype=torch.long)
                noise_pred = self(z, cond_norm, t_batch)
                
                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alphas_cumprod[t]
                beta_t = self.betas[t]
                
                mean = (z - beta_t / torch.sqrt(1.0 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_t)
                
                if t > 0:
                    z = mean + torch.sqrt(beta_t) * torch.randn_like(z)
                else:
                    z = mean
        
        z = z * self.z_std.to(self.device) + self.z_mean.to(self.device)
        return z.cpu().numpy()


def evaluate_objectives(z, mol_model, target_logp):
    """评估目标函数"""
    mol = mol_model.decode(z)
    if not mol['valid']:
        return np.array([10.0, 10.0, 10000.0])
    
    return np.array([
        -mol['qed'],  # 最大化QED
        abs(mol['logp'] - target_logp),  # 最小化logP偏差
        mol['mw']  # 最小化分子量
    ])


def sbx_crossover(p1, p2, eta=15, prob=0.9, xl=-5, xu=5):
    """模拟二进制交叉"""
    if np.random.random() > prob:
        return p1.copy(), p2.copy()
    
    c1, c2 = p1.copy(), p2.copy()
    
    for i in range(len(p1)):
        if np.random.random() <= 0.5 and abs(p1[i] - p2[i]) > 1e-14:
            y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
            
            beta = 1.0 + (2.0 * (y1 - xl) / (y2 - y1))
            alpha = 2.0 - beta ** (-(eta + 1.0))
            
            rand = np.random.random()
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
            
            c1[i] = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
            
            beta = 1.0 + (2.0 * (xu - y2) / (y2 - y1))
            alpha = 2.0 - beta ** (-(eta + 1.0))
            
            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
            
            c2[i] = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
            
            c1[i] = np.clip(c1[i], xl, xu)
            c2[i] = np.clip(c2[i], xl, xu)
    
    return c1, c2


def polynomial_mutation(x, eta=20, prob=0.1, xl=-5, xu=5):
    """多项式变异"""
    y = x.copy()
    
    for i in range(len(x)):
        if np.random.random() <= prob:
            delta1 = (y[i] - xl) / (xu - xl)
            delta2 = (xu - y[i]) / (xu - xl)
            
            rand = np.random.random()
            mut_pow = 1.0 / (eta + 1.0)
            
            if rand <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                delta_q = 1.0 - val ** mut_pow
            
            y[i] += delta_q * (xu - xl)
            y[i] = np.clip(y[i], xl, xu)
    
    return y


def non_dominated_sort(objs):
    """非支配排序"""
    n = len(objs)
    S = [[] for _ in range(n)]
    n_dominated = [0] * n
    fronts = [[]]
    
    for p in range(n):
        for q in range(n):
            if p != q:
                if np.all(objs[p] <= objs[q]) and np.any(objs[p] < objs[q]):
                    S[p].append(q)
                elif np.all(objs[q] <= objs[p]) and np.any(objs[q] < objs[p]):
                    n_dominated[p] += 1
        
        if n_dominated[p] == 0:
            fronts[0].append(p)
    
    i = 0
    while len(fronts[i]) > 0:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n_dominated[q] -= 1
                if n_dominated[q] == 0:
                    Q.append(q)
        i += 1
        fronts.append(Q)
    
    return fronts[:-1]


def environmental_selection(population, objs, n_pop):
    """环境选择"""
    if len(population) <= n_pop:
        return population, objs
    
    fronts = non_dominated_sort(objs)
    
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= n_pop:
            selected.extend(front)
        else:
            # 按拥挤度选择
            front_objs = objs[front]
            n = len(front)
            distances = np.zeros(n)
            
            for m in range(front_objs.shape[1]):
                sorted_idx = np.argsort(front_objs[:, m])
                distances[sorted_idx[0]] = np.inf
                distances[sorted_idx[-1]] = np.inf
                
                f_range = front_objs[sorted_idx[-1], m] - front_objs[sorted_idx[0], m]
                if f_range > 0:
                    for j in range(1, n - 1):
                        distances[sorted_idx[j]] += (
                            front_objs[sorted_idx[j+1], m] - front_objs[sorted_idx[j-1], m]
                        ) / f_range
            
            sorted_dist = np.argsort(distances)[::-1]
            remaining = n_pop - len(selected)
            selected.extend([front[i] for i in sorted_dist[:remaining]])
            break
    
    return [population[i] for i in selected], objs[selected]


def run_optimization(target_logp=2.0, n_gen=100, pop_size=50, 
                     latent_dim=32, dm_ratio=0.4):
    """运行优化"""
    
    print("="*60)
    print("DDD分子多目标优化 - 演示版本")
    print("="*60)
    print(f"目标logP: {target_logp}")
    print(f"进化代数: {n_gen}")
    print(f"种群大小: {pop_size}")
    print(f"潜在维度: {latent_dim}")
    print("="*60)
    
    # 初始化
    print("\n初始化模型...")
    mol_model = SimplifiedMoleculeModel(latent_dim)
    diffusion = ConditionalDiffusionModel(latent_dim, 3, device='cpu')
    
    # 生成训练数据
    print("生成训练数据...")
    n_train = 200
    z_train = []
    cond_train = []
    
    for _ in range(n_train):
        z = np.random.randn(latent_dim)
        mol = mol_model.decode(z)
        # 简化版本：所有生成的都是有效的
        z_train.append(z)
        cond_train.append([mol['qed'], mol['logp'], mol['mw']])
    
    z_train = np.array(z_train)
    cond_train = np.array(cond_train)
    print(f"训练样本: {len(z_train)}")
    
    # 训练扩散模型
    print("\n训练扩散模型...")
    diffusion.train_model(z_train, cond_train, epochs=50)
    
    # 初始化种群
    print("\n初始化种群...")
    population = []
    pop_objs = []
    
    # 使用扩散模型生成一半
    n_dm = pop_size // 2
    cond = np.random.uniform([0.8, 0.5, 100], [1.0, 1.5, 300], (n_dm, 3))
    z_dm = diffusion.sample(cond)
    
    for z in z_dm:
        obj = evaluate_objectives(z, mol_model, target_logp)
        population.append(z)
        pop_objs.append(obj)
    
    # 随机生成一半
    for _ in range(pop_size - len(population)):
        z = np.random.randn(latent_dim) * 2
        obj = evaluate_objectives(z, mol_model, target_logp)
        population.append(z)
        pop_objs.append(obj)
    
    pop_objs = np.array(pop_objs)
    
    # 优化循环
    print("\n开始优化...")
    history = []
    
    for gen in range(n_gen):
        offspring = []
        off_objs = []
        
        # GA后代
        n_ga = int(pop_size * (1 - dm_ratio))
        for _ in range(n_ga // 2):
            # 锦标赛选择
            candidates = np.random.choice(len(population), 3, replace=False)
            fitness = np.sum(pop_objs[candidates], axis=1)
            p1_idx = candidates[np.argmin(fitness)]
            
            candidates = np.random.choice(len(population), 3, replace=False)
            fitness = np.sum(pop_objs[candidates], axis=1)
            p2_idx = candidates[np.argmin(fitness)]
            
            # 交叉和变异
            c1, c2 = sbx_crossover(population[p1_idx], population[p2_idx])
            c1 = polynomial_mutation(c1)
            c2 = polynomial_mutation(c2)
            
            for c in [c1, c2]:
                obj = evaluate_objectives(c, mol_model, target_logp)
                offspring.append(c)
                off_objs.append(obj)
        
        # DM后代
        n_dm_off = pop_size - len(offspring)
        target_objs = np.random.uniform([0.8, 0.0, 100], [1.0, 2.0, 400], (n_dm_off, 3))
        z_dm = diffusion.sample(target_objs)
        
        for z in z_dm:
            obj = evaluate_objectives(z, mol_model, target_logp)
            offspring.append(z)
            off_objs.append(obj)
        
        off_objs = np.array(off_objs)
        
        # 环境选择
        combined_pop = population + offspring
        combined_objs = np.vstack([pop_objs, off_objs])
        
        population, pop_objs = environmental_selection(combined_pop, combined_objs, pop_size)
        
        # 记录历史
        best_obj = np.min(pop_objs, axis=0)
        history.append(best_obj.copy())
        
        if (gen + 1) % 20 == 0:
            avg_qed = -np.mean(pop_objs[:, 0])
            avg_logp_err = np.mean(pop_objs[:, 1])
            avg_mw = np.mean(pop_objs[:, 2])
            print(f"Gen {gen+1:3d}: QED={avg_qed:.3f}, |logP-target|={avg_logp_err:.3f}, MW={avg_mw:.1f}")
    
    print("\n优化完成!")
    return population, pop_objs, np.array(history), mol_model


def plot_results(pop_objs, history, target_logp, save_dir='results'):
    """绘制结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 帕累托前沿
    is_pareto = np.ones(len(pop_objs), dtype=bool)
    for i in range(len(pop_objs)):
        for j in range(len(pop_objs)):
            if i != j:
                if np.all(pop_objs[j] <= pop_objs[i]) and np.any(pop_objs[j] < pop_objs[i]):
                    is_pareto[i] = False
                    break
    
    pareto_objs = pop_objs[is_pareto]
    
    fig = plt.figure(figsize=(15, 5))
    
    # QED vs logP Error
    ax1 = fig.add_subplot(131)
    ax1.scatter(-pop_objs[:, 0], pop_objs[:, 1], c='lightgray', alpha=0.5, s=30, label='All')
    ax1.scatter(-pareto_objs[:, 0], pareto_objs[:, 1], c='red', s=50, label='Pareto')
    ax1.set_xlabel('QED (Drug-likeness)')
    ax1.set_ylabel(f'|logP - {target_logp}|')
    ax1.set_title('QED vs logP Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # QED vs MW
    ax2 = fig.add_subplot(132)
    ax2.scatter(-pop_objs[:, 0], pop_objs[:, 2], c='lightgray', alpha=0.5, s=30, label='All')
    ax2.scatter(-pareto_objs[:, 0], pareto_objs[:, 2], c='red', s=50, label='Pareto')
    ax2.set_xlabel('QED (Drug-likeness)')
    ax2.set_ylabel('Molecular Weight')
    ax2.set_title('QED vs Molecular Weight')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Convergence
    ax3 = fig.add_subplot(133)
    ax3.plot(range(1, len(history) + 1), history[:, 0], label='QED objective')
    ax3.plot(range(1, len(history) + 1), history[:, 1], label='logP error')
    ax3.plot(range(1, len(history) + 1), history[:, 2] / 100, label='MW/100')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Objective Value')
    ax3.set_title('Convergence History')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/demo_results.png', dpi=150)
    print(f"\n结果图已保存到 {save_dir}/demo_results.png")
    plt.close()
    
    # 打印摘要
    print("\n" + "="*60)
    print("优化结果摘要")
    print("="*60)
    print(f"帕累托解数量: {len(pareto_objs)} / {len(pop_objs)}")
    print(f"\n平均 QED: {-np.mean(pareto_objs[:, 0]):.4f}")
    print(f"平均 logP 误差: {np.mean(pareto_objs[:, 1]):.4f}")
    print(f"平均分子量: {np.mean(pareto_objs[:, 2]):.2f}")
    print(f"\n最佳 QED: {-np.min(pareto_objs[:, 0]):.4f}")
    print(f"最佳 logP 误差: {np.min(pareto_objs[:, 1]):.4f}")
    print("="*60)


if __name__ == '__main__':
    # 运行演示
    population, pop_objs, history, mol_model = run_optimization(
        target_logp=2.0,
        n_gen=100,
        pop_size=50,
        latent_dim=32,
        dm_ratio=0.4
    )
    
    # 绘制结果
    plot_results(pop_objs, history, target_logp=2.0)
