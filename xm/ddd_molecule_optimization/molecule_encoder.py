import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from typing import List, Optional, Union

try:
    # 尝试导入guacamol（如果可用）
    from guacamol.utils.chemistry import canonicalize
    from guacamol.models import load_model
    has_guacamol = True
except ImportError:
    has_guacamol = False

class MoleculeEncoder:
    """
    分子编码器-解码器类，支持PCA和VAE两种方法
    """
    def __init__(self, latent_dim: int = 128, method: str = 'pca'):
        """
        初始化分子编码器
        
        Args:
            latent_dim: 潜在空间维度
            method: 编码方法，可选'pca'或'vae'
        """
        self.latent_dim = latent_dim
        self.method = method
        self.pca = None
        self.neighbors = None
        self.smiles_list = []
        self.z_list = []
        self.vae_model = None
    
    def fit(self, smiles_list: List[str]) -> None:
        """
        拟合编码器
        
        Args:
            smiles_list: SMILES字符串列表
        """
        if self.method == 'pca':
            # 计算摩根指纹
            fingerprints = []
            valid_smiles = []
            
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                    fingerprints.append(np.array(fp))
                    valid_smiles.append(smiles)
            
            if len(fingerprints) == 0:
                raise ValueError("没有有效的分子用于拟合")
            
            # 拟合PCA
            self.pca = PCA(n_components=self.latent_dim)
            self.z_list = self.pca.fit_transform(np.array(fingerprints))
            self.smiles_list = valid_smiles
            
            # 构建最近邻索引
            self.neighbors = NearestNeighbors(n_neighbors=1, metric='euclidean')
            self.neighbors.fit(self.z_list)
        elif self.method == 'vae' and has_guacamol:
            # 加载预训练VAE模型（这里使用示例路径，实际项目中需要替换）
            try:
                self.vae_model = load_model('guacamol_vae')
            except Exception as e:
                print(f"加载VAE模型失败: {e}")
                print("切换到PCA方法")
                self.method = 'pca'
                self.fit(smiles_list)
        else:
            # 默认为PCA方法
            self.method = 'pca'
            self.fit(smiles_list)
    
    def encode(self, smiles: str) -> np.ndarray:
        """
        将SMILES编码为潜在向量
        
        Args:
            smiles: SMILES字符串
        
        Returns:
            潜在向量z (shape: [latent_dim])
        """
        if self.method == 'pca':
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(self.latent_dim)
            
            # 计算摩根指纹
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fp_array = np.array(fp).reshape(1, -1)
            
            # 使用PCA降维
            if self.pca is None:
                raise ValueError("PCA模型未拟合")
            z = self.pca.transform(fp_array)[0]
            return z
        elif self.method == 'vae' and self.vae_model is not None:
            # 使用VAE编码
            try:
                z = self.vae_model.encode(smiles)
                return z
            except Exception:
                return np.zeros(self.latent_dim)
        else:
            return np.zeros(self.latent_dim)
    
    def encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        """
        批量编码SMILES
        
        Args:
            smiles_list: SMILES字符串列表
        
        Returns:
            批量潜在向量z (shape: [n, latent_dim])
        """
        z_list = []
        for smiles in smiles_list:
            z = self.encode(smiles)
            z_list.append(z)
        return np.array(z_list)
    
    def decode(self, z: np.ndarray, k: int = 1) -> Optional[str]:
        """
        将潜在向量解码为SMILES
        
        Args:
            z: 潜在向量
            k: 最近邻数量
        
        Returns:
            SMILES字符串，若失败则返回None
        """
        if self.method == 'pca':
            if self.neighbors is None or len(self.smiles_list) == 0:
                return None
            
            # 查找最近邻
            distances, indices = self.neighbors.kneighbors(z.reshape(1, -1), n_neighbors=k)
            
            # 检查距离阈值
            if distances[0][0] > 5.0:
                return None
            
            return self.smiles_list[indices[0][0]]
        elif self.method == 'vae' and self.vae_model is not None:
            # 使用VAE解码
            try:
                smiles = self.vae_model.decode(z)
                return smiles
            except Exception:
                return None
        else:
            return None
