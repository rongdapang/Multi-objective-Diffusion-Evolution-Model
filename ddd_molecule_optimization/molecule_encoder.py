"""
分子编码器模块 - 用于将分子编码到潜在空间
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from typing import List, Optional, Union


class MoleculeEncoder:
    """
    分子编码器-解码器类，支持PCA和指纹方法
    """
    
    def __init__(self, latent_dim: int = 128, method: str = 'pca'):
        """
        初始化分子编码器
        
        Args:
            latent_dim: 潜在空间维度
            method: 编码方法，可选'pca'或'fingerprint'
        """
        self.latent_dim = latent_dim
        self.method = method
        self.pca = None
        self.neighbors = None
        self.smiles_list = []
        self.z_list = []
        self.fingerprint_dim = 2048
        
    def fit(self, smiles_list: List[str]) -> None:
        """
        拟合编码器
        
        Args:
            smiles_list: SMILES字符串列表
        """
        if self.method == 'pca':
            self._fit_pca(smiles_list)
        elif self.method == 'fingerprint':
            self._fit_fingerprint(smiles_list)
        else:
            raise ValueError(f"不支持的编码方法: {self.method}")
    
    def _fit_pca(self, smiles_list: List[str]) -> None:
        """
        使用PCA拟合编码器
        """
        # 计算摩根指纹
        fingerprints = []
        valid_smiles = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, radius=2, nBits=self.fingerprint_dim
                )
                fingerprints.append(np.array(fp))
                valid_smiles.append(smiles)
        
        if len(fingerprints) == 0:
            raise ValueError("没有有效的分子用于拟合")
        
        # 调整潜在维度
        actual_dim = min(self.latent_dim, len(fingerprints), self.fingerprint_dim)
        if actual_dim < self.latent_dim:
            print(f"警告: 调整潜在维度为 {actual_dim}")
            self.latent_dim = actual_dim
        
        # 拟合PCA
        self.pca = PCA(n_components=self.latent_dim)
        self.z_list = self.pca.fit_transform(np.array(fingerprints))
        self.smiles_list = valid_smiles
        
        # 构建最近邻索引
        self.neighbors = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.neighbors.fit(self.z_list)
        
        # 保存指纹用于指纹方法
        self.fingerprints = np.array(fingerprints)
        
        print(f"PCA拟合完成: {len(valid_smiles)} 个分子, 解释方差比: {self.pca.explained_variance_ratio_.sum():.3f}")
    
    def _fit_fingerprint(self, smiles_list: List[str]) -> None:
        """
        使用指纹方法拟合编码器
        """
        fingerprints = []
        valid_smiles = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, radius=2, nBits=self.fingerprint_dim
                )
                fingerprints.append(np.array(fp))
                valid_smiles.append(smiles)
        
        if len(fingerprints) == 0:
            raise ValueError("没有有效的分子用于拟合")
        
        # 使用PCA降维到指定维度
        actual_dim = min(self.latent_dim, len(fingerprints), self.fingerprint_dim)
        self.latent_dim = actual_dim
        
        self.pca = PCA(n_components=actual_dim)
        self.z_list = self.pca.fit_transform(np.array(fingerprints))
        self.smiles_list = valid_smiles
        self.fingerprints = np.array(fingerprints)
        
        # 构建最近邻索引
        self.neighbors = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.neighbors.fit(self.z_list)
    
    def encode(self, smiles: str) -> np.ndarray:
        """
        将SMILES编码为潜在向量
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            潜在向量z (shape: [latent_dim])
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.latent_dim)
        
        # 计算摩根指纹
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=2, nBits=self.fingerprint_dim
        )
        fp_array = np.array(fp).reshape(1, -1)
        
        # 使用PCA降维
        if self.pca is None:
            raise ValueError("编码器未拟合")
        
        z = self.pca.transform(fp_array)[0]
        return z
    
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
    
    def decode(self, z: np.ndarray, k: int = 3) -> Optional[str]:
        """
        将潜在向量解码为SMILES
        
        Args:
            z: 潜在向量
            k: 最近邻数量
            
        Returns:
            SMILES字符串，若失败则返回None
        """
        if self.neighbors is None or len(self.smiles_list) == 0:
            return None
        
        # 查找最近邻
        distances, indices = self.neighbors.kneighbors(
            z.reshape(1, -1), n_neighbors=min(k, len(self.smiles_list))
        )
        
        # 检查距离阈值
        if distances[0][0] > 10.0:
            return None
        
        # 返回最近的分子
        return self.smiles_list[indices[0][0]]
    
    def decode_batch(self, z_batch: np.ndarray, k: int = 3) -> List[Optional[str]]:
        """
        批量解码潜在向量
        
        Args:
            z_batch: 批量潜在向量 (shape: [n, latent_dim])
            k: 最近邻数量
            
        Returns:
            SMILES字符串列表
        """
        smiles_list = []
        for z in z_batch:
            smiles = self.decode(z, k=k)
            smiles_list.append(smiles)
        return smiles_list
    
    def get_reconstruction_error(self, smiles_list: List[str]) -> float:
        """
        计算重构误差
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            平均重构误差
        """
        errors = []
        
        for smiles in smiles_list:
            z = self.encode(smiles)
            decoded = self.decode(z)
            
            if decoded is not None:
                # 计算原始分子和解码分子的相似度
                mol1 = Chem.MolFromSmiles(smiles)
                mol2 = Chem.MolFromSmiles(decoded)
                
                if mol1 is not None and mol2 is not None:
                    fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, radius=2)
                    fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, radius=2)
                    
                    # 使用Tanimoto相似度
                    from rdkit import DataStructs
                    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                    errors.append(1 - similarity)
        
        return np.mean(errors) if errors else 1.0
