"""
数据工具模块 - 用于加载和处理分子数据
"""

import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from typing import List, Tuple, Optional


def load_qm9_data(num_samples: int = 5000, data_path: str = 'qm9.csv') -> Tuple[List[str], pd.DataFrame]:
    """
    加载QM9数据集，计算分子性质并过滤无效分子
    
    Args:
        num_samples: 要使用的样本数量
        data_path: CSV文件路径
        
    Returns:
        smiles_list: 有效分子的SMILES字符串列表
        properties_df: 包含分子性质的DataFrame，列名['qed', 'logp', 'mw']
    """
    # 检查文件是否存在
    if os.path.exists(data_path):
        # 从本地文件读取
        df = pd.read_csv(data_path)
        if 'smiles' not in df.columns or 'qed' not in df.columns or 'logp' not in df.columns or 'mw' not in df.columns:
            raise ValueError("CSV文件必须包含'smiles', 'qed', 'logp', 'mw'列")
    else:
        # 生成示例数据
        print("正在生成示例数据...")
        smiles_list, qed_list, logp_list, mw_list = _generate_sample_data(num_samples)
        df = pd.DataFrame({
            'smiles': smiles_list,
            'qed': qed_list,
            'logp': logp_list,
            'mw': mw_list
        })
        # 保存到本地
        df.to_csv(data_path, index=False)
        print(f"示例数据已保存到 {data_path}")
    
    # 过滤无效分子
    valid_smiles = []
    valid_indices = []
    
    for idx, smiles in enumerate(df['smiles']):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
            valid_indices.append(idx)
    
    # 提取有效性质
    properties_df = df.iloc[valid_indices][['qed', 'logp', 'mw']].reset_index(drop=True)
    
    print(f"加载了 {len(valid_smiles)} 个有效分子")
    return valid_smiles, properties_df


def _generate_sample_data(num_samples: int) -> Tuple[List[str], List[float], List[float], List[float]]:
    """
    生成示例分子数据
    
    Args:
        num_samples: 样本数量
        
    Returns:
        smiles_list, qed_list, logp_list, mw_list
    """
    smiles_list = []
    qed_list = []
    logp_list = []
    mw_list = []
    
    # 基础分子模板
    base_smiles = [
        'C', 'CC', 'CCC', 'CCCC', 'CCCCC', 'CCCCCC',
        'CO', 'CCO', 'CCCO', 'CCCCO', 'CCCCCO',
        'C=O', 'CC=O', 'CCC=O', 'CCCC=O',
        'CC(C)C', 'CC(C)(C)C', 'CC(C)CC',
        'c1ccccc1', 'c1ccccc1C', 'c1ccccc1CC',
        'c1ccccc1O', 'c1ccccc1CO', 'c1ccccc1OC',
        'CC(=O)O', 'CCC(=O)O', 'CCCC(=O)O',
        'CCN', 'CCCN', 'CCCCN',
        'C1CCCCC1', 'C1CCCCC1C', 'C1CCCC1',
        'CC=C', 'CCC=C', 'CCCC=C',
        'C#C', 'CC#C', 'CCC#C',
        'c1ccc(O)cc1', 'c1ccc(C)cc1', 'c1ccc(Cl)cc1',
        'CC(=O)OC', 'CCOC(=O)C', 'CC(C)O',
        'CN', 'CCCN(C)C', 'c1ccncc1',
        'CCS', 'CCCS', 'CCCCS',
        'CF', 'CCF', 'CCCl', 'CCBr',
    ]
    
    # 扩展分子列表
    extended_smiles = []
    for smiles in base_smiles:
        extended_smiles.append(smiles)
        # 添加一些变体
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            qed = QED.qed(mol)
            logp = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            
            # 根据分子量添加变体
            if mw < 100:
                extended_smiles.append(smiles + 'C')
                extended_smiles.append(smiles + 'CC')
            elif mw < 200:
                extended_smiles.append(smiles + 'C')
    
    # 生成所需数量的样本
    np.random.seed(42)
    while len(smiles_list) < num_samples:
        smiles = np.random.choice(extended_smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            smiles_list.append(smiles)
            qed_list.append(QED.qed(mol))
            logp_list.append(Descriptors.MolLogP(mol))
            mw_list.append(Descriptors.MolWt(mol))
    
    return smiles_list[:num_samples], qed_list[:num_samples], logp_list[:num_samples], mw_list[:num_samples]


def compute_properties(smiles: str) -> Optional[dict]:
    """
    计算分子的各项性质
    
    Args:
        smiles: SMILES字符串
        
    Returns:
        包含性质的字典，若分子无效则返回None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        props = {
            'qed': QED.qed(mol),
            'logp': Descriptors.MolLogP(mol),
            'mw': Descriptors.MolWt(mol),
            'tpsa': Descriptors.TPSA(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'rb': Descriptors.NumRotatableBonds(mol),
            'rings': Descriptors.RingCount(mol)
        }
        return props
    except Exception as e:
        print(f"计算性质时出错: {e}")
        return None


def filter_druglike_molecules(smiles_list: List[str], 
                               qed_threshold: float = 0.5,
                               mw_range: Tuple[float, float] = (150, 500),
                               logp_range: Tuple[float, float] = (-2, 5)) -> List[str]:
    """
    过滤类药分子
    
    Args:
        smiles_list: SMILES字符串列表
        qed_threshold: QED阈值
        mw_range: 分子量范围
        logp_range: logP范围
        
    Returns:
        符合条件的分子列表
    """
    filtered = []
    
    for smiles in smiles_list:
        props = compute_properties(smiles)
        if props is None:
            continue
        
        if (props['qed'] >= qed_threshold and
            mw_range[0] <= props['mw'] <= mw_range[1] and
            logp_range[0] <= props['logp'] <= logp_range[1]):
            filtered.append(smiles)
    
    return filtered


def normalize_properties(properties_df: pd.DataFrame) -> pd.DataFrame:
    """
    归一化分子性质
    
    Args:
        properties_df: 性质DataFrame
        
    Returns:
        归一化后的DataFrame
    """
    normalized_df = properties_df.copy()
    
    for col in properties_df.columns:
        min_val = properties_df[col].min()
        max_val = properties_df[col].max()
        if max_val > min_val:
            normalized_df[col] = (properties_df[col] - min_val) / (max_val - min_val)
        else:
            normalized_df[col] = 0.5
    
    return normalized_df
