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
        # 尝试从网络下载（这里使用简化实现，实际项目中可使用DeepChem）
        # 注意：这只是一个示例，实际使用时需要替换为真实的下载逻辑
        print("正在生成示例数据...")
        # 生成示例数据
        from rdkit.Chem import AllChem
        smiles_list = []
        qed_list = []
        logp_list = []
        mw_list = []
        
        # 生成一些简单的分子作为示例
        sample_smiles = [
            'C', 'CC', 'CCC', 'CCCC', 'CCCCC',
            'CO', 'CCO', 'CCCO', 'CCCCO',
            'C=O', 'CC=O', 'CCC=O',
            'c1ccccc1', 'c1ccccc1O', 'c1ccccc1CO'
        ]
        
        for smiles in sample_smiles * (num_samples // len(sample_smiles) + 1):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                smiles_list.append(smiles)
                qed_list.append(QED.qed(mol))
                logp_list.append(Descriptors.MolLogP(mol))
                mw_list.append(Descriptors.MolWt(mol))
            if len(smiles_list) >= num_samples:
                break
        
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
    valid_properties = []
    
    for _, row in df.iterrows():
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
            valid_properties.append({
                'qed': row['qed'],
                'logp': row['logp'],
                'mw': row['mw']
            })
        if len(valid_smiles) >= num_samples:
            break
    
    properties_df = pd.DataFrame(valid_properties)
    return valid_smiles, properties_df

def compute_properties(smiles: str) -> Optional[dict]:
    """
    计算单个分子的性质
    
    Args:
        smiles: 分子的SMILES字符串
    
    Returns:
        包含分子性质的字典，若分子无效则返回None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return {
        'qed': QED.qed(mol),
        'logp': Descriptors.MolLogP(mol),
        'mw': Descriptors.MolWt(mol)
    }
