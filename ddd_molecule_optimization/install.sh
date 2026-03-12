#!/bin/bash

# DDD分子多目标优化 - 安装脚本
# 本脚本用于安装项目所需的所有依赖

echo "=========================================="
echo "DDD分子多目标优化 - 安装脚本"
echo "=========================================="

# 检查Python版本
echo ""
echo "检查Python版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 创建虚拟环境（可选）
read -p "是否创建虚拟环境? (y/n): " create_venv
if [ "$create_venv" = "y" ] || [ "$create_venv" = "Y" ]; then
    echo ""
    echo "创建虚拟环境..."
    python3 -m venv venv
    source venv/bin/activate
    echo "虚拟环境已激活"
fi

# 升级pip
echo ""
echo "升级pip..."
pip install --upgrade pip

# 安装基础依赖
echo ""
echo "安装基础依赖..."
pip install numpy scipy pandas matplotlib tqdm scikit-learn

# 安装PyTorch
echo ""
echo "安装PyTorch..."
if command -v nvcc &> /dev/null; then
    echo "检测到CUDA，安装GPU版本PyTorch..."
    pip install torch torchvision torchaudio
else
    echo "未检测到CUDA，安装CPU版本PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 安装RDKit
echo ""
echo "安装RDKit..."
pip install rdkit

# 安装pymoo
echo ""
echo "安装pymoo..."
pip install pymoo

# 安装其他依赖
echo ""
echo "安装其他依赖..."
pip install -r requirements.txt

# 验证安装
echo ""
echo "验证安装..."
python3 -c "
import sys
print('Python版本:', sys.version)
print()

try:
    import numpy
    print('✓ numpy', numpy.__version__)
except ImportError:
    print('✗ numpy 未安装')

try:
    import torch
    print('✓ torch', torch.__version__)
    print('  CUDA可用:', torch.cuda.is_available())
except ImportError:
    print('✗ torch 未安装')

try:
    import rdkit
    print('✓ rdkit 已安装')
except ImportError:
    print('✗ rdkit 未安装')

try:
    import sklearn
    print('✓ scikit-learn', sklearn.__version__)
except ImportError:
    print('✗ scikit-learn 未安装')

try:
    import pandas
    print('✓ pandas', pandas.__version__)
except ImportError:
    print('✗ pandas 未安装')

try:
    import matplotlib
    print('✓ matplotlib', matplotlib.__version__)
except ImportError:
    print('✗ matplotlib 未安装')

try:
    import pymoo
    print('✓ pymoo', pymoo.__version__)
except ImportError:
    print('✗ pymoo 未安装')
"

echo ""
echo "=========================================="
echo "安装完成!"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  1. 运行优化: python main.py"
echo "  2. 查看帮助: python main.py --help"
echo "  3. 使用自定义参数: python main.py --target_logp 2.5 --n_gen 300"
echo ""

if [ "$create_venv" = "y" ] || [ "$create_venv" = "Y" ]; then
    echo "注意: 使用虚拟环境时需要先激活:"
    echo "  source venv/bin/activate"
    echo ""
fi
