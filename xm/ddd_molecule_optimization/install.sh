#!/bin/bash

# DDD分子优化项目 - 一键环境安装脚本
# 适用于 Ubuntu 22.04 / 24.04 等 Debian 系系统

set -e  # 遇到错误立即退出

echo "========================================="
echo "  DDD分子优化项目 环境安装脚本"
echo "========================================="

# 检查 Python 版本
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
REQUIRED_VERSION="3.8"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ 需要 Python 3.8 或更高版本，当前版本为 $PYTHON_VERSION"
    exit 1
fi
echo "✅ Python 版本 $PYTHON_VERSION 符合要求"

# 安装必要的系统工具
echo "📦 安装系统依赖 (build-essential, python3-venv, 编译工具)..."
sudo apt update
sudo apt install -y python3-venv python3-full build-essential libboost-all-dev libeigen3-dev

# 创建虚拟环境
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "🔧 创建虚拟环境..."
    python3 -m venv $VENV_DIR
else
    echo "✅ 虚拟环境已存在，跳过创建"
fi

# 激活虚拟环境并安装依赖
echo "🚀 进入虚拟环境并安装 Python 包..."
source $VENV_DIR/bin/activate

# 升级 pip 并设置国内镜像（可选，注释掉以使用官方源）
echo "⚙️ 升级 pip 并配置镜像源（清华源）..."
pip install --upgrade pip
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 设置超时时间
PIP_OPTIONS="--default-timeout=100"

# 分批安装依赖（先轻量后重型）
echo "📦 安装基础科学计算包 (numpy, scipy, matplotlib, pandas, tqdm, scikit-learn)..."
pip install $PIP_OPTIONS numpy scipy matplotlib pandas tqdm scikit-learn

echo "📦 安装 PyTorch (>=1.9.0)..."
pip install $PIP_OPTIONS "torch>=1.9.0"

echo "📦 安装 pymoo (>=0.6.0)..."
pip install $PIP_OPTIONS "pymoo>=0.6.0"

echo "📦 安装 RDKit (>=2020.09.1) - 这可能耗时较长，需编译..."
if pip install $PIP_OPTIONS "rdkit>=2020.09.1"; then
    echo "✅ RDKit 安装成功"
else
    echo "⚠️  RDKit 通过 pip 安装失败，可能是编译环境问题。"
    echo "   尝试使用 conda 安装（需要先安装 Miniconda）..."
    echo ""
    echo "   备用方案："
    echo "   # 安装 Miniconda"
    echo "   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "   bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3"
    echo "   ~/miniconda3/bin/conda init"
    echo "   然后重新打开终端，创建新环境："
    echo "   conda create -n ddd python=3.9"
    echo "   conda activate ddd"
    echo "   conda install -c conda-forge rdkit pytorch numpy scipy matplotlib pandas scikit-learn pymoo"
    echo "   conda install -c conda-forge selfies guacamol"
    echo ""
    echo "   之后使用 conda 环境运行项目即可。"
    exit 1
fi

# 安装可选库
echo "📦 安装可选库 (selfies, guacamol)..."
pip install $PIP_OPTIONS selfies guacamol

echo "✅ 所有依赖安装完成！"

# 提示用户如何运行
echo ""
echo "========================================="
echo " 安装成功！使用以下命令激活环境并运行项目："
echo ""
echo "  source $VENV_DIR/bin/activate"
echo "  python main.py [参数]"
echo ""
echo "  （可选参数参见 README.md）"
echo "========================================="
