"""
DiffusionEvolution 安装配置
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="diffusion-evolution",
    version="1.0.0",
    author="AI Assistant",
    author_email="developer@example.com",
    description="扩散模型与进化算法混合的多目标优化算法",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/DiffusionEvolution_Python",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "examples": [
            "matplotlib>=3.3.0",
            "pandas>=1.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "diffusion-evolution=diffusion_evolution.cli:main",
        ],
    },
)