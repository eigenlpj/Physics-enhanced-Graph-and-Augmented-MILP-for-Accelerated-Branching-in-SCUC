conda install -c conda-forge ecole=0.7.3

pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

pip install tqdm

pip install tensorboard

pip install scipy
TORCH=1.7.1 && CUDA=cu101 && \
pip install torch-scatter --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
pip install torch-sparse --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
pip install torch-cluster --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
pip install torch-spline-conv --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
pip install torch-geometric==2.1.0


# 2025.8.10 佳明修改 上述库是运行在RTX2080Ti的,本人使用的是RTX4090,经试验,发现兼容的库如下:
# 虚拟环境名称: CAMbranch
# 路径: D:\Anaconda\envs\CAMbranch\python.exe
# 备注1: windows平台暂时无法运行ecole库,使用了gasse源码的utilities.py文件来实现提取特征的功能
# 备注2: 后续是否考虑不使用torch-geometric库或者升级改版本,因为该库的适配较难,且在安装时警告官方目前已经放弃维护torch-geometric2.1.0版本
# python == 3.7.0
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm==4.67.1
pip install tensorboard==2.11.2
pip install scipy==1.7.3
TORCH=1.13.1 && CUDA=cu117 && \
pip install torch-scatter --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
pip install torch-sparse --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
pip install torch-cluster --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
pip install torch-spline-conv --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
pip install torch-geometric==2.1.0


# 2025.12.11 佳明修改 由于scip10.0.0的pyscipopt必须要求python>=3.8,故该虚拟环境选用python3.10
# 虚拟环境名称: scip1000
# 路径: D:\Anaconda\envs\scip1000\python.exe
# 1. 创建并激活环境
conda create -n scip1000 python=3.10 -y
conda activate scip1000
# 2. 升级 pip
python -m pip install --upgrade pip
# 3. 安装 PyTorch (cu121)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# 4. 安装 PyG 扩展
pip install torch-scatter     -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-sparse      -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-cluster     -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-geometric==2.6.1
# 5. 安装其他依赖
pip install "numpy>=2.0" "scipy>=1.13.0" pandas openpyxl matplotlib tqdm pyomo tensorboard
# 6. 新增：安装 PySCIPOpt 的构建依赖(建议numpy2.1.2)
pip install Cython coverage
# 7. 安装本人定制的 PySCIPOpt(主要修改了获取二部图的函数, 调用指定名称的分支策略的函数, 提取香草强分支的执行结果的函数)
使用命令行窗口 cd D:\LiJiamigFile\scip10.0.0\PySCIPOpt-6.0.0-revised-by-LJM
conda activate scip1000
pip install .


# 2026.2.27 佳明修改 由于2025.12.11的提取特征函数可能不够准确且缺乏c编译导致效率太低,考虑使用gasse的getState函数重新在scip10.0.0实现特征提取
# 虚拟环境名称: scip1000_getState
# 路径: D:\Anaconda\envs\scip1000_getState\python.exe
# 1. 创建并激活环境
conda create -n scip1000_getState python==3.10.19
conda activate scip1000_getState
# 2. 升级 pip
python -m pip install --upgrade pip
# 3. 新增：安装 PySCIPOpt 的构建依赖(建议numpy2.1.2)
pip install Cython coverage numpy==2.1.2
# 4. 安装本人定制的 PySCIPOpt(根据gasse的getState代码,针对scip10.0.0适配,增加了获取二部图的函数getState, 调用指定名称的分支策略的函数, 提取香草强分支的执行结果的函数)
使用命令行窗口 cd D:\LiJiamigFile\scip10.0.0\pyscipopt_6.0.0_Add_getState_2026.2.27
conda activate scip1000_getState
pip install . 或者 python setup.py install
# 3. 安装 PyTorch (cu121)
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# 4. 安装 PyG 扩展
pip install torch-scatter     -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-sparse      -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-cluster     -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-geometric==2.6.1
# 5. 安装其他依赖
pip install "numpy>=2.0" "scipy>=1.13.0" pandas openpyxl matplotlib tqdm pyomo tensorboard

