## Build SCIP from Source
SCIP 10.0.0 (Apache-2.0 license, free for academic use): https://github.com/scipopt/scip

Set `myscip` as your SCIP root directory. 
Modify `myscip/scip/src/scip/branch.c` to enable the Python API `SCIPexecBranchruleLP`, which executes the branching rule for fractional LP solutions. 
Our modified version is provided.

Then, build the SCIP source code using CMake and Visual C++.

To enable our customized PySCIPOpt to locate the SCIP libraries, we package the compiled SCIP into a distribution named `SCIPOPTDIR` (as provided in `SCIPOPTDIR.zip` in this repository), 
which contains three folders: `bin`, `lib`, and `include`. 
A Windows environment variable `SCIPOPTDIR` is then created and set to the `SCIPOPTDIR/bin` directory.


## Python Virtual Environment
Since `PySCIPOpt` for SCIP 10.0.0 requires Python >= 3.8, we use Python 3.10 for this project.
- Virtual environment name: `scip1000_getState`
- Path: `D:\Anaconda\envs\scip1000_getState\python.exe`

Run:
```bash
conda create -n scip1000_getState python==3.10.19
conda activate scip1000_getState
python -m pip install --upgrade pip
pip install Cython coverage numpy==2.1.2
pip install "scipy>=1.13.0" pandas openpyxl matplotlib tqdm pyomo tensorboard
```


## Build PySCIPOpt from Source
We use **PySCIPOpt 6.0.0** (under the **MIT license**), which is the Python interface for the SCIP Optimization Suite. 
The official repository is available at: [https://github.com/scipopt/PySCIPOpt](https://github.com/scipopt/PySCIPOpt)
Once the `SCIPOPTDIR` environment variable has been set (as described in the previous section), you can install our customized version, **`PySCIPOpt-6.0.0-for-PGNN`**, using the `pip` tool:
```bash
cd ../PySCIPOpt-6.0.0-for-PGNN
pip install .
```
In this customized version, our custom SCIP functions, feature extraction, and label capture routines are defined in `src/pyscipopt/scip.pxi` and `src/pyscipopt/scip.pxd`.
A successful installation will display the following message in the terminal:
```
Installing collected packages: PySCIPOpt
Successfully installed PySCIPOpt-6.0.0
```

## Machine Learning Platform
**For RTX 2080Ti**：
```bash
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
TORCH=1.7.1 && CUDA=cu101 && \
pip install torch-scatter --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
pip install torch-sparse --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
pip install torch-cluster --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
pip install torch-spline-conv --no-index -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html && \
pip install torch-geometric==2.1.0
```

**For RTX 4090**：
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter     -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-sparse      -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-cluster     -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch-geometric==2.6.1
```
