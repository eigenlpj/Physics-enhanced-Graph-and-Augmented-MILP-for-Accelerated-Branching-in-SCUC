# Physics-enhanced Graph and Augmented MILP for Accelerated Branching in Security-Constrained Unit Commitment
(Security-Constrained Unit Commitment) SCUC  is essential for day-ahead market clearing and is typically formulated as a Mixed-Integer Linear Programming (MILP) problem, solved by branch-and-cut (B&C)-based solvers. While recent Graph Neural Network (GNN)-based branching strategies have shown promise, existing methods face two critical bottlenecks. First, generic mathematical features fail to capture the intrinsic physical characteristics of SCUC, limiting branching quality. Second, massive strong branching (StB) samples required for training on large-scale systems are computationally prohibitive. To address these challenges, this paper proposes a physics-informed graph neural network (PGNN) branching framework incorporating two novel modules:(i) a physics-enhanced graph (Pgraph); and (ii) an Augmented MILP (AMILP) method.

Peijie Li, Jiaming Li, Xiaoqing Bai

---

## Table of Contents

1. [Environment and Dependencies](#environment-and-dependencies)
2. [Sample Collection](#sample-collection)
3. [PGNN Training](#model-training)
4. [PGNN Deployment and Instance Testing](#instance-testing)

## Citation

If you use this code in your work, please cite our paper:

```bibtex
@article{li2026physics,
  title={Physics-enhanced Graph and Augmented-MILP for Accelerated Branching in Security-Constrained Unit Commitment},
  author={Li, Jiaming and Li, Peijie and Bai, Xiaoqing},
  note={under review at IEEE Transactions on Power Systems}
}
---

## Acknowledgments

This repository uses or draws inspiration from the following open-source projects:

- **Gasse et al. (NeurIPS 2019)** - Provides a generic bipartite graph representation for MILP, which serves as the foundation of our implementation.
  - Paper: "Exact Combinatorial Optimization with Graph Convolutional Neural Networks"
  - Repository: https://github.com/ds4dm/learn2branch

- **Lin et al. (ICLR 2024)** - The AMILP method in this work is adapted from CAMBranch.
  - Paper: "CAMBranch: Contrastive learning with augmented MILPs for branching"
  - Author's GitHub: https://github.com/linjc16

We thank the authors for making their work publicly available.
