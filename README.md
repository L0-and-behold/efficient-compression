# Efficient compression of neural networks and datasets

Regularization methods that substantially decrease the number of parameters of neural networks, while maintaining high test accuracy.

The folder `compressing_transformers` contains the codebase to train transformer decoder models on (part of) the Wiki40b/english datasets using different sparsity inducing training approaches (DRR, PMMP, RL1) in Pytorch (Python).

The folder `compressing_classifiers_MLPs` contains the codebase to train classifier models on MNIST and CIFAR-10 or teacher-student MLPs on synthetic data using different sparsity inducing training approaches (DRR, PMMP, RL1) in Lux (Julia).

Each folder contains its own readme file with instructions on installation and experiment execution.

## Citation

If you use this code, please cite our paper

```bibtex
@misc{barth2025efficientcompressionneuralnetworks,
      title={Efficient compression of neural networks and datasets}, 
      author={Lukas Silvester Barth and Paulo {von Petersenn}},
      year={2025},
      eprint={2505.17469},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.17469}, 
}
```
