# Efficient compression of neural networks and datasets

Regularization methods that substantially decrease the number of parameters of neural networks, while maintaining high test accuracy.

The folder `compressing_transformers` contains the codebase to train transformer decoder models on (part of) the Wiki40b/english datasets using different sparsity inducing training approaches (DRR, PMMP, RL1) in Pytorch (Python).

The folder `compressing_classifiers_and_MLPs` contains the codebase to train classifier models on MNIST, CIFAR-10, and ImageNet, or teacher-student MLPs on synthetic data, using different sparsity inducing training approaches (DRR, PMMP, RL1) in Lux (Julia).

Each folder contains its own readme file with instructions on installation and experiment execution.

## Citation

If you use this code, please cite our paper, which you can view at [arxiv.org/abs/2505.17469](https://arxiv.org/abs/2505.17469).

```bibtex
@misc{barth2026efficientcompressionneuralnetworks,
      title={Efficient compression of neural networks and datasets}, 
      author={Lukas Silvester Barth and Paulo von Petersenn},
      year={2026},
      eprint={2505.17469},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.17469}, 
}
```
