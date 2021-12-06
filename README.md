# Effectiveness of Second-Order Optimization for Non-convex Machine Learning - ECE 5554

## About
Classical optimization algorithms have become unfavorable in the world of large models and big data. Over the past decade, first-order optimization methods such as SGD and Adam have dominated deep learning literature. Despite proven theoretical properties, second-order optimization techniques are far less prevalent and unconventional in modern deep learning due to their prohibitive computation. We bring awareness to these second-order techniques by highlighting their advantages, such as robustness to hyperparameters, finite step convergence, escaping saddle points, and resilience to adversarial effects.

![Alt text](https://github.com/mnguyen0226/soo_non_convex_ml/blob/main/docs/results/first_order_vs_second_order.png)

## Reproducibility


## Results
- Optimizers Performance Comparison on MNIST Dataset:
![Alt text](https://github.com/mnguyen0226/soo_non_convex_ml/blob/main/docs/results/Optimizers%20Performance%20Comparison%20on%20MNIST%20Datset.png)

- Optimizers Performance Comparison on CIFAR-10 Dataset:
![Alt text](https://github.com/mnguyen0226/soo_non_convex_ml/blob/main/docs/results/Optimizers%20Performance%20Comparison%20on%20CIFAR-10%20Dataset.png)

- Optimizers Robustness to Learning Rate Comparison:
![Alt text](https://github.com/mnguyen0226/soo_non_convex_ml/blob/main/docs/results/Optimizers%20Robustness%20to%20Learning%20Rate%20Comparison.png)


## Paper
- [Proposal](https://github.com/mnguyen0226/soo_non_convex_ml/tree/main/docs/Proposal)
- [Final Paper]

## Citation
- [1] Naman Agarwal, Brian Bullins, and Elad Hazan. “Second-order stochastic optimization inlinear time”. In:stat1050 (2016), p. 15.
- [2] Naman Agarwal et al. “Finding approximate local minima faster than gradient descent”. In:Proceedings of the 49th Annual ACM SIGACT Symposium on Theory of Computing. 2017,pp. 1195–1199.
- [3] Shun-Ichi Amari. “Natural gradient works efficiently in learning”. In:Neural computation10.2 (1998), pp. 251–276.
- [4] Shun-ichi Amari et al. “When Does Preconditioning Help or Hurt Generalization?” In:arXivpreprint arXiv:2006.10732(2020).
- [5] Costas Bekas, Effrosyni Kokiopoulou, and Yousef Saad. “An estimator for the diagonal of amatrix”. In:Applied numerical mathematics57.11-12 (2007), pp. 1214–1229.
- [6] Raghu Bollapragada, Richard H Byrd, and Jorge Nocedal. “Exact and inexact subsampledNewton methods for optimization”. In:IMA Journal of Numerical Analysis39.2 (2019),pp. 545–578.
- [7] Dami Choi et al.On Empirical Comparison of Optimizers of Deep Learning. 2020.URL:https://arxiv.org/pdf/1910.05446.pdf.
- [8] Dan C. Ciresan et al.High-Performance Neural Networks for Visual Object Classification.2011.URL:https://arxiv.org/pdf/1102.0183.pdf.
- [9] E. M. Dogo et al. “A Comparative Analysis of Gradient Descent-Based Optimization Algo-rithms on Convolutional Neural Networks”. In:International Conference on ComputationalTechniques, Electronics and Mechanical Systems (CTEMS)(2018).
- [10] J. Duchi, E. Hazan, and Y. Singer. “Adaptive Subgradient Methods for Online Learning andStochastic Optimization”. In:Journal of Machine Learning Research(2011).
- [11] John Duchi, Elad Hazan, and Yoram Singer. “Adaptive subgradient methods for online learningand stochastic optimization.” In:Journal of machine learning research12.7 (2011).
- [12] Gradient Descent.URL:https://en.wikipedia.org/wiki/Gradient_descent.
- [13] G. Hinton, N. Srivastava, and K. Swersky. “rmsprop: Divide the gradient by a running averageof its recent magnitude”. In: (2012).
- [14] Loshchilov  Ilya  and  Hutter  Frank.Decoupled  Weight  Decay  Regularization. 2019.URL:https://arxiv.org/pdf/1711.05101.pdf.
- [15] Sutskever Ilya et al.On the importance of initialization and momentum in deep learning. 2013.URL:https://www.cs.toronto.edu/~hinton/absps/momentum.pdf.
- [16] Diederik P. Kingma and Jimmy Lei Ba.Adam: A Method For Stochastic Optimization. 2017.URL:https://arxiv.org/pdf/1412.6980.pdf.
- [17] A. Krizhevsky, I. Sutskever, and G. E. Hinton. “ImageNet Classification with Deep Convolu-tional Neural Networks”. In:Advances in neural information processing systems(2012).
- [18] Alex Krizhevsky.Learning Multiple Layers of Features from Tiny Images. 2009.URL:https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf.-
- [19] Yann A LeCun et al. “Efficient backprop”. In:Neural networks: Tricks of the trade. Springer,2012, pp. 9–48.
- [20] F. Schneider, L. Balles, and P. Hennig.DeepOBS: A Deep Learning Optimizer BenchmarkSuite. 2019.URL:https://arxiv.org/abs/1903.05499.
- [21] S. Schneider et al.Past, present and future approaches using computer vision for animalreidentification from camera trap data. 2019.
- [22] Mittal Sparsh and Vaishay Shraiysh. “A Survey of Techniques for Optimizing Deep Learningon GPUs”. In:Journal of Systems Architecture(2019).
- [23] A Wilson et al. “The marginal value of adaptive gradient methods in machine learning”. In:Advances in Neural Information Processing Systems(2017).
- [24] Robert E Wyatt. “Matrix spectroscopy: Computation of interior eigenstates of large matricesusing layered iteration”. In:Physical Review E51.4 (1995), p. 3643.
- [25] Peng Xu, Roosta Fred, and W. Mahoney Michael. “Newton-type methods for non-convexoptimization under inexact hessian information.” In:Mathematical Programming 184.1(2020).
- [26] Peng Xu, Roosta Fred, and W. Mahoney Michael. “Second-order optimization for non-convexmachine learning: An empirical study”. In:Society for Industrial and Applied Mathematics(SIAM)(2020).
- [27] Zhewei Yao et al.ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning.2021.URL:https://arxiv.org/abs/2006.00719.
- [28] Chengxi Ye et al.On the Importance of Consistency in Training Deep Neural Networks.2017.URL:https://www.researchgate.net/publication/318868196_On_the_Importance_of_Consistency_in_Training_Deep_Neural_Networks.
