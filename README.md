# GTA-Domain-Adaptation

We study the problem to domain adaptation between GTA and cityscapes datasets.
Hyperparameters are defined in *params.yaml*.
We implement two different architectures, namely DualGAN's and CycleGAN.
Both these architectures use two Generative Adversarial Networks and use cycle-constistency to incorporate reconstruction losses.
We use a semantic segmentation model use compare the performance of our models.

To replicate the training, run the *train.ipynb* ipython notebook.
The demo can be run using *demo.ipynb*
