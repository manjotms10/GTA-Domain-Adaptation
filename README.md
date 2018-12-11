# GTA-Domain-Adaptation

We study the problem to domain adaptation between GTA and cityscapes datasets. <br>
Hyperparameters are defined in *params.yaml*. <br>
We implement two different architectures, namely DualGAN's and CycleGAN. <br>
Both these architectures use two Generative Adversarial Networks and use cycle-constistency to incorporate reconstruction losses. <br><br>
We use a semantic segmentation model to compare the performance of our models. <br>

<br><br><br>
To replicate the training, run the *train.ipynb* ipython notebook. <br>
The demo can be run using *demo.ipynb*. <br>
