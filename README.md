# Deep-Learning-Paper-List

## Excellent Slides

  - [Deep Reinforcement Learning](http://hunch.net/~beygel/deep_rl_tutorial.pdf) David Silver. ICML 2016 Tutorial.
  - [Memory Networks for Language Understanding](http://www.thespermwhale.com/jaseweston/icml2016/icml2016-memnn-tutorial.pdf) Jason Weston. ICML 2016 Tutorial.
  - [Deep Residual Networks: Deep Learning Gets Way Deeper](http://icml.cc/2016/tutorials/icml2016_tutorial_deep_residual_networks_kaiminghe.pdf) Kaiming He. ICML 2016 Tutorial.
  - [Recent Advances in Non-Convex Optimization](http://newport.eecs.uci.edu/anandkumar/slides/icml2016-tutorial.pdf) Anima Anandkumar. ICML 2016 Tutorial.


## Architecture

### Recurrent Neural Networks(RNNs)
  - [Long short-term memory](http://web.eecs.utk.edu/~itamar/courses/ECE-692/Bobby_paper1.pdf) ** LSTM ** S. Hochreiter and J. Schmidhuber.
  - [On the properties of neural machine translation: Encoder-decoder approaches](https://arxiv.org/abs/1409.1259) ** GRU ** Kyunghyun Cho, Bart van Merrienboer, Dzmitry Bahdanau, Yoshua Bengio.
  - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555) ** LSTM v.s. GRU ** Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, Yoshua Bengio.
  - [Training RNNs as Fast as CNNs](https://arxiv.org/abs/1709.02755) ** SRU ** Tao Lei, Yu Zhang
  - [Optimizing performance of recurrent neural networks on gpus](https://arxiv.org/abs/1604.01946) ** cuDNN LSTM ** Jeremy Appleyard, Tomas Kocisky, Phil Blunsom.


### Convolutional Neural Networks(CNNs)
  - [Deep Residual Learning for Image Recognition]() Kaiming. He, Xiangyu. Zhang, Shaoqing. Ren, Jian. Sun. CVPR, 2016.
  - [Identity Mappings in Deep Residual Networks]() Kaiming. He, Xiangyu. Zhang, Shaoqing. Ren, Jian. Sun. ECCV, 2016.
  - [Deep Residual Networks: Deep Learning Gets Way Deeper]() Kaiming He. ICML Tutorial, 2016.
  - [Residual Networks Behave Like Ensembles of Relatively Shallow Networks]() Andreas. Veit, Micheal. Wilber, Serge. Belongie. 2016.
  - [Very deep convolutional networks for large-scale image recognition]() K. Simonyan and A. Zisserman. 2014.

## Loss & Optimization

### Global and Local Minima, Saddle Points
  - [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836) N. S. Keskar, D. Mudidere, J. Nocedal, M. Smelyanskiy, P. T. P. Tang.
  - [The Loss Surfaces of Multilayer Networks](https://arxiv.org/abs/1412.0233) A. Choromanska, M. Henaff, M. Mathieu, G. B. Arous, Y. LeCun.
  - [Escaping from Saddle Points](https://arxiv.org/abs/1503.02101) - Online Stochastic Gradients for Tensor Decomposition. Rong Ge, Furong Huang, Chi Jin, Yang Yuan.
  - [Deep Learning without Poor Local Minima](https://arxiv.org/abs/1605.07110) K. Kawaguchi.

### Optimization Algorithm

  - [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) G. Hinton, Nitish Srivastava, Kevin Swersky.
  - [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf). J. Duchi, E. Hazan, Y. Singer. JMLR, 2011.
  - [Adadelta - an adaptive learning rate method](http://arxiv.org/abs/1212.5701) Matthew D. Zeiler
  - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8) Diederik P. Kingma, Jimmy Ba.
  - [Incorporating Nesterov Momentum into Adam](http://cs229.stanford.edu/proj2015/054_report.pdf) Timothy Dozat.
  - [Large Scale Distributed Deep Networks](), Jeffrey Dean, Greg S. Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Quoc V. Le, Mark Z. Mao, Marc’Aurelio Ranzato, Andrew Senior, Paul Tucker, Ke Yang, Andrew Y. Ng. NIPS, 2012.
  - [Asynchronous Stochastic Gradient Descent with Delay Compensation for Distributed Deep Learning](), Shuxin Zheng, Qi Meng, Taifeng Wang, Wei Chen, Nenghai Yu, Zhi-Ming Ma, Tie-Yan Liu.
  
### Gradient Vanishing & Activations

### Others

 - [Batch normalization: Accelerating deep network training by reducing internal covariate shift]() S. Loffe and C. Szegedy.
  
## Regularization

 - [Dropout: A simple way to prevent neural networks from overfitting]()  N. Srivastava et al.
 - [Improving neural networks by preventing co-adaptation of feature detectors]()  G. Hinton et al.


# Applications

## Natural Language Processing

### Embeddings

  - [Efficient Estimation of Word Representations in Vector Space]() Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
  - [Distributed Representations of Words and Phrases and their Compositionality]() T. Mikolov, I. Sutskever, Kai Chen, G. Corrado, J. Dean
  - [Representations in Vector Space Efficient Estimation of Word Representations in Vector Space]() T. Mikolov, Kai Chen, G. Corrado, J. Dean
  - [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) Jeffrey Pennington, Richard Socher, Christopher D. Manning
  - [Neural Word Embedding as Implicit Matrix Factorization]() O. Levy, Y. Goldberg.

### Text Classification

  - [Convolutional Neural Networks for Sentence Classificatio]() Yoon Kim.
  - [Bag of Tricks for Efficient Text Classification]() A. Joulin, E. Grave, P. Bojanowski, T. Mikolov. 
  - [Text understanding from scratch]()  Xiang Zhang and Yann LeCun. 2015.
  - [Character-level convolutional networks for text classification]()  Xiang Zhang, Junbo Zhao, and Yann LeCun. NIPS, 2015.

### Machine Translation

  - [Statistical Phrase-based Translation]() P. Koehn, F. J. Och, D. Marcu. NAACL, 2003.
  - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches]() K. Cho, B. v. Merrienboer, D. Bahdanau, Y. Bengio
  - [Recurrent Continuous Translation Models]() N. Kalchbrenner, P. Blunsom. 2013.
  - [Sequence to Sequence Learning with Neural Networks]() I. Sutskever, O. Vinyals, Q. V. Le. 2014.
  - [Neural Machine Translation by Jointly Learning to Align and Translate]() D. Bahdanau, K. Cho, Y. Bengio, 2014.
  - [On using very large target vocabulary for neural machine translation]() S. Jean, K. Cho, R. Memisevic, Y. Bengio. ACL, 2015.
  - [Addressing the Rare Word Problem in Neural Machine Translation]() Minh-Thang Luong, I. Sutskever, Q. V. Le, O. Vinyals, W. Zaremba.
  - [Neural Machine Translation of Rare Words with Subword Units]() R. Sennrich, B. Haddow, A. Birch.
  - [Deep Residual Learning for Image Recognition]() Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. CVPR, 2015.

### Question Answering

  - [Memory Networks. J. Weston, S. Chopra, A. Bordes]() arXiv, 2015.
  - [End-to-End Memory Networks]() S. SukhbaatarM A. Szlam, J. Weston, R. Fergus. arXiv, 2015.
  - [Iterative Alternating Neural Attention for Machine Reading]() A. Sordoni, P. Bachman, Y. Bengio. arXiv, 2016.
  - [Key-Value Memory Networks for Directly Reading Documents]() A. Miller, A. Fisch, J. Dodge, A. Karimi, A. Bordes, J. Weston. arXiv, 2016.
  - [Question Answering with Subgraph Embedding]() A. Bordes, J. Weston, S. Chopra. arXiv, 2014

### Other Topics

  - [Dependency Parsing](https://github.com/kemaswill/Deep-Learning-Paper-List/blob/master/NLP/Other_Topics_of_NLP.md)

## Image Classification

  - [ImageNet Classification with Deep Convolutional
     Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton. NIPS.
  - [Deep Residual Learning for Image Recognition]() Kaiming. He, Xiangyu. Zhang, Shaoqing. Ren, Jian. Sun. CVPR, 2016.
  - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852) Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  - [OverFeat: Integrated recognition, localization and detection using convolutional networks]() P. Sermanet et al.

## Knowledge Graph

  - [Translating Embeddings for Modeling Multi-relational Data]() A. Bordes, N. Usunier, A. Garcia-Duran, Jason Weston, O. Yakhnenko. NIPS, 2013.
  - [Knowledge Graph Embedding by Translating on Hyperplanes]() Zhen Wang, Jianwen Zhang, Jianlin Feng, Zheng Chen. AAAI, 2014.
  - [Learning Entity and Relation Embeddings for Knowledge Graph Completion]() Y. Lin, Zhuyuan Liu, Maosong Sun, Yang Liu, Xuan Zhu. AAAI, 2015.
  - [Knowledge Graph Embedding via Dynamic Mapping Matrix]() G. Ji, Shizhu He, Liheng Xu, Kang Liu, Jun Zhao. ACL, 2015.
  - [Reasoning With Neural Tensor Networks for Knowledge Base Completion]() R. Socher, Danqi Chen, C. Manning, A. Ng. NIPS, 2013.
  
## Generative Adversarial Networks
  
  - [Generative adversarial nets]()  I. Goodfellow et al.
  - [Improved techniques for training GANs]() T. Salimans et al.
  - [Wasserstein GAN](https://arxiv.org/abs/1701.07875) Martin Arjovsky, Soumith Chintala, Léon Bottou

## Deep Reinforcement Learning
  - [Human-level control through deep reinforcement learning]() V. Mnih et al.
  - [Playing atari with deep reinforcement learning ]() V. Mnih et al.
  - [Asynchronous methods for deep reinforcement learning]() V. Mnih et al.
  - [Deep Reinforcement Learning with Double Q-Learning]() H. Hasselt et al.
  - [Mastering the game of Go with deep neural networks and tree search]() D. Silver et al.
