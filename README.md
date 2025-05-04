# Introduction to Deep Learning

## 1. Project Overview
This code repository aims to document my learning journey in the field of deep learning, covering various aspects from fundamental theories to practical project applications. By continuously updating and refining the code, I hope to systematically organize the knowledge I've acquired. Meanwhile, it can also serve as a reference for others who are interested in deep learning.

## 2. Directory Structure
```plaintext
.
├── Deep_Learning_Code/  # Contains theoretical learning and experimental codes
│   ├── Chapter-00/
    ├── Chapter-01/
    ├── Chapter-02/
    ...
├── datasets/            # Datasets (Some large files are downloaded via Baidu Netdisk links)
├── README.md            # Project description file
└── requirements.txt     # List of project dependency libraries
```

## 3. Learning Content
This repository covers the following content:  

### Principles and Practices of Neural Networks
### Chapter 01 
- The basic principles of neural networks and the structure of the Multi-Layer Perceptron (MLP).
- The mechanisms of forward propagation and backpropagation. 
- Regression and classification problems, including linear regression and multi-class classification.
  
### Chapter 02, 03
- Common problems during the training process, such as overfitting and underfitting. Corresponding countermeasures are introduced, including regularization, Dropout and their code implementations. It also covers the problems of vanishing gradients and exploding gradients, as well as the read and write operations of model files.
- Gradient descent algorithms and their various variants, such as Stochastic Gradient Descent, Mini-Batch Gradient Descent, Momentum Method, AdaGrad Algorithm, RMSProp/Adadelta Algorithm, Adam Algorithm, etc. There are also examples of gradient descent algorithms and related content of learning rate schedulers.

### Convolutional Neural Networks (CNN)
### Chapter 04
- Convolutional layers and their common operations, as well as pooling layers.
- The LeNet convolutional neural network, and complex classic CNN models, such as AlexNet, VGGNet, GoogleNet, Residual Network (ResNet), Dense Connectivity Network (DenseNet), etc.
  
### Chapter 05
- New convolution methods, such as pointwise convolution, depthwise convolution, depthwise separable convolution, etc.
- New models, such as Yolo, Unet, MobileNet, etc.

### Recurrent Neural Networks (RNN)
- 针对序列数据，介绍序列建模、文本数据预处理等基础知识。  
- 讲解循环神经网络原理、随时间反向传播算法以及基础 RNN 的代码实现。  
- 探讨长期依赖问题，并介绍复杂循环神经网络结构，如深度循环神经网络、双向循环神经网络、门控循环单元（GRU）、长短时记忆网络（LSTM）及其代码实现。  
- 还涉及编码器 - 解码器网络、序列到序列学习代码实现、束搜索算法以及机器翻译与相关数据集等内容。  

### Attention Neural Networks
- 阐述注意力机制的基本概念、计算方式，包括键值对注意力和多头注意力、自注意力机制。  
- 给出注意力池化及代码实现，详细讲解 Transformer 模型及其代码实现。  
- 进一步介绍复杂注意力神经网络模型，如 BERT 模型、GPT 系列（GPT2/GPT3）、T5 模型、ViT 模型、Swin Transformer 以及 GPT 代码实现等内容。  

### Generative Models and Applications
- 介绍深度生成模型相关知识，如蒙特卡洛方法、变分推断、变分自编码器、生成对抗网络（GAN）、Diffusion 扩散模型。  
- 包含计算机视觉领域的项目实战，如自定义数据增强、迁移学习、经典视觉数据集应用，以及“猫狗大战（Dogs vs. Cats）”项目实战。  
- 在自然语言处理方面，涉及词嵌入（如 word2vec）、词义搜索、预训练语言模型、HuggingFace 库介绍、经典 NLP 数据集以及“电影评论情感分析”项目实战。  

### Cutting-edge Content
- 介绍多模态 AI 及内容生成相关模型，如 Instruct GPT（ChatGPT）、Dall - E 模型。  
- 分析深度学习最新发展趋势，并给出下一步学习建议。  

## 4. Contact Information
If you have any questions or want to discuss topics related to deep learning, you can contact me through the following methods:
- **Email**：[zhn0504@outlook.com](mailto:zhn0504@outlook.com)
- **GitHub**：Welcome to visit my [GitHub Profile](https://github.com/zhn0504). On the GitHub repository page of this code repository, you can perform the following operations:
    - **View the Code**：Browse the code files directly in the repository to understand the specific implementation of the project.
    - **Submit an Issue**：If you find problems with the code, have feature requests, or any questions, you can submit a new issue on the [Issues Page](https://github.com/your_github_username/your_repo_name/issues).
    - **Initiate a Pull Request**：If you have suggestions for improving the code, you are welcome to Fork this repository, make modifications, and then initiate a Pull Request. Let's improve this project together.
 
## 5. Copyright Notice
The code and documents in this project are for learning and reference only. Without authorization, please do not use them for commercial purposes.
