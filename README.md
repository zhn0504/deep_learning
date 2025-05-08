# Introduction to Deep Learning

## 1. Project Overview
This code repository aims to document my learning journey in the field of deep learning, covering various aspects from fundamental theories to practical project applications. By continuously updating and refining the code, I hope to systematically organize the knowledge I've acquired. Meanwhile, it can also serve as a reference for others who are interested in deep learning.

## 2. Directory Structure
```plaintext
.
├── Deep_Learning_Code/  # Contains theoretical learning and experimental codes
│   ├── Chapter-01/
    ├── Chapter-02/
    ...
├── datasets/            # Datasets (Some large files are downloaded via Baidu Netdisk links)
├── README.md            # Project description file
└── requirements.txt     # List of project dependency libraries
```

## 3. Learning Content
This repository covers the following content: 

### 3.1 Neural Networks
#### Chapter 01 
- The basic principles of neural networks and the structure of the Multi-Layer Perceptron (MLP).
- Forward propagation and backpropagation. 
- Regression and classification problems, including linear regression and multi-class classification.
  
#### Chapter 02, 03
- Common problems during the training process, such as overfitting and underfitting. Corresponding countermeasures are introduced, including regularization, Dropout and their code implementations. It also covers the problems of vanishing gradients and exploding gradients, as well as the read and write operations of model files.
- Gradient descent algorithms and their various variants, such as Stochastic Gradient Descent, Mini-Batch Gradient Descent, Momentum Method, AdaGrad Algorithm, RMSProp/Adadelta Algorithm, Adam Algorithm, etc. There are also examples of gradient descent algorithms and related content of learning rate schedulers.

### 3.2 Convolutional Neural Networks (CNN)
#### Chapter 04
- Convolutional layers and their common operations, as well as pooling layers.
- The LeNet convolutional neural network, and complex classic CNN models, such as AlexNet, VGGNet, GoogleNet, Residual Network (ResNet), Dense Connectivity Network (DenseNet), etc.
  
#### Chapter 05
- New convolution methods, such as pointwise convolution, depthwise convolution, depthwise separable convolution, etc.
- New models, such as Yolo, Unet, MobileNet, etc.

### 3.3 Recurrent Neural Networks (RNN)
- For sequential data, basic knowledge such as sequential modeling and text data preprocessing is introduced.
- The principles of recurrent neural networks, the backpropagation through time algorithm, and the code implementation of basic RNNs are explained.
- The problem of long-term dependencies is explored, and complex recurrent neural network structures are introduced, such as deep recurrent neural networks, bidirectional recurrent neural networks, Gated Recurrent Unit (GRU), Long Short-Term Memory Network (LSTM) and their code implementations.
- It also involves encoder-decoder networks, code implementation of sequence-to-sequence learning, beam search algorithm, as well as machine translation and related datasets.

### 3.4 Attention Neural Networks
- The basic concepts and calculation methods of the attention mechanism are elaborated, including key-value pair attention, multi-head attention, and self-attention mechanism.
- Attention pooling and its code implementation are given, and the Transformer model and its code implementation are explained in detail.
- Further introduction of complex attention neural network models, such as BERT model, GPT series (GPT2/GPT3), T5 model, Vision Transformer (ViT) model, Swin Transformer, and the code implementation of GPT, etc.

### 3.5 Generative Models and Applications
- Deep generative models, such as Monte Carlo method, variational inference, variational autoencoders, Generative Adversarial Networks (GAN), and Diffusion models.
- Practical projects in the field of computer vision, such as custom data augmentation, transfer learning, application of classic visual datasets, and the practical project of "Dogs vs. Cats".
- In nlp, it involves word embeddings (such as word2vec), semantic search, pre-trained language models, introduction of the HuggingFace library, classic NLP datasets, and the practical project of "Sentiment Analysis of Movie Reviews". 

### 3.6 Cutting-edge Content
- Introduction to multi-modal AI and related content generation models, such as Instruct GPT (ChatGPT), Dall-E model.
- Analysis of the latest development trends in deep learning, and suggestions for the next step of learning are given.

## 4. Contact Information
If you have any questions or want to discuss topics related to deep learning, you can contact me through the following methods:
- **Email**：[zhn0504@outlook.com](mailto:zhn0504@outlook.com)
- **GitHub**：Welcome to visit my [GitHub Profile](https://github.com/zhn0504). On the GitHub repository page of this code repository, you can perform the following operations:
    - **View the Code**：Browse the code files directly in the repository to understand the specific implementation of the project.
    - **Submit an Issue**：If you find problems with the code, have feature requests, or any questions, you can submit a new issue on the [Issues Page](https://github.com/your_github_username/your_repo_name/issues).
    - **Initiate a Pull Request**：If you have suggestions for improving the code, you are welcome to Fork this repository, make modifications, and then initiate a Pull Request. Let's improve this project together.
 
## 5. Copyright Notice
The code and documents in this project are for learning and reference only. Without authorization, please do not use them for commercial purposes.
