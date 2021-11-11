# 欢迎来到John LIU的图神经网络学习笔记

本篇笔记将主要采用简体中文进行记录，主要涵括内容有：

1. DGL以及OGB环境安装及配置
2. 基于DGL的基本模型搭建过程
3. 基于OGB的数据集使用
4. 后期考虑添加一些经典论文的讲解和复现 ...

本文所主要参考的资料有：

1. [DGL](https://docs.dgl.ai/guide_cn/index.html)
2. [OGB](https://ogb.stanford.edu/)

本篇笔记所有权归澳门大学智慧城市物联网国家重点实验室智能交通课题组所有.

## 环境配置

工欲善其事，必先利其器。装环境对于每一个研究人员来说都是一件非常disgusting的事情，在此我将会给出一个非常一劳永逸的环境安装方法。

所需安装的工具如下：

1. 一台带有显卡的电脑（比如我的会发光的3090...）
2. Anaconda
3. CUDA & CUDANN
4. PyTorch
5. DGL
6. OGB

对于Anaconda + CUDA + PyTorch的安装过程，在此建议参考此篇博客[安装](https://blog.csdn.net/u012369535/article/details/106950286/)。

对于该部分环境的配置，我强烈建议使用Anaconda和Jupyter来进行程序的编写，Jupyter可以将文本和代码结合，用过都说好，安装方法也极其简单，可以通过Anaconda Navigator来进行傻瓜式安装和打开。

此过程需要特别注意：

1. 在Anaconda安装时需要选择合适的Python版本，目前PyTorch只可以支持到Python 3.8，在此建议安装Python 3.8.
2. 安装CUDA过程中需要先通过'nvidia-msi'命令查看显卡驱动情况，[此处](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)查看对应所可以安装的CUDA版本情况。
3. 基于目前CUDA 11.0+并没有合适的CUDANN和DGL同时支持的能力，在此建议安装CUDA 10.2，虽然老一些，但是稳定很多。

对于DGL的安装，建议直接通过官网介绍进行安装，[链接在此](https://www.dgl.ai/pages/start.html)。切记需要选择合适的CUDA及Python版本，另外需要提前安装[VC2015 Redistributable](https://www.microsoft.com/en-us/download/details.aspx?id=48145)。

对于OGB的安装，还是官网安装方法更方便，[链接在此](https://ogb.stanford.edu/docs/home/)

## Deep Graph Library (DGL)

Why is DGL not PyG？

PyG太麻烦，另外维护的人太少了，仅此而已，听话，用DGL，API足够用。

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```
