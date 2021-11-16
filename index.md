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

## 基于Deep Graph Library (DGL) 的模型搭建过程

Why is DGL not PyG？

PyG太麻烦，另外维护的人太少了，仅此而已，听话，用DGL，API足够用。

### 1 消息传递

在DGL中，消息函数 接受一个参数 edges，这是一个 EdgeBatch 的实例， 在消息传递时，它被DGL在内部生成以表示一批边。 edges 有 src、 dst 和 data 共3个成员属性， 分别用于访问源节点、目标节点和边的特征。

聚合函数 接受一个参数 nodes，这是一个 NodeBatch 的实例， 在消息传递时，它被DGL在内部生成以表示一批节点。 nodes 的成员属性 mailbox 可以用来访问节点收到的消息。 一些最常见的聚合操作包括 sum、max、min 等。

更新函数 接受一个如上所述的参数 nodes。此函数对聚合函数的聚合结果进行操作， 通常在消息传递的最后一步将其与节点的特征相结合，并将输出作为节点的新特征。

DGL在命名空间 dgl.function 中实现了常用的消息函数和聚合函数作为内置函数。 一般来说，DGL建议尽可能使用内置函数，因为它们经过了大量优化，并且可以自动处理维度广播。

内置消息函数可以是一元函数或二元函数。对于一元函数，DGL支持 copy函数。对于二元函数， DGL现在支持 add、 sub、 mul、 div、 dot 函数。消息的内置函数的命名约定是u 表示 源 节点，v表示目标节点，e表示边。这些函数的参数是字符串，指示相应节点和边的输入和输出特征字段名。 关于内置函数的列表，请参见 [DGL Built-in Function](https://docs.dgl.ai/api/python/dgl.function.html#api-built-in)。

例如，要对源节点的 hu 特征和目标节点的 hv 特征求和， 然后将结果保存在边的 he 特征上，用户可以使用内置函数

```markdown
Syntax highlighted code block

`dgl.function.u_add_v('hu', 'hv', 'he')`
```

DGL支持内置的聚合函数 sum、 max、 min 和 mean 操作。 聚合函数通常有两个参数，它们的类型都是字符串。一个用于指定 mailbox 中的字段名，一个用于指示目标节点特征的字段名， 例如， 

```markdown
Syntax highlighted code block

`dgl.function.sum('m', 'h')`
```

在DGL中，也可以在不涉及消息传递的情况下，通过 apply_edges() 单独调用逐边计算。 apply_edges() 的参数是一个消息函数。并且在默认情况下，这个接口将更新所有的边。例如：
```markdown
Syntax highlighted code block

`import dgl.function as fn
graph.apply_edges(fn.u_add_v('el', 'er', 'e'))`
```

updaste_all() 的参数是一个消息函数、一个聚合函数和一个更新函数。示例正确用法如下：

```markdown
Syntax highlighted code block

`def updata_all_example(graph):
    # 在graph.ndata['ft']中存储结果
    graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                     fn.sum('m', 'ft'))
    # 在update_all外调用更新函数
    final_ft = graph.ndata['ft'] * 2
    return final_ft`
```

一些优化：
基于以上内容，DGL优化了消息传递的内存消耗和计算速度。利用这些优化的一个常见实践是通过基于内置函数的 update_all() 来开发消息传递功能。

有一个以下操作：拼接 源 节点和 目标 节点特征， 然后应用一个线性层，即 W×(u||v)。 源 节点和 目标 节点特征维数较高，而线性层输出维数较低。

一类常见的图神经网络建模的做法是在消息聚合前使用边的权重， 比如在 图注意力网络(GAT) 和一些 GCN的变种 。 DGL的处理方法是：

将权重存为边的特征。

在消息函数中用边的特征与源节点的特征相乘。

例如：

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
