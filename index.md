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

### 2.1 消息传递

在DGL中，消息函数 接受一个参数 edges，这是一个 EdgeBatch 的实例， 在消息传递时，它被DGL在内部生成以表示一批边。 edges 有 src、 dst 和 data 共3个成员属性， 分别用于访问源节点、目标节点和边的特征。

聚合函数 接受一个参数 nodes，这是一个 NodeBatch 的实例， 在消息传递时，它被DGL在内部生成以表示一批节点。 nodes 的成员属性 mailbox 可以用来访问节点收到的消息。 一些最常见的聚合操作包括 sum、max、min 等。

更新函数 接受一个如上所述的参数 nodes。此函数对聚合函数的聚合结果进行操作， 通常在消息传递的最后一步将其与节点的特征相结合，并将输出作为节点的新特征。

DGL在命名空间 dgl.function 中实现了常用的消息函数和聚合函数作为内置函数。 一般来说，DGL建议尽可能使用内置函数，因为它们经过了大量优化，并且可以自动处理维度广播。

内置消息函数可以是一元函数或二元函数。对于一元函数，DGL支持 copy函数。对于二元函数， DGL现在支持 add、 sub、 mul、 div、 dot 函数。消息的内置函数的命名约定是u 表示 源 节点，v表示目标节点，e表示边。这些函数的参数是字符串，指示相应节点和边的输入和输出特征字段名。 

关于内置函数的列表，请参见 [DGL Built-in Function](https://docs.dgl.ai/api/python/dgl.function.html#api-built-in)。

例如，要对源节点的 hu 特征和目标节点的 hv 特征求和， 然后将结果保存在边的 he 特征上，用户可以使用内置函数

```markdown

dgl.function.u_add_v('hu', 'hv', 'he')
```

DGL支持内置的聚合函数 sum、 max、 min 和 mean 操作。 聚合函数通常有两个参数，它们的类型都是字符串。一个用于指定 mailbox 中的字段名，一个用于指示目标节点特征的字段名， 例如， 

```markdown

dgl.function.sum('m', 'h')
```

在DGL中，也可以在不涉及消息传递的情况下，通过 apply_edges() 单独调用逐边计算。 apply_edges() 的参数是一个消息函数。并且在默认情况下，这个接口将更新所有的边。例如：

```markdown

import dgl.function as fn
graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
```

updaste_all() 的参数是一个消息函数、一个聚合函数和一个更新函数。示例正确用法如下：

```markdown

def updata_all_example(graph):
    # 在graph.ndata['ft']中存储结果
    graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                     fn.sum('m', 'ft'))
    # 在update_all外调用更新函数
    final_ft = graph.ndata['ft'] * 2
    return final_ft
```

**一些优化**

基于以上内容，DGL优化了消息传递的内存消耗和计算速度。利用这些优化的一个常见实践是通过基于内置函数的 update_all() 来开发消息传递功能。

**2.1.1 线性拆分**

有一个以下操作：拼接源节点和目标节点特征， 然后应用一个线性层，即 W×(u||v)。 源节点和目标节点特征维数较高，而线性层输出维数较低。

建议的实现是将线性操作分成两部分，一个应用于 源 节点特征，另一个应用于 目标 节点特征。 在最后一个阶段，在边上将以上两部分线性操作的结果相加，即执行 Wl×u+Wr×v， 因为 W×(u||v)=Wl×u+Wr×v，其中 Wl 和 Wr 分别是矩阵 W 的左半部分和右半部分：

```markdown

import dgl.function as fn

linear_src = nn.Parameter(torch.FloatTensor(size=(node_feat_dim, out_dim)))
linear_dst = nn.Parameter(torch.FloatTensor(size=(node_feat_dim, out_dim)))
out_src = g.ndata['feat'] @ linear_src
out_dst = g.ndata['feat'] @ linear_dst
g.srcdata.update({'out_src': out_src})
g.dstdata.update({'out_dst': out_dst})
g.apply_edges(fn.u_add_v('out_src', 'out_dst', 'out'))
```

**2.1.2 加权重**

一类常见的图神经网络建模的做法是在消息聚合前使用边的权重， 比如在 图注意力网络(GAT) 和一些 GCN的变种 。 DGL的处理方法是：

- 将权重存为边的特征。

- 在消息函数中用边的特征与源节点的特征相乘。

例如：

```markdown

import dgl.function as fn

# 假定eweight是一个形状为(E, *)的张量，E是边的数量。
graph.edata['a'] = eweight
graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                 fn.sum('m', 'ft'))
```

**2.1.3 异构图**

异构图是包含不同类型的节点和边的图。 不同类型的节点和边常常具有不同类型的属性。这些属性旨在刻画每一种节点和边的特征。

在使用图神经网络时，根据其复杂性，可能需要使用不同维度的表示来对不同类型的节点和边进行建模。

异构图上的消息传递可以分为两个部分：

- 对每个关系计算和聚合消息。

- 对每个结点聚合来自不同关系的消息。

在DGL中，对异构图进行消息传递的接口是 multi_update_all()。 multi_update_all() 接受一个字典。这个字典的每一个键值对里，键是一种关系， 值是这种关系对应 update_all() 的参数。

multi_update_all() 还接受一个字符串来表示跨类型整合函数，来指定整合不同关系聚合结果的方式。 这个整合方式可以是 sum、 min、 max、 mean 和 stack 中的一个。

```markdown

import dgl.function as fn

for c_etype in G.canonical_etypes:
    srctype, etype, dsttype = c_etype
    Wh = self.weight[etype](feat_dict[srctype])
    # 把它存在图中用来做消息传递
    G.nodes[srctype].data['Wh_%s' % etype] = Wh
    # 指定每个关系的消息传递函数：(message_func, reduce_func).
    # 注意结果保存在同一个目标特征“h”，说明聚合是逐类进行的。
    funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
# 将每个类型消息聚合的结果相加。
G.multi_update_all(funcs, 'sum')
# 返回更新过的节点特征字典
return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}
```

### 2.2 让我们来一起搭模型！

搭模型就像玩乐高，拼拼凑凑即可，没有太多的coding要求。对于DGL，NN模块就是我们要玩的东西，在这里选择继承PyTorch的NN模块！

**2.2.1 构造函数**

先来看看构造函数！以GraphSAGE为例子，以下代码主要是完成设置选项与初始化参数的功能：

```markdown

import torch.nn as nn

from dgl.utils import expand_as_pair

class SAGEConv(nn.Module):
    def __init__(self,
                 in_feats, #输入维度，输入维度可以分为源节点特征维度和目标节点特征维度
                 out_feats, #输出维度
                 aggregator_type, #聚合类型，常用的聚合类型包括 mean、 sum、 max 和 min。一些模块可能会使用更加复杂的聚合函数，比如 lstm
                 bias=True,
                 norm=None, #norm 是用于特征归一化的可调用函数。在SAGEConv论文里，归一化可以是L2归一化: hv=hv/||h||^2。
                 activation=None):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats) #输入维度可以分为源节点特征维度和目标节点特征维度
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.activation = activation
```

SAGEConv中，子模块根据聚合类型而有所不同。这些模块是纯PyTorch NN模块，例如 nn.Linear、 nn.LSTM 等。 构造函数的最后调用了 reset_parameters() 进行权重初始化。

**2.2.2 forward函数**

2.2.2.1 首先会检查输入图对象是否规范

forward() 函数需要处理输入的许多极端情况，这些情况可能导致计算和消息传递中的值无效。 比如在 GraphConv 等conv模块中，DGL会检查输入图中是否有入度为0的节点。 

当1个节点入度为0时， mailbox 将为空，并且聚合函数的输出值全为0， 这可能会导致模型性能不佳。

但是，在 SAGEConv 模块中，被聚合的特征将会与节点的初始特征拼接起来， forward() 函数的输出不会全为0。在这种情况下，无需进行此类检验。

```markdown

def forward(self, graph, feat):
    with graph.local_scope():
        # 指定图类型，然后根据图类型扩展输入特征
        feat_src, feat_dst = expand_as_pair(feat, graph) #就是要拿到区分开的输入输出特征
```

对于同构图上的全图训练，源节点和目标节点相同，它们都是图中的所有节点。

在异构图的情况下，图可以分为几个二分图，每种关系对应一个。关系表示为 (src_type, edge_type, dst_dtype)。 当输入特征 feat 是1个元组时，图将会被视为二分图。元组中的第1个元素为源节点特征，第2个元素为目标节点特征。

在小批次训练中，计算应用于给定的一堆目标节点所采样的子图。子图在DGL中称为区块(block)。 在区块创建的阶段，dst nodes 位于节点列表的最前面。通过索引 [0:g.number_of_dst_nodes()] 可以找到 feat_dst。

2.2.2.2 消息聚合和传递

```markdown

import dgl.function as fn
import torch.nn.functional as F
from dgl.utils import check_eq_shape

if self._aggre_type == 'mean':
    graph.srcdata['h'] = feat_src
    graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
    h_neigh = graph.dstdata['neigh']
elif self._aggre_type == 'gcn':
    check_eq_shape(feat)
    graph.srcdata['h'] = feat_src
    graph.dstdata['h'] = feat_dst
    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
    # 除以入度
    degs = graph.in_degrees().to(feat_dst)
    h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
elif self._aggre_type == 'max_pool':
    graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
    graph.update_all(fn.copy_u('h', 'm'), fn.max('m', 'neigh'))
    h_neigh = graph.dstdata['neigh']
else:
    raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

# GraphSAGE中gcn聚合不需要fc_self
if self._aggre_type == 'gcn':
    rst = self.fc_neigh(h_neigh)
else:
    rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
```

**2.2.3 异构图**



### 2.3 Training

### 2.4 Mini-batch Training

## 基于OGB 的数据处理过程

## 基于DGL的经典示例代码解析
