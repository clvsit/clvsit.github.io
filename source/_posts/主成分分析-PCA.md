---
title: 主成分分析 PCA
date: 2019-04-19 16:21:57
top_img: https://i.loli.net/2019/03/28/5c9c3e0dbbf95.jpg
cover: https://i.loli.net/2019/03/28/5c9c3e0dbbf95.jpg
mathjax: true
tags:
category:
- 机器学习
- 降维
---

主成分分析（Principal Component Analysis，简称 PCA）是一种统计方法，通过正交变换将一组可能存在相关性的变量转换为一组线性不相关的变量，转换后的这组变量称为主成分。PCA 旨在找到数据中的主成分，并利用这些主成分表征原始数据，从而达到降维的目的。

简单地说，PCA 就是要找出原始数据集中最主要的内容（特征），用这些内容来代替原始数据集。

【示例】：在三维空间中有一系列数据点，这些点分布在一个过原点的平面上。如果我们用自然坐标系 x,y,z 三个轴来表示数据，就需要使用三个维度。而实际上，这些点只出现在一个二维平面上，如果我们**通过坐标系旋转变换（获得新坐标系）**使得数据所在平面与 x,y 平面重合，那么我们就可以通过 x′,y′ 两个维度表达原始数据，并且没有任何损失，这样就完成了数据的降维。而 x′,y′ 两个轴所包含的信息就是我们要找到的主成分。

一般地，将原始数据集从 m 维降到 n 维（n < m）都会存在一定的损失，我们的目标是让损失尽可能地小。

【优点】：通过上面的介绍，我们可以归纳出 PCA 的优点。
- 降低数据的复杂性
- 识别最重要的多个特征

# 最大方差理论
在高维空间中，我们往往不能像刚才那样直观地想象出数据的分布形式，也就更难精确地找到主成分对应的轴是哪些。不妨从最简单的二维数据来看看 PCA 究竟是如何工作的。

![最大方差理论](https://i.loli.net/2019/03/27/5c9b0ca59867c.jpg)

图(a)是二维空间中经过中心化的一组数据，通过图像可以很容易看出主成分所在轴的大致方向，即图(b)黄线所处的轴。因为在黄线所处的轴上，数据分布得更为分散，这也意味着数据在该方向上方差更大。

【问】：为什么要选择数据方差更大的方向？

【答】：在信号处理领域，我们认为信号具有较大方差，噪声具有较小方差，信号与噪声之比称为信噪比。信噪比越大意味着数据的质量越好，反之，信噪比越小意味着数据的质量越差。从数据分布角度来看，选择方差更大的方向能够使得样本投影后更为分散、可以保留最多的信息，不易造成损失；而选择方差较小的方向会使得投影后的样本出现重叠，造成损失。

由此引出 PCA 的目标：**最大化投影方差，让数据在超平面上投影的方差最大**。

## 目标函数及求解方法
在 PCA 中，数据从原来的坐标系转换到新的坐标系，新坐标系的选择是由数据本身决定的。

给定一组样本 $\{v_1, v_2, \cdots, v_n\}$，其中所有向量均为列向量。首先对样本进行中心化处理，使得 $\sum_i x_i = 0$：

$$
\mu = \frac{1}{n}\sum_{i=1}^nv_i \\
\{x_1, x_2, \cdots, x_n\} = \{v_1-\mu, v_2-\mu,\cdots,v_n-\mu\} \\
\sum_{i=1}^n x_i = \sum_{i=1}^n v_i - n\mu = 0
$$

![中心化处理](https://i.loli.net/2019/04/16/5cb5966876bce.jpg)

通过观察上面的两张图可以明显地发现，对样本进行中心化处理实际上是将样本进行了平移，将多维空间的原点作为样本集的中心点。那么，为什么要对样本集做中心化处理？先将这个问题保留在心中，接着往下看。

再假定投影变换后得到的新坐标系为 $\{w_1, w_2, \cdots, w_d\}$，其中 $w_i$ 是标准正交基向量，$||w_i||_2 = 1, \ w_i^Tw_j = 0(i \neq j)$。

向量内积在几何上表示为第一个向量投影到第二个向量上的长度。以二维空间为例，A(1, 0)、B(0, 1) 分别为该空间的一组正交基，假设此时二维空间中有一个样本 X(3, 2)，X 分别与 A、B 的内积为 3 和 2。为二维空间建立坐标系 A 向量为 x 轴，B 向量为 y 轴，实际上 X 与 A、B 的内积为 X 投影到 A 和 B 的长度，恰好是 X 在 x 轴和 y 轴的坐标。扩展到多维空间，则多维空间中的一个向量与该多维空间的一组标准正交基的内积——当前向量在这组标准正交基建立的坐标系的坐标。

![坐标轴](https://i.loli.net/2019/04/16/5cb5e06825f23.jpg)

因此向量 $x_i$ 在新坐标系 $w = (w_1, w_2, \cdots, w_d)$ 上的投影坐标可以表示为 $x_i^Tw$。现在的目标是找到一个投影方向 w，使得原始样本的投影 $z_1, z_2, \cdots, z_n$ 在 w 上的投影方差尽可能大。

但如果单纯只选择方差最大的方向，后续方向应该会和方差最大的方向接近重合。为了让两个特征尽可能表示更多的原始信息，我们自然不希望它们之间存在线性相关性。

此时，优化目标为：
- 找到一个投影方向 w，使得原始样本的投影在 w 上的投影方差尽可能大；
- 投影后各特征之间不存在线性相关性。

那么，怎么衡量特征之间的相关性呢？可以用特征之间的协方差来进行衡量。若协方差为零，则表示当前特征无关；若为 1，则表示最大正相关；若为 -1，则表示最大负相关。若两个特征向量的协方差为零，反映到坐标系中就是这两个特征向量正交。

根据方差的计算公式 $D(X) = \sum_{i=1}^nx_i - E(X)$，我们需要计算原始样本投影变换后的坐标 $x_i^Tw$ 以及它们的均值 $\mu'$。

$$
x_1 + x_2 + \cdots + x_n = v_1 - \mu + v_2 - \mu + \cdots + v_n - \mu = 0 \\
\mu' = \frac{1}{n}\sum_{i=1}^nx_i^Tw = (\frac{1}{n}\sum_{i=1}^nx_i^T)w = 0
$$

由于对样本集做了中心化处理，使得投影之后的均值为 0，简化了方差的计算，这就是对样本集做中心化处理的原因。

投影后的方差可以表示为

$$
D(x) = \frac{1}{n}\sum_{i=1}^n(x_i^Tw - 0)^2 = \frac{1}{n}\sum_{i=1}^n(x_i^Tw)^T(x_i^Tw) \\
= \frac{1}{n}\sum_{i=1}^nw^Tx_ix_i^Tw = w^T(\frac{1}{n}\sum_{i=1}^nx_ix_i^T)w \quad (1)
$$

另外，由于 w 是单位向量，即有 $w^Tw = 1$。因此我们要求解一个最优化问题，可以表示为

$$
\begin{cases}
max\ tr(w^TXX^Tw), \\
s.t. \quad w^Tw = 1
\end{cases}
$$

引入拉格朗日乘子

$$
f = w^TXX^Tw - \lambda(1-w^Tw)
$$

对 w 求导并令其等于 0

$$
\frac{\partial f}{\partial w} = 2XX^Tw - 2\lambda w = 0
$$

可解得

$$
XX^Tw = \lambda w
$$

然后求特征值，并将特征值按从大到下的顺序排列，选择前 d 个特征值对应的特征向量作为 W。

【PCA 求解方法总结】：
1. 对样本数据进行中心化处理。
2. 求样本协方差矩阵。
3. 对协方差矩阵进行特征值分解，将特征值从大到小排列。
4. 取特征值前 d 大对应的特征向量 $w_1, w_2, \cdots, w_d$，通过以下映射将 n 维样本映射到 d 维。

$$
x_i' = \begin{bmatrix}
w_1^Tx_i \\
w_2^Tx_i \\
\cdots \\
w_d^Tx_i \\
\end{bmatrix}
$$

新的 $x_i'$ 的第 d 维就是 $x_i$ 在第 d 个主成分 $w_d$ 方向上的投影，通过选取最大的 d 个特征值对应的特征向量，将方差较小的特征（噪声）抛弃，使得每个 n 维列向量 $x_i$ 被映射为 d 维列向量 $x_i'$。

【降维的结果】：最小的 d - d' 个特征值的特征向量被舍弃了，但舍弃这部分信息往往是必要的。
- 舍弃这部分信息之后能使样本的采样密度增大，这正是降维的重要动机；
- 当数据受到噪声影响时，最小的特征值所对应的特征向量往往与噪声有关，将它们舍弃能在一定程度上起到去噪的效果。

# 最小平方误差理论
【问题】：PCA 求解的其实是最佳投影方向，即一条直线，这与数学中线性回归问题的目标不谋而合，能否从回归的角度定义 PCA 的目标并相应地求解问题呢？

继续考虑二维空间中的样本点。在最大方差理论中我们求得一条直线，使得样本点投影到该直线上的方差最大。从求解直线的思路出发，很容易联想到数学中的线性回归问题，其目标也是求解一个线性函数使得对应直线能够更好地拟合样本点集合。如果我们从该角度定义 PCA 的目标，那么问题就会转化为一个回归问题。

顺着这个思路，在高维空间中实际是要找到一个 d 维超平面，使得数据点到这个超平面的距离平方和最小。以 d = 1 为例，超平面退化为直线，即把样本点投影到最佳直线，最小化的就是所有点到直线的距离平方之和。

![最小平方误差理论](https://i.loli.net/2019/03/28/5c9c3e0dbbf95.jpg)

数据集中每个点 $x_k$ 到 d 维超平面 D 的距离

$$
distance(x_k, D) = ||x_k - \widetilde{x_k}||_2
$$

其中 $\widetilde{x_k}$ 表示 $x_k$ 在超平面 D 上的投影向量。如果该超平面由 d 个标准正交基 $W = \{w_1, w_2, \cdots, w_d\}$ 构成，根据线性代数理论 $\widetilde{x_k}$ 可以由这组基线性表示

$$
\widetilde{x_k} = \sum_{i=1}^d(w_i^Tx_k)w_i \quad (7)
$$

其中 $w_i^Tx_k$ 表示 $x_k$ 在 $w_i$ 方向上投影的长度。因此，$\widetilde{x_k}$ 实际上就是 $x_k$ 在 W 这组标准正交基下的坐标。而 PCA 要优化的目标为

$$
\begin{cases}
arg \ min_W\sum_{k=1}^n||x_k - \widetilde{x_k}||^2_2, \\
s.t. \quad w_i^Tw_j = \delta_{ij} = \begin{cases}
1, i = j; \\
0, i \not = j.
\end{cases}
\quad \forall_{i,j}
\end{cases}
\quad (8)
$$

也就是说，我们要在超平面 D 上找到一个方向使得

由向量内积的性质，可知 $x_k^T\widetilde{x_k} = \widetilde{x_k}^Tx_k$，将上式中的每一个距离展开

$$
||x_k - \widetilde{x_k}||_2^2 = (x_k - \widetilde{x_k})^T(x_k - \widetilde{x_k}) \\
= x_k^Tx_k - x_k^T\widetilde{x_k} - \widetilde{x_k}^Tx_k + \widetilde{x_k}^T\widetilde{x_k} \\
= x_k^Tx_k - 2x_k^T\widetilde{x_k} + \widetilde{x_k}^T\widetilde{x_k}
\quad (9)
$$

其中第一项 $x_k^Tx_k$ 与选取的 w 无关，是个常数。将式（7）代入到（9）的第二项和第三项可得

$$
x_k^T\widetilde{x_k} = x_k^T\sum_{i=1}^d(w_i^Tx_k)w_i
= \sum_{i=1}^d(w_i^Tx_k)x_k^Tw_i
= \sum_{i=1}^dw_i^Tx_kx_k^Tw_i \quad (10)
$$

$$
\hat{x_k}^T\widetilde{x_k} = (\sum_{i=1}^d(w_i^Tx_k)w_i)^T(\sum_{j=1}^d(w_j^Tx_k)w_j) \\
= \sum_{i=1}^d\sum_{j=1}^d((w_i^Tx_k)w_i)^T((w_j^Tx_k)w_j) \quad (11)
$$

注意到，其中 $w_i^Tx_k$ 和 $w_j^Tx_k$ 表示投影长度，都是数字。且当 i ≠ j 时，$w_i^Tw_j = 0$，因此（11）的交叉项中只剩下 d 项。

$$
\widetilde{x_k}^T\widetilde{x_k} = \sum_{i=1}^d\sum_{j=1}^d((w_i^Tx_k)w_i)^T((w_j^Tx_k)w_j) \\
= \sum_{i=1}^d(w_i^Tx_k)(w_i^Tx_k) \\
= \sum_{i=1}^d(w_i^Tx_k)(w_k^Tw_i) \\
= \sum_{i=1}^dw_i^Tx_kx_k^Tw_i. \quad (12)
$$

（12）实际上就是矩阵 $W^Tx_kx_k^TW$ 的迹（对角线元素之和），于是可以将（9）继续化简。

$$
||x_k - \widetilde{x_k}||^2_2 = -\sum_{i=1}^dw_i^Tx_kx_k^Tw_i + x_k^Tx_k \\
= -tr(W^Tx_kx_k^TW) + x_k^Tx_k. \quad (13)
$$

因此（8）可以写成

$$
min_W \sum_{k=1}^n||x_k-\widetilde{x_k}||^2_2 = \sum_{k=1}^n(-tr(W^Tx_kx_k^TW) + x_k^Tx_k) \\
= -\sum_{k=1}^ntr(W^Tx_kx_k^TW) + C
$$

根据矩阵乘法的性质，优化问题可以转化为

$$
\begin{cases}
max_W \ tr(W^TXX^TW), \\
s.t. \quad W^TW = I
\end{cases}
$$

如果我们对 W 中的 d 个基 $w_1, w_2, \cdots, w_d$ 依次求解，就会发现和最大方差理论的方法完全等价。比如当 d = 1 时，实际求解的问题是

$$
\begin{cases}
arg \ max_w w_1^TXX^Tw_1, \\
s.t. \quad w_1^Tw_1 = 1
\end{cases}
$$

最佳直线 W 与最大方差法求解的最佳投影方向一致，即协方差矩阵的最大特征值所对应的特征向量，差别仅是协方差矩阵 $\sum$ 的一个倍数，以及常数 $\sum_{k=1}^nx_k^Tx_k$ 偏差，但这不影响对最大值的优化。

# 推导理解
回到最大方差计算式子

$$
D(x) = w^T(\frac{1}{n}\sum_{i=1}^nx_ix_i^T)w \quad (1)
$$

上式括号内的内容其实就是原始样本的**协方差矩阵**，用 C 表示。接着我们求协方差矩阵的对角化。由于协方差矩阵是实对称矩阵，其主要性质之一就是可以正交对角化，因此就一定可以分解为特征向量和特征值。

$$
PCP^T = P(\frac{1}{n}\sum_{i=1}^nx_ix_i^T)P = \frac{1}{n}\sum_{i=1}^n Px_ix_i^TP =  \Lambda \quad(2)
$$

其中 $\Lambda$ 是一个由特征值构成的对角阵。观察上式，$Px_i$ 表示样本 $x_i$ 投影到坐标系 P 上的新坐标点。PX 表示所有原始样本投影到坐标系 P 上的新坐标，我们假设投影后的样本仅有两个特征 a 和 b。那么 $\frac{1}{n}\sum_{i=1}^n Px_ix_i^TP$ 就是投影后的协方差矩阵即 $\Lambda$，

![PCA 公式 1](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/ml/%E7%89%B9%E5%BE%81%E5%B7%A5%E7%A8%8B/%E9%99%8D%E7%BB%B4/PCA/PCA%E5%85%AC%E5%BC%8F1.png)

【协方差矩阵特点】：
- 协方差矩阵主对角线上的元素为对应特征的方差；
- 协方差矩阵 (i, j) 处的元素为特征 i 和 特征 j 的协方差；
- 协方差矩阵元素关于主对角线对称。

根据前面所讲的内容，优化目标是要最大化投影方向的方差，以及保证投影后的特征之间两两线性无关，即特征之间的协方差为零。经过式子（2）和（3）的推导，我们发现达到优化目标就等价于将原始样本的协方差矩阵对角化——即除对角线外的其它元素化为 0，并且在对角线上将特征值按大小从上到下排列。

投影样本的方差就是原始样本协方差矩阵的特征值。我们要找到最大的方差也就是原始样本协方差矩阵最大的特征值，最佳投影方向就是最大特征值所对应的特征向量。那么问题就转换为求原始样本协方差矩阵的特征值和特征向量。

# 代码实现
首先编写去中心化函数。
```python
def decentration(self, dataset):
    dataset_mean = np.mean(dataset, axis=0)
    return dataset - dataset_mean
```
然后开始编写 PCA 函数的主体。
- 调用去中心化函数，并求去中心化后数据的协方差矩阵。

```python
# 去中心化
dataset_removed = decentration(dataset)
# 求协方差矩阵
xTx = np.dot(dataset_removed.T, dataset_removed)
```
【说明】：也可以使用 `np.cov(dataset_removeda.T)` 方法来求协方差矩阵。
- 求协方差矩阵的特征值与特征向量，并按照特征值大小升序排列。具体用法可参考[传送门](https://docs.scipy.org/doc/numpy/reference/generated/numpy.cov.html)，推荐使用 cov() 函数，毕竟是专业的。

```python
feature_values, feature_vectors = np.linalg.eig(xTx)
feature_values_ind = np.argsort(feature_values)
```
【说明】：关于 eig() 函数可参考[传送门](https://www.numpy.org/devdocs/reference/generated/numpy.linalg.eig.html)，argsort() 函数可参考[传送门](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html)
- 选择其中最大的 d 个特征向量构成转换矩阵。

```python
transform_vectors = feature_vectors[:, feature_values_ind[-self.n_components:]]
```
- 最后，返回转换后的新矩阵。

```python
return np.dot(dataset_removed, self.transform_vectors)
```

【完整代码】：
```python
class PCA:
    
    def __init__(self, n_components=None):
        if n_components and isinstance(n_components, int):
            self.n_components = n_components
        else:
            self.n_components = None
        
    def _decentration(self, dataset):
        dataset_mean = np.mean(dataset, axis=0)
        return dataset - dataset_mean
    
    def fit(self, dataset):
        if self.n_components is None:
            self.n_components = np.min(dataset.shape)
        # 去中心化
        dataset_removed = self._decentration(dataset)
        # 求协方差矩阵
        # xTx = np.dot(dataset_removed.T, dataset_removed)
        xTx = np.cov(dataset_removeda.T)
        # 分解协方差矩阵
        feature_values, feature_vectors = np.linalg.eig(xTx)
        # 对 feature_values 进行升序排序
        feature_values_ind = np.argsort(feature_values)
        # 选择其中最大的 d 个特征向量
        self.transform_vectors = feature_vectors[:, feature_values_ind[-self.n_components:]]
        
    def transform(self, dataset):
        dataset_removed = self._decentration(dataset)
        return np.dot(dataset_removed, self.transform_vectors)
    
    def fit_transform(self, dataset):
        if self.n_components is None:
            self.n_components = np.min(dataset.shape)
        dataset_removed = self._decentration(dataset)
        # xTx = np.dot(dataset_removed.T, dataset_removed)
        xTx = np.cov(dataset_removed.T)
        feature_values, feature_vectors = np.linalg.eig(xTx)
        feature_values_ind = np.argsort(feature_values)
        self.transform_vectors = feature_vectors[:, feature_values_ind[-self.n_components:]]
        return np.dot(dataset_removed, self.transform_vectors)
```

【关于 sklearn PCA 的使用】：
- 官方文档：[传送门](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- sklearn 中 PCA 的使用方法：一个比较简单的例子，可供参考，[传送门](https://www.jianshu.com/p/8642d5ea5389)

## 《机器学习实战》
- 【PCA 实现】：https://github.com/clvsit/Machine-Learning-Note/tree/master/%E9%99%8D%E7%BB%B4/PCA-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E5%AE%9E%E6%88%98
- 【示例：利用 PCA 对半导体制造数据降维】：https://github.com/clvsit/Machine-Learning-Note/tree/master/%E9%99%8D%E7%BB%B4/%E7%A4%BA%E4%BE%8B%EF%BC%9A%E5%88%A9%E7%94%A8PCA%E5%AF%B9%E5%8D%8A%E5%AF%BC%E4%BD%93%E5%88%B6%E9%80%A0%E6%95%B0%E6%8D%AE%E9%99%8D%E7%BB%B4

# 其他问题

### 如何选择合适的 d 值？
PCA 需要手工输入降维后样本的维度，即 d 值，那么该如何指定这个值呢？一般地，有两种方式来加以判断。

【信息比】：定义降维后的信息占比为
$$
\eta = \sqrt{\frac{\sum_{i=1}^d\lambda_i^2}{\sum_{i=1}^n\lambda_i^2}}
$$
其中 $\lambda$ 为原始样本协方差矩阵的特征值（方差），通过对比降维前后特征值的平方累加和，并设定阈值例如 90%，来判断当前 d 值的选择是否合适。若 $\eta < 90\%$，则说明降维后信息损失超过 10%，可认为当前选择的 d 值不合适。

【寻找最优 d 值】：首先将所有可能的 d 值都列举出来，然后获得各 d 值对应的降维数据。寻找一个执行效率高的分类器（视你当前的任务而定，如果是回归，则选择相应的回归模型），将降维数据放入进行训练，选择其中得分最高模型所使用的 d 值。

### 存在的局限性
PCA 是一种线性降维方法，虽然经典，但具有一定的局限性，可能损失有用信息。见下图，在 PCA 中，算法没有考虑数据的标签，只是把原数据映射到一些方差比较大的方向上。

![思维导图](https://i.loli.net/2019/03/27/5c9adb88d54fc.jpg)

此外，降维后的数据将失去原有的物理意义，例如原本的 4 维样本，包括年龄、性别、身高、体重等四个特征，进行降维操作后变成 2 维样本，此时就不存在年龄、性别、身高以及体重。

# 参考
- 《机器学习》周志华
- 《百面机器学习 算法工程师带你去面试》
- 《机器学习实战》
- 如何通俗易懂地解释「协方差」与「相关系数」的概念？：https://www.zhihu.com/question/20852004
- 矩阵对角化与奇异值分解：https://zhuanlan.zhihu.com/p/34281291
- sklearn中PCA的使用方法：https://www.jianshu.com/p/8642d5ea5389