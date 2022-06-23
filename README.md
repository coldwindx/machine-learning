# machine-learning
吴恩达机器学习系列课程
# 基础
- 监督学习：数据集的每个样本均包含正确答案（回归|分类）
# 模型描述
```
                Training Set
                    |
                    v
            Learning Algorithm
                    |
                    v
           x --->  h(x)  ---> y
```

## 数据集
- 留出法与交叉验证法适用于数据量充足的场景，但效率偏低
- 自助法适用于数据集小，难以划分的场景，但会改变数据集分布，产生估计偏差。
## 评估
- 均方误差函数：$J=\frac{1}{2m}\sum_{i=1}^m(h(x) - y)^2$，最常见的代价函数，通常能解决大部分问题
- 错误率：分类错误的样本数占总样本数的比例为错误率，精度=1-错误率

## 梯度下降
- 重复步骤$\theta := \theta - \alpha\frac{\partial{J}}{\partial{\theta}}$，其中$\alpha$为学习率，反应沿梯度下降的步幅。
- 学习率过大可能会导致代价函数不减反增的现象
$\frac{\partial{J}}{\partial{\theta}}=\frac{1}{m}\sum_{i=1}^m(h(x) - y)x$
- 特征缩放：使不同特征值取值在相近的范围内，从而使梯度下降法能够更快收敛。
- mean normalization：一种特征缩放的简单方法，令$x=\frac{x-\mu}{s}$
## 正规方程法
- 对于某些线性回归问题，正规方程法是更好的解决方案。正规方程法可以不使用特征缩放。
- 正规方程法实际上是直接令$\frac{\partial{J}}{\partial{\theta}}=0$求出最优解$\theta=(X^TX)^{-1}XY$
- 正规方程法不需要学习率，不需要迭代，但特征量n过大时矩阵的计算成本极高$O(n^3)$，当n>10000时可以考虑其他方法
- $(X^TX)$矩阵不可逆：存在多余特征或特征数量大于训练集数量
# 分类问题
## 逻辑回归
- 逻辑回归假设函数：$h_{\theta}(x)=g(\theta^Tx)$
- 使用以上假设函数会导致代价函数$J(\theta)$为非凹函数，影响梯度下降，通常将代价函数定义为$$J(\theta)=\frac{1}{m}\sum_{i=1}^{m}Cost(h(x), y),\\其中
Cost=\left\{
        \begin{matrix}
        -log(h(x)),&if&y=1\\
        -log(1-h(x)),&if&y=0
        \end{matrix}
        \right.$$
- 简化的Cost函数$Cost=-y\log{h(x)}-(1-y)\log(1-h(x))$
### 逻辑函数
- sigmoid函数：$g(z)=\frac{1}{1+e^{-z}}$
# 过拟合
- 欠拟合：高偏差，曲线不能很好地贴合样本
- 过拟合：高方差，曲线为高阶多项式，不够平滑。
- 过拟合：学习能力太强，将训练样本中的非一般特征也学到，导致泛化性下降。过拟合不可消除。
## 正则化
- 引入正则化后代价函数：$J(\theta)= J(\theta) + \frac{\lambda}{2m}\sum_{j=1}^{n}\theta^2$，通过正则化缩小某些特征值地系数
- $\lambda$为正则化参数，过大时会使$\theta$趋于0，导致欠拟合
- [正则化为什么可以缓解过拟合？](https://zhuanlan.zhihu.com/p/361181741)
## 多分类问题
- 为每一类创建一个分类器，进行多次逻辑回归训练
- 向量化可以避免循环，使训练更高效
- 引入向量化后，$\frac{\partial{J}}{\partial{\theta}}=\frac{1}{m}X^T(h(x)-y)$
- 向量化后的梯度更新公式:$\theta_j=\theta_j-\alpha[\frac{1}{m}X^T(h(x)-y)+\frac{\lambda}{m}\theta_j]$