## 概述
Batch Normalization（BN）和 Layer Normalization（LN）是深度学习中两种常用的归一化技术，旨在缓解内部协变量偏移（Internal Covariate Shift）问题，加速训练并提升模型性能。它们的核心区别在于​​归一化的维度不同​​，因此适用于不同的场景。
## 核心区别
1. **归一化维度不同**：
   - **Batch Normalization (BN)**：在每个批次内对数据进行归一化，适用于深度神经网络中的全连接层和卷积层。
   - **Layer Normalization (LN)**：在每个样本内部对数据进行归一化，适用于循环神经网络（RNN）和Transformer等序列模型。
2. **适用场景不同**：
   - **Batch Normalization (BN)**：适用于深度神经网络，尤其是在训练过程中数据分布发生变化时。
   - **Layer Normalization (LN)**：适用于循环神经网络和Transformer等序列模型，尤其是在处理长序列数据时。