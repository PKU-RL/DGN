# GraphConv4MARL

**The codes are the implementations of DGN in the three scenarios (Jungle, Battle, and Routing) presented in the paper
[Graph Convolution Reinforcement Learning for Multi-Agent Cooperation](https://arxiv.org/abs/1810.09202)**

![DGN](https://github.com/PKU-AI-Edge/DGN-Paper/blob/master/figures/arch.pdf "DGN")


**DGN** functionally consists of three components: graph convolution, relation kernel, and temporal regularization. 


*The paper [Learning Transferable Cooperative Behavior in Multi-Agent Teams](https://arxiv.org/pdf/1906.01202.pdf) mentions that our DGN is limited by the fixed number of agent neigbhors. This is not true. DGN does not have this limitation. The fixed number came from our implementations. Our current implementations are based on TensorFlow, however, tensorFlow does not support dynamic computational graph. So, in these implementations, we fix the number of neighbors for each agent. Indeed, DGN is fully dynamic, no matter how many neighbors each agent has at a timestep and no matter how the graph of agents changes (disconnected or fully connected). We will appriciate it if anyone can implement DGN using PyTorch. Please let us know if you need any help.*


Please cite our paper if you are using the codes.

Jiechuan Jiang, Chen Dun, and Zongqing Lu. Graph convolutional reinforcement learning for multi-agent cooperation. arXiv preprint arXiv:1810.09202, 2018.
