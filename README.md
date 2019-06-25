# DGN

**The codes are the implementations of DGN in the three scenarios, i.e., Jungle, Battle and Routing, presented in the paper
[Graph Convolution Reinforcement Learning for Multi-Agent Cooperation](https://arxiv.org/abs/1810.09202)**

<img src="arch.png" alt="DGN" width="500">

DGN works as follows. First, each agent encodes the observation and sends the features to its neighboring agents. Then, each agent integrates the received features using relation kernel and again sends it to its neighboring agents. The process continues as more CONV layers are added. Finally, the features of all the preceding layers are concatenated and fed into Q network. 

All agents share weights for the modules. The main reason is that agents use relation kernels to extract their relations based the encodings of their observations. If the encoders are different (agents encodes the observation in different ways), the relation kernels can hardly learn to extract their relations since the graph of agents is highly dynamic. 

Another very important benefit comes from parameter-sharing among agents is **DGN can naturally avoid non-stationarity.** From the optimization point of view, DGN optimizes a set of parameters for N objectives, one objective for each agent. As illustrated in the figure above, DGN as a whole can be seen as taking all the observations as input and outputting actions for all the agents, and thus DGN implicitly avoids the non-stationarity. 

**DGN** is simply and effcient. It emprically outperforms many state-of-art algorithms. DGN is applicable to many real applications. DGN has bee applied to traffic signal control by researchers ([CoLight: Learning Network-level Cooperation for Traffic Signal Control](https://arxiv.org/abs/1905.05717)). 

*The paper ([Learning Transferable Cooperative Behavior in Multi-Agent Teams](https://arxiv.org/pdf/1906.01202.pdf)) mentions that our DGN is limited by the fixed number of agent neigbhors. This is not true. DGN does not have this limitation. The fixed number came from our implementations. Our current implementations are based on TensorFlow, however, tensorFlow does not support dynamic computational graph. So, we fix the number of neighbors for each agent in these implementations. Indeed, DGN adapts to fully dynamic environments, no matter how many neighbors each agent has at a timestep and no matter how the graph of agents changes (disconnected or fully connected). We will appriciate it if anyone can implement DGN using PyTorch. Please let us know if you need any help.*




Please cite our paper if you are using the codes.

[Jiechuan Jiang, Chen Dun, and Zongqing Lu. *Graph convolutional reinforcement learning for multi-agent cooperation*. arXiv preprint arXiv:1810.09202, 2018.](https://arxiv.org/abs/1810.09202)
