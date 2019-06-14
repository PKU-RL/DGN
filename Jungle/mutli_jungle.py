import os, sys, time
import numpy as np
import magent
from magent.model import BaseModel
from magent.builtin.tf_model import DeepQNetwork
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
import random
from ReplayBuffer_v2 import ReplayBuffer
from keras.layers import Dense, Dropout, Conv2D, Input, Lambda, Flatten, TimeDistributed, merge
from keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector
from keras.models import Model
from keras.layers.core import Activation
from keras.utils import np_utils,to_categorical
from keras.engine.topology import Layer
from keras.callbacks import TensorBoard
from magent.builtin.tf_model import DeepQNetwork
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   
session = tf.Session(config=config)
KTF.set_session(session)

np.random.seed(16)

def Adjacency(state):
    adj = []
    dis = []
    for j in range(20):
        dis.append([state[j][-2],state[j][-1],j])
    for j in range(20):
        f = []
        for r in range(len(dis)):
            f.append([(dis[r][0]-dis[j][0])**2+(dis[r][1]-dis[j][1])**2,r])
        f.sort(key=lambda x:x[0])
        y = []
        for r in range(4):
            y.append(f[r][1])
        y = to_categorical(y,num_classes=20)
        adj.append(y)
    return adj

def observation(state1,state2):
    state = []
    for j in range(20):
        state.append(np.hstack(((state1[j][0:11,0:11,1]-state1[j][0:11,0:11,4]).flatten(),state2[j][-1:-3:-1])))
    return state

def MLP():
    In_0 = Input(shape=[123])

    h = Dense(512, activation='relu',kernel_initializer='random_normal')(In_0)
    h = Dense(128, activation='relu',kernel_initializer='random_normal')(h)

    h = Reshape((1,128))(h)

    model = Model(input=In_0,output=h)
    return model

def MultiHeadsAttModel(l=2, d=128, dv=16, dout=128, nv = 8 ):

    v1 = Input(shape = (l, d))
    q1 = Input(shape = (l, d))
    k1 = Input(shape = (l, d))
    ve = Input(shape = (1, l))

    v2 = Dense(dv*nv, activation = "relu",kernel_initializer='random_normal')(v1)
    q2 = Dense(dv*nv, activation = "relu",kernel_initializer='random_normal')(q1)
    k2 = Dense(dv*nv, activation = "relu",kernel_initializer='random_normal')(k1)

    v = Reshape((l, nv, dv))(v2)
    q = Reshape((l, nv, dv))(q2)
    k = Reshape((l, nv, dv))(k2)
    v = Lambda(lambda x: K.permute_dimensions(x, (0,2,1,3)))(v)
    k = Lambda(lambda x: K.permute_dimensions(x, (0,2,1,3)))(k)
    q = Lambda(lambda x: K.permute_dimensions(x, (0,2,1,3)))(q)

    att = Lambda(lambda x: K.batch_dot(x[0],x[1] ,axes=[3,3]) / np.sqrt(dv))([q,k])# l, nv, nv
    att = Lambda(lambda x: K.softmax(x))(att)
    out = Lambda(lambda x: K.batch_dot(x[0], x[1],axes=[3,2]))([att, v])
    out = Lambda(lambda x: K.permute_dimensions(x, (0,2,1,3)))(out)

    out = Reshape((l, dv*nv))(out)

    T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([ve,out])

    out = Dense(dout, activation = "relu",kernel_initializer='random_normal')(T)
    model = Model(inputs=[q1,k1,v1,ve], outputs=out)
    return model

def Q_Net(action_dim):
    I1 = Input(shape = (1, 128))
    I2 = Input(shape = (1, 128))
    I3 = Input(shape = (1, 128))

    h1 = Flatten()(I1)
    h2 = Flatten()(I2)
    h3 = Flatten()(I3)

    h = Concatenate()([h1,h2,h3])
    V = Dense(action_dim,kernel_initializer='random_normal')(h)

    model = Model(input=[I1,I2,I3],output=V)
    return model

path="data/battle_model"
map_size=100
capacity = 200000
batch_size = 256
totalTime = 0
TAU = 0.01     
LRA = 0.0001        
param = None
alpha = 0.6
GAMMA = 0.96
n_episode = 100000
max_steps = 120
episode_before_train = 200
n_agent=20
magent.utility.init_logger("eat")
env = magent.GridWorld("eat", map_size=30)
env.set_render_dir("build/render")
handles = env.get_handles()
sess = tf.Session()
K.set_session(sess)
n = len(handles)
n_actions=env.get_action_space(handles[0])[0]
i_episode=0
buff=ReplayBuffer(capacity)
l=40

print(env.get_action_space(handles[0])[0])
print(env.get_action_space(handles[1])[0])
f = open('log_eat_re.txt','w')

######build the model#########
cnn = MLP()
m1 = MultiHeadsAttModel(l=4)
m2 = MultiHeadsAttModel(l=4)
q_net = Q_Net(action_dim = 9)
vec = np.zeros((1,4))
vec[0][0] = 1

In= []
for j in range(n_agent):
    In.append(Input(shape=[123]))
    In.append(Input(shape=(4,20)))
In.append(Input(shape=(1,4)))
feature = []
for j in range(n_agent):
    feature.append(cnn(In[j*2]))

feature_ = Concatenate(axis=1)(feature)

relation1 = []
for j in range(n_agent):
    T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([In[j*2+1],feature_])
    relation1.append(m1([T,T,T,In[40]]))

relation1_ = Concatenate(axis=1)(relation1)

relation2 = []
for j in range(n_agent):
    T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([In[j*2+1],relation1_])
    relation2.append(m2([T,T,T,In[40]]))

V = []
for j in range(n_agent):
    V.append(q_net([feature[j],relation1[j],relation2[j]]))

model = Model(input=In,output=V)
model.compile(optimizer=Adam(lr = 0.0001), loss='mse')
model.summary()

######build the target model#########
cnn_t = MLP()
m1_t = MultiHeadsAttModel(l=4)
m2_t = MultiHeadsAttModel(l=4)
q_net_t = Q_Net(action_dim = 9)
In_t= []
for j in range(n_agent):
    In_t.append(Input(shape=[123]))
    In_t.append(Input(shape=(4,20)))
In_t.append(Input(shape=(1,4)))

feature_t = []
for j in range(n_agent):
    feature_t.append(cnn_t(In_t[j*2]))

feature_t_ = Concatenate(axis=1)(feature_t)

relation1_t = []
for j in range(n_agent):
    T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([In_t[j*2+1],feature_t_])
    relation1_t.append(m1_t([T,T,T,In_t[40]]))

relation1_t_ = Concatenate(axis=1)(relation1_t)

relation2_t = []
for j in range(n_agent):
    T = Lambda(lambda x: K.batch_dot(x[0],x[1]))([In_t[j*2+1],relation1_t_])
    relation2_t.append(m2_t([T,T,T,In_t[40]]))

V_t = []
for j in range(n_agent):
    V_t.append(q_net_t([feature_t[j],relation1_t[j],relation2_t[j]]))

model_t = Model(input=In_t,output=V_t)

###########playing#############
while i_episode<n_episode:
    alpha*=0.996
    if alpha<0.01:
        alpha=0.01
    print(i_episode)
    i_episode=i_episode+1
    env.reset()
    #env.add_walls(method="random", n=map_size * map_size * 0.03)
    env.add_agents(handles[0], method="random", n=20)
    env.add_agents(handles[1], method="random", n=12)
    step_ct = 0
    done = False
    n = len(handles)
    obs  = [[] for _ in range(n)]
    ids  = [[] for _ in range(n)]
    action = [[] for _ in range(n)]
    nums = [env.get_num(handle) for handle in handles]
    steps = 0
    score = 0
    loss = 0
    dead = [0,0]
    
    while steps<max_steps:
        steps+=1
        i=0
        obs[i] = env.get_observation(handles[i])
        adj = Adjacency(obs[i][1])
        flat_ob = observation(obs[i][0],obs[i][1])
        ob=[]
        for j in range(n_agent):
            ob.append(np.asarray([flat_ob[j]]))
            ob.append(np.asarray([adj[j]]))
        ob.append(np.asarray([vec]))
        acts = model.predict(ob)
        action[i]=np.zeros(n_agent,dtype = np.int32)
        for j in range(n_agent):
            if np.random.rand()<alpha:
                action[i][j]=random.randrange(n_actions)
            else:
                action[i][j]=np.argmax(acts[j])
        env.set_action(handles[i], action[i])
        done = env.step()
        
        next_obs = env.get_observation(handles[0])
        flat_next_obs = observation(next_obs[0],next_obs[1])
        rewards = env.get_reward(handles[0])
        score += sum(rewards)
        if steps%3 ==0:
            buff.add(flat_ob, action[0], flat_next_obs, rewards, done, adj)

        if (i_episode-1) % 10 ==0:
            env.render()
        if max_steps == steps:
            damage = 0
            for j_ in range(n_agent):
                damage = damage + 400- obs[0][0][j_][5][5][2]*400
            damage = damage/20
            print(damage,end='\t')
            print(score/300,end='\t')
            #f.write(str(dead[i])+'\t'+str(score[i]/300)+'\t')
            #f.write(str(loss/100)+'\n')
            print(loss/100,end='\n')
            f.write(str(damage)+'\t'+str(score/300)+'\t'+str(loss/100)+'\n')
        env.clear_dead()

        if i_episode < episode_before_train:
            continue
        if steps%3 != 0:
            continue
        #############training###########
        batch = buff.getBatch(10)
        states,actions,rewards,new_states,dones,adj=[],[],[],[],[],[]
        for i_ in  range(n_agent*2+1):
            states.append([])
            new_states.append([])
        for e in batch:
            for j in range(n_agent):
                states[j*2].append(e[0][j])
                states[j*2+1].append(e[5][j])
                new_states[j*2].append(e[2][j])
                new_states[j*2+1].append(e[5][j])
            states[40].append(vec)
            new_states[40].append(vec)
            actions.append(e[1])
            rewards.append(e[3])
            dones.append(e[4])
        
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        dones = np.asarray(dones)
        
        for i_ in  range(n_agent*2+1):
            states[i_]=np.asarray(states[i_])
            new_states[i_]=np.asarray(new_states[i_])

        q_values = model.predict(states)
        target_q_values = model_t.predict(new_states)

        for k in range(len(batch)):
            if dones[k]:
                for j in range(n_agent):
                    q_values[j][k][actions[k][j]] = rewards[k][j]
            else:
                for j in range(n_agent):
                    q_values[j][k][actions[k][j]] =rewards[k][j] + GAMMA*np.max(target_q_values[j][k])

        history=model.fit(states, q_values,epochs=1,batch_size=10,verbose=0)
        his=0
        for (k,v) in history.history.items():
            his+=v[0]
        loss+=(his/20)
        #########train target model#########
        weights = cnn.get_weights()
        target_weights = cnn_t.get_weights()
        for w in range(len(weights)):
            target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]
        cnn_t.set_weights(target_weights)

        weights = q_net.get_weights()
        target_weights = q_net_t.get_weights()
        for w in range(len(weights)):
            target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]
        q_net_t.set_weights(target_weights)

        weights = m1.get_weights()
        target_weights = m1_t.get_weights()
        for w in range(len(weights)):
            target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]
        m1_t.set_weights(target_weights)

        weights = m2.get_weights()
        target_weights = m2_t.get_weights()
        for w in range(len(weights)):
            target_weights[w] = TAU * weights[w] + (1 - TAU)* target_weights[w]
        m2_t.set_weights(target_weights)

        #######save model###############
    model.save('gdn.h5')


