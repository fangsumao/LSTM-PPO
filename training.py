import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import drone_sim2d
import numpy as np
from datetime import datetime
import collections

device = "cpu"

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
    
    def clear_memory(self):
        del self.actions[:]   #取数组中从0到0的元素，即清除
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

class ActorCritic(nn.Module):  #所有模型的父类，自定义模型通过重写该模型来实现
    def __init__(self, state_dim, action_dim, n_var, action_std):      #state = 2+2+1+240 = 245
        super(ActorCritic, self).__init__()        
        ######### Actor Layers #########
        self.lstm1_a = nn.LSTM(input_size=3,hidden_size=63)
        self.fc2_a = nn.Linear(64,64)  #神经网络线性层 y=xA^T+b A为权重
        self.fc3_a = nn.Linear(64,2)
        self.h0_a = torch.zeros(1,1,63)   #生成1*1*63且使用0填充的tensor
        self.c0_a = torch.zeros(1,1,63)                    
                

        ######### Critic Layers #########
        self.lstm1_c = nn.LSTM(input_size=3,hidden_size=63)
        self.fc2_c = nn.Linear(64,64)
        self.fc3_c = nn.Linear(64,1)
        self.h0_c = torch.zeros(1,1,63)
        self.c0_c = torch.zeros(1,1,63)
        

        #self.critic = nn.Sequential(critic_net)
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)  #创建长度为action_dim的一维张量，使用action_std*action_std填充
        
    ###add actor definition
    def actor(self,x,y):
        x = torch.FloatTensor(np.array(x)).view(-1,1,3)
        #numpy.array 创建numpy数组
        #FloatTensor 转化为Tensor
        #torch.Tensor.view 转化为view(d1,d2,...)格式的tensor，-1表示该维度从其他维度推断
        x, _ = self.lstm1_a(x,(self.h0_a,self.c0_a))
        x = x[-1,:,:]  #选取第一个轴的最后一行的所有元素
        x = x.view(-1)
        x = torch.tanh(x)
           
        z = torch.cat((x,torch.FloatTensor(np.array([y]))),dim=-1)  #x与y的直接拼接
        z = self.fc2_a(z)  #64->64线性 
        z = torch.tanh(z)
        z = self.fc3_a(z)  #64->2线性
        z = torch.tanh(z)
        return z

    def critic(self,x,y):
        x = torch.FloatTensor(np.array(x)).view(-1,1,3)
        x, _ = self.lstm1_c(x,(self.h0_c,self.c0_c))
        x = x[-1,:,:]
        x = x.view(-1)
        x = torch.tanh(x)
           
        z = torch.cat((x,torch.FloatTensor(np.array([y]))),dim=-1)
        z = self.fc2_c(z)
        z = torch.tanh(z)
        z = self.fc3_c(z)
        z = torch.tanh(z)
        return z

    def forward(self):
        print("t")
        raise NotImplementedError #说明该方法需要在子类中重现和实现，否则抛出NotImplementedError异常
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        dist = MultivariateNormal(action_mean, torch.diag(self.action_var).to(device))
        #MultivariateNormal（tensor x, tensor y） 创建多元分布采样的高斯分布，返回一个类 x为均值，y为协方差矩阵
        #torch.diag 将一维张量转化为对角线方阵（或取方阵的对角线）
        action = dist.sample()  #抽取样本
        action_logprob = dist.log_prob(action)  #对数概率密度
        
        memory.states.append(state)  #将state追加到数组的末尾
        memory.actions.append(action)
        memory.logprobs.append(action_logprob) 
        
        return action.detach()  #返回action张量的一个视图且返回值不再参与自动微分
    
    def evaluate(self, state, action):
        action_mean = []
        state_value = []
        no_vehicles = len(state)  #返回state的包含的元素数
        for idx_1 in range(no_vehicles):
            action_mean.append(self.actor(state[idx_1][1],state[idx_1][0]))
            state_value.append(self.critic(state[idx_1][1],state[idx_1][0]))
        action_mean = torch.stack(action_mean).view(-1,1,2)  #torch.stack 将张量序列沿着默认维度（dim==0）堆叠
        state_value = torch.stack(state_value).view(-1,1,1)
        
        #action_mean = self.actor(state)
        dist = MultivariateNormal(torch.squeeze(action_mean), torch.diag(self.action_var))  #torch.squeeze 移除张量中大小为1的维度
        
        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()  #计算概率分布的熵，返回tensor包含给定概率分布的每个元素的熵
        #state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy 
        #action_logprobs: 衡量模型输出动作的对数概率，通常希望最大化动作对数概率，可以增加模型选择正确动作的可能
        #state_value:当前状态的预期回报，
        #dist_entrop:输出动作的概率分布的熵，即不确定性度量，最大化概率分布的熵，可以增加模型的探索性

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(device)
        #filename = "PPO_Continuous_drones2d-v0_288000_1562915157"
        #self.policy.load_state_dict(torch.load("./models/"+filename+".pth"))
        self.optimizer = torch.optim.Adam(self.policy.parameters(),lr=lr, betas=betas)  
        #lr:learning rate
        #troch.Module.parameters 返回一个iterator[parameter] iterator:迭代器 parameter:模型参数 
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var, action_std).to(device)
        #self.policy_old.load_state_dict(torch.load("./models/"+filename+".pth"))
        self.MseLoss = nn.MSELoss()  #返回L2范数的平方
    
    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
        #tensor.data.numpy 将tensor转换位numpy数组 numpy.flatten 将一个numpy数组展开成一个维度
        #tensor.cpu 将tensor移回gpu，通常在转化为numpy前要进行
    
    def update(self, memory):
        # Monte Carlo estimate of rewards: 蒙特卡罗估计，取完成整个事件的完整过程的回报均值作为回报估计
        rewards = []
        discounted_reward = 0
        for reward in reversed(memory.rewards):  #reversed(Iterator) 反向迭代器
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)  #将元素插入括号前的位置
        
        # Normalizing the rewards: 标准化奖励 (reward-Mean rewards)/standard deviation of rewards 
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        #tensor.mean() 张量的均值
        #tensor.std() 张量的标准差，默认无偏估计（n-1）
        
        # convert list to tensor
        old_states = memory.states
        old_actions = torch.stack(memory.actions).to(device).detach()
        #torch.stack() 将一系列新张量在新轴上堆叠
        #tensor.detach() 分离张量使其不参与后续梯度运算
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()
        
        # Optimize policy for K epochs: 最优策略（在K次迭代下生成）
        for _ in range(self.K_epochs):
            # Evaluating old actions and values : 评估旧动作和价值
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
     
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss: PPO模型的损失函数计算
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages #torch.clamp(tensor, x1, x2) 使得tensor中最小值为x1(替代)，最大值为x2(替代)
            loss = -torch.min(surr1.float(), surr2.float()) + 0.5*self.MseLoss(state_values, rewards.float()) - 0.01*dist_entropy
            
            # take gradient step 梯度下降
            self.optimizer.zero_grad() #梯度清零，用于每个训练迭代的开始
            loss.mean().backward() #计算反向传播
            self.optimizer.step() #一步梯度下降更新
            
        # Copy new weights into old policy: 更新权重
        self.policy_old.load_state_dict(self.policy.state_dict())

class AgentHandler:
    def __init__(self):
        self.agentmemory = AgentMemory()

    def state_handler(self,state): #将张量中不符合要求的维度转化为符合张量的维度
        no_vehicles = len(state)
        new_states = []
        for idx_1 in range(no_vehicles):
            new_states = state
            if len(state[idx_1][1]) == 0:
                new_states[idx_1] = (state[idx_1][0],[np.array([0,0,0])])
        return new_states

    def del_done(self,response):

        no_vehicles = len(response)
        del_list = []
        for idx_1 in range(no_vehicles):
            if response[idx_1][2]==True:
                del_list.append(idx_1)
        del_list.sort(reverse=True)
        for idx_2 in range(len(del_list)):
            del response[del_list[idx_2]]
        return response

    def select_action(self,state,actor):
        self.action_var = torch.full((2,), 0.6*0.6).to(device)  
        #manually change action_dim action_std 
        #设置动作方差为0.6*0.6,较大的动作方差意味着动作变化的范围更广。动作分布是描述代理在环境中选择动作的概率分布
        no_vehicles = len(state)
        action_list = []
        for idx_1 in range(no_vehicles):
            state1 = state[idx_1]
            target1 = state1[0]
            other_vehicles1 = state1[1] 
            action_mean = actor(other_vehicles1,target1)
            dist = MultivariateNormal(action_mean, torch.diag(self.action_var).to(device))      
            ##ATTENTION: manually change variance in torch.diag(var) 
            #创建多变量正态分布对象，action_mean为动作均值,torch.diag(self.action_var)为指定的动作方差
            action = dist.sample() #从正态分布中抽取一个动作
            action_logprob = dist.log_prob(action) #计算当前动作的对数概率
            self.agentmemory.memory_list[idx_1].states.append(state1) #添加状态
            self.agentmemory.memory_list[idx_1].actions.append(action) #添加动作
            self.agentmemory.memory_list[idx_1].logprobs.append(action_logprob) #添加动作的对数概率

            action_list.append(action.detach().cpu().data.numpy().flatten()) #动作添加到动作列表
            
        return action_list   

    def response_eval(self,response): #将respone的各部分存储到对应列表中
        no_vehicles = len(response)
        state_list = []
        reward_list = []
        done_list = []
        for idx_1 in range(no_vehicles):            
            state = response[idx_1][0]   #only works one additional vehicle
            if len(state[1]) == 0:
                state = (state[0],[np.array([0,0,0])])
            reward = response[idx_1][1]
            done = response[idx_1][2] 

            state_list.append(state)
            reward_list.append(reward)
            done_list.append(done)           

        return state_list,reward_list,done_list

    def reward_memory_append(self,reward): #添加奖励值
        no_vehicles = len(reward)
        for idx_1 in range(no_vehicles):
            self.agentmemory.memory_list[idx_1].rewards.append(reward[idx_1])
    

       


class AgentMemory:
    def __init__(self):
        self.memory_list = []
    def create_memory(self,amount):
        self.memory_list = []
        for _ in range(amount):
            self.memory_list.append(Memory())
    
    def delete_memory(self):
        del self.memory_list
    




def main():
    ############## Hyperparameters ##############
    env_name = "drones2d-v0"
    render = False                  # rendering mode, needs to be set to False. 渲染模式
    solved_reward = 100000          # stop training if avg_reward > solved_reward 
    log_interval = 10               # print avg reward in the interval 每隔interval打印avg reward
    save_interval = 1000            # Interval model is saved 每隔interval存储模型,original 20
    max_episodes = 100000           # max training episodes 最大训练步数
    max_timesteps = 100             # max timesteps in one episode 单步内最大时间步
    n_latent_var = [128,128,64]     # list of neurons in hidden layers 神经元隐藏层
    update_timestep = 4000          # update policy every n timesteps 每n个时间步更细策略
    action_std = 0.6                # constant std for action distribution 
    lr = 0.0001                     # learning rate 学习率
    betas = (0.9, 0.999)            # betas 模型参数
    gamma = 0.99                    # discount factor 折扣因子
    K_epochs = 2                    # update policy for K epochs 完整迭代次数K
    eps_clip = 0.2                  # clip parameter for PPO 超参数，用于在优化步骤中限制策略更新的幅度，以确保模型的稳定性
    random_seed = None              # random seed 随机种子
    xrange_init = [-30,30]          # initial x-coordinate-range of vehicles 初始车辆的坐标范围x轴
    yrange_init = [-30,30]          # initial y-coordinate-range of vehicles 初始车辆的坐标范围y轴
    xrange_target = [-30,30]        # target x-coordinate-range of vehicles 目标车辆的坐标范围x轴
    yrange_target = [-30,30]        # target y-coordinate-range of vehicles 目标车辆的坐标范围y轴
    agents = 5                      # max no of agents in the simulation 模拟中最大代理的数量
    #############################################
    
    # creating environment
    env = gym.make(env_name)
    state_dim = 7
    action_dim = 2
    time_stamp = str(int(datetime.timestamp(datetime.now()))) #时间戳
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        torch.manual_seed(random_seed) #设置随机数生成种器的种子
        env.seed(random_seed) #设置环境的种子
        np.random.seed(random_seed) #设置numpy的随机种子
    
    memory = Memory()
    handler = AgentHandler()
    ppo = PPO(state_dim, action_dim, n_latent_var, action_std, lr, betas, gamma, K_epochs, eps_clip) #PPO模型
    print(lr,betas)
    file_string = "./logs/log_"+time_stamp
    f = open(file_string+"_parameters.txt","a+")
    f.write('Env-Name:\t\t{}\nn_latent_var:\t\t{}\nupdate_timestep:\t{}\naction_std:\t\t{}\nlr:\t\t\t{}\nbetas:\t\t\t{}\ngamma:\t\t\t{}\nK_epochs:\t\t{}\neps_clip:\t\t{}\nxrange_init:\t\t{}\nyrange_init:\t\t{}\nxrange_target:\t\t{}\nyrange_target:\t\t{}\n'.format(env_name,n_latent_var,update_timestep,action_std,lr,betas,gamma,K_epochs,eps_clip,xrange_init,yrange_init,xrange_target,yrange_target)) 
    f.close()

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    
    # training loop
    for i_episode in range(1, max_episodes+1):
        agents = np.random.randint(low=1,high=5) #生成随机整数，范围[low, high)
        state = env.reset(amount = agents,xrange_init=xrange_init,yrange_init=yrange_init,xrange_target=xrange_target,yrange_target=yrange_target,eps_arr=1)
        #初始化环境状态,并保证每次初始化种子相同
        state = handler.state_handler(state)
        handler.agentmemory.delete_memory()     #delete old memory
        handler.agentmemory.create_memory(len(state))
        ##### Version with omitted position state #####
        for t in range(max_timesteps):
            time_step +=1
            # Running policy_old:
            action = handler.select_action(state,ppo.policy_old.actor)       #for use with multi-agent environment
            response = env.step(action) #返回奖励
            # response = [response[0],response[1],response[2],response[4]]
            # next_state, reward, done, info =env.step(action) 其中,done表示当前回合是否结束,info包含一些额外的环境信息
            state,reward,done = handler.response_eval(response)

            # Saving reward:
            handler.reward_memory_append(reward)

            #Procedure to write the memory of done agents to the complete memory
            no_vehicles = len(done)
            del_list = []
            for idx_3 in range(no_vehicles):
                if done[idx_3]==True:
                    memory.states.extend(handler.agentmemory.memory_list[idx_3].states)
                    memory.actions.extend(handler.agentmemory.memory_list[idx_3].actions)
                    memory.rewards.extend(handler.agentmemory.memory_list[idx_3].rewards)
                    memory.logprobs.extend(handler.agentmemory.memory_list[idx_3].logprobs)
                    del_list.append(idx_3)
            
            del_list.sort(reverse=True) #降序排列
            #delete done memory from agentmemory
            for idx_3b in range(len(del_list)):
                del handler.agentmemory.memory_list[del_list[idx_3b]]

            #if max_timesteps is reached copy agent's memory to central memory
            
            if t == max_timesteps-1:
                no_vehicles = len(handler.agentmemory.memory_list)

                for idx_4 in range(no_vehicles):
                    memory.states.extend(handler.agentmemory.memory_list[idx_4].states)
                    memory.actions.extend(handler.agentmemory.memory_list[idx_4].actions)
                    memory.rewards.extend(handler.agentmemory.memory_list[idx_4].rewards)
                    memory.logprobs.extend(handler.agentmemory.memory_list[idx_4].logprobs)

            running_reward += np.sum(reward)  #change to something more appliccaple
            if render:
                env.render()
            if all(elem == True for elem in done):    
                break
            response = handler.del_done(response)
            state,reward,done = handler.response_eval(response)
        
        #update after every n episode
        if i_episode % 10 == 0:
            ppo.update(memory)
            memory.clear_memory()
            time_step = 0

        avg_length += t
        
        #save the model at the last step
        if i_episode % save_interval == 0:
            print("Model saved!")
            torch.save(ppo.policy.state_dict(), './models/PPO_Continuous_{}_{}_{}.pth'.format(env_name,i_episode,time_stamp))
            #break

        #Log Avg_length and Avg_reward at every log_interval steps and write to file
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)+1
            running_reward = int((running_reward/log_interval/agents))
            
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            f = open(file_string+"_data.txt","a+")
            f.write('{}\t{}\t{}\n'.format(i_episode,avg_length,running_reward))
            f.close()
            running_reward = 0
            avg_length = 0
            
if __name__ == '__main__':
    main()
    