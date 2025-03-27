import heapq
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Global parameters setup
NUM_ROUNDS = 200
NUM_EDGES = 10000
NUM_CLOUDS = 1
NUM_FOGS = 100

FOG_CAPACITY = 10
CLOUD_CAPACITY = 50
FOG_COMPUTE_POWER = 20.0
CLOUD_COMPUTE_POWER = 100.0

EXTENDED_LAYERS = 8
EXTENDED_LAYERS_COMPUTE = [1.5, 2.5, 2.0, 2.0, 1.0, 1.0, 1.5, 0.5]
EXTENDED_LAYERS_DATA    = [1.2, 1.8, 1.0, 1.0, 0.8, 0.5, 0.5, 0.3]

EDGE_POWER_MIN = 0.5
EDGE_POWER_MAX = 1.5

EDGE_FOG_BW_RANGE = (3.0, 10.0)
FOG_CLOUD_BW_RANGE = (5.0, 15.0)
FOG_EDGE_BW_RANGE = (3.0, 10.0)
CLOUD_EDGE_BW_RANGE = (5.0, 15.0)

DEBUG = True
def debug_print(*args):
    if DEBUG:
        print(*args)

# PPO related
RL_LR = 1e-4
RL_BETAS = (0.9, 0.999)
RL_GAMMA = 0.99
K_EPOCHS = 5
EPS_CLIP = 0.2
UPDATE_TIMESTEP = 5

# fault tolerance related
FAIL_PROB_FOG_BASE = 0.03
FAIL_PROB_CLOUD_BASE = 0.1
FAIL_REPAIR_TIME = 5.0
BACKUP_LAUNCH_TIME = 3.0

# kmeans
def kmeans_clustering(device_features, k=2, max_iter=10):
    N = device_features.shape[0]
    centers = device_features[np.random.choice(N, k, replace=False), :]
    for _ in range(max_iter):
        dist = np.sqrt(((device_features[:,None,:] - centers[None,:,:])**2).sum(axis=2))
        labels = dist.argmin(axis=1)
        new_centers=[]
        for c in range(k):
            pts = device_features[labels==c]
            if len(pts)>0:
                new_centers.append(pts.mean(axis=0))
            else:
                new_centers.append(device_features[np.random.randint(N)])
        new_centers = np.array(new_centers)
        if np.allclose(centers,new_centers):
            break
        centers=new_centers
    return labels

# Memory
class Memory:
    def __init__(self):
        self.states=[]
        self.actions=[]
        self.logprobs=[]
        self.rewards=[]
        self.is_terminals=[]
    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()

# PPO (MLP) => FedAdapt
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1=nn.Linear(state_dim,32)
        self.fc2=nn.Linear(32,32)
        self.actor=nn.Linear(32,action_dim)
        self.critic=nn.Linear(32,1)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return x

    def act(self, state):
        logits=self.actor(self.forward(state))
        dist=torch.distributions.Categorical(logits=logits)
        action= dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, state, action):
        x=self.forward(state)
        logits=self.actor(x)
        dist=torch.distributions.Categorical(logits=logits)
        action_logprobs= dist.log_prob(action)
        dist_entropy= dist.entropy()
        state_values= self.critic(x)
        return action_logprobs, torch.squeeze(state_values), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim):
        self.gamma=RL_GAMMA
        self.eps_clip=EPS_CLIP
        self.K_epochs=K_EPOCHS

        self.policy=ActorCritic(state_dim, action_dim)
        self.policy_old=ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer=optim.Adam(self.policy.parameters(), lr=RL_LR, betas=RL_BETAS)
        self.mseLoss=nn.MSELoss()

    def select_action(self, state, memory):
        st=torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, logp= self.policy_old.act(st[0])
        memory.states.append(state)
        memory.actions.append(action.item())
        memory.logprobs.append(logp.item())
        return action.item()

    def update(self, memory):
        n_data=len(memory.states)
        if n_data<1:
            debug_print("[PPO] skip => not enough data")
            memory.clear_memory()
            return

        rewards=[]
        discounted=0
        for r,is_term in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_term:
                discounted=0
            discounted= r + self.gamma*discounted
            rewards.insert(0, discounted)
        rewards=torch.FloatTensor(rewards)
        stdv=rewards.std()
        if stdv<1e-9:
            debug_print("[PPO] reward std too small => skip update.")
            memory.clear_memory()
            return

        if len(memory.states)!= len(memory.rewards):
            debug_print(f"ERROR: states vs. rewards mismatch => {len(memory.states)} {len(memory.rewards)}")

        rewards=(rewards- rewards.mean())/(stdv+1e-5)
        old_states=torch.FloatTensor(memory.states)
        old_actions=torch.LongTensor(memory.actions)
        old_logprobs=torch.FloatTensor(memory.logprobs)

        for _ in range(self.K_epochs):
            logps, vals, dist_entropy= self.policy.evaluate(old_states, old_actions)
            ratio=torch.exp(logps- old_logprobs.detach())
            adv= rewards- vals.detach()

            surr1= ratio*adv
            surr2= torch.clamp(ratio,1-self.eps_clip,1+self.eps_clip)*adv
            loss= -torch.min(surr1,surr2)+ 0.5*self.mseLoss(vals,rewards)-0.01*dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(),1.0)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        memory.clear_memory()


# FaultTolerantTransformerPPO(FTFormer)
class FaultTolerantTransformerActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=128, nhead=8, num_layers=6, dropout=0.2):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.embedding= nn.Linear(state_dim, d_model)
        encoder_layer= nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer= nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.actor= nn.Linear(d_model, action_dim)
        self.critic= nn.Linear(d_model,1)

    def forward(self, x_multi):
        b, T, sdim= x_multi.shape
        emb_list=[]
        for i in range(T):
            e= self.embedding(x_multi[:,i])
            emb_list.append(e)
        emb= torch.stack(emb_list, dim=1)
        emb= self.dropout(emb)
        emb= emb.transpose(0,1)
        out= self.transformer(emb)
        out= out.mean(dim=0)
        return out

    def act(self, x_multi):
        if x_multi.dim()==2:
            x_multi= x_multi.unsqueeze(0)
        out= self.forward(x_multi)
        logits= self.actor(out)
        dist= torch.distributions.Categorical(logits=logits)
        action= dist.sample()
        logp= dist.log_prob(action)
        return action[0], logp[0]

    def evaluate(self, x_multi, action):
        out= self.forward(x_multi)
        logits= self.actor(out)
        dist= torch.distributions.Categorical(logits=logits)
        action_logprobs= dist.log_prob(action)
        dist_entropy= dist.entropy()
        values= self.critic(out).squeeze()
        return action_logprobs, values, dist_entropy

class FaultTolerantTransformerPPO:
    def __init__(self, state_dim, action_dim):
        self.gamma= RL_GAMMA
        self.eps_clip= EPS_CLIP
        self.K_epochs=K_EPOCHS

        self.policy= FaultTolerantTransformerActorCritic(
            state_dim= state_dim, action_dim= action_dim,
            d_model=128, nhead=8, num_layers=6, dropout=0.2
        )
        self.policy_old= FaultTolerantTransformerActorCritic(
            state_dim= state_dim, action_dim= action_dim,
            d_model=128, nhead=8, num_layers=6, dropout=0.2
        )
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer= optim.Adam(self.policy.parameters(), lr=RL_LR, betas=RL_BETAS)
        self.mseLoss= nn.MSELoss()

    def build_tokens(self, edge_state, fog_state, cloud_state):
        return np.array([edge_state, fog_state, cloud_state], dtype=np.float32)

    def select_action(self, x_multi, memory):
        x_t= torch.FloatTensor(x_multi)
        with torch.no_grad():
            a, lp= self.policy_old.act(x_t)
        memory.states.append(x_multi)
        memory.actions.append(a.item())
        memory.logprobs.append(lp.item())
        return a.item()

    def update(self, memory):
        n_data= len(memory.states)
        if n_data<1:
            debug_print("[FaultTolerantTransformerPPO] skip => no data")
            memory.clear_memory()
            return

        rewards=[]
        discounted=0
        for r,done in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if done:
                discounted=0
            discounted= r + self.gamma* discounted
            rewards.insert(0, discounted)
        rewards= torch.FloatTensor(rewards)
        stdv= rewards.std()
        if stdv<1e-9:
            debug_print("[FaultTolerantTransformerPPO] reward std => skip")
            memory.clear_memory()
            return

        if len(memory.states)!= len(memory.rewards):
            debug_print(f"ERROR: states vs. rewards mismatch => {len(memory.states)} {len(memory.rewards)}")

        rewards=(rewards- rewards.mean())/(stdv+1e-5)
        multi_states= np.stack(memory.states, axis=0)
        old_states= torch.FloatTensor(multi_states)
        old_actions= torch.LongTensor(memory.actions)
        old_logprobs= torch.FloatTensor(memory.logprobs)

        for _ in range(self.K_epochs):
            logps, vals, dist_entropy= self.evaluate(old_states, old_actions)
            ratio= torch.exp(logps- old_logprobs.detach())
            adv= rewards- vals.detach()

            surr1= ratio*adv
            surr2= torch.clamp(ratio,1-self.eps_clip,1+self.eps_clip)*adv
            loss= -torch.min(surr1,surr2)+ 0.5*self.mseLoss(vals,rewards)-0.01*dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(),1.0)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        memory.clear_memory()

    def evaluate(self, x_multi, action):
        logps, vals, dist_entropy= self.policy.evaluate(x_multi, action)
        return logps, vals, dist_entropy

# Node + Fault Tolerance
class Node:
    def __init__(self, name, capacity, compute_power, simulator, parent=None):
        self.name=name
        self.capacity=capacity
        self.compute_power=compute_power
        self.sim=simulator
        self.parent=parent

        self.active_tasks=0
        self.waiting_queue=[]
        self.is_failed=False
        self.fail_recover_time=0.0

    def add_task(self, tkey, current_time, event_queue):
        if self.is_failed:
            debug_print(f"[Node={self.name}] is FAILED => handle_fail_task for {tkey}")
            self.handle_fail_task(tkey, current_time, event_queue)
            return
        if self.active_tasks< self.capacity:
            self.start_task(tkey, current_time, event_queue)
        else:
            self.waiting_queue.append(tkey)

    def start_task(self, tkey, current_time, event_queue):
        if self.is_failed:
            debug_print(f"[Node={self.name}] start_task but FAIL => handle_fail_task")
            self.handle_fail_task(tkey, current_time, event_queue)
            return
        self.active_tasks+=1
        offp= self.sim.task_offload_point[tkey]
        csum= sum(EXTENDED_LAYERS_COMPUTE[offp: EXTENDED_LAYERS])
        ctime= csum/ self.compute_power
        finish_time= current_time+ ctime
        e_id= self.sim.next_event_id()
        if "Fog" in self.name:
            e_type= "finish_fog_task"
        else:
            e_type= "finish_cloud_task"
        heapq.heappush(event_queue,(finish_time,e_id,e_type,self,tkey))

    def finish_task(self, tkey, finish_time, event_queue):
        self.active_tasks-=1
        if self.waiting_queue:
            nxt= self.waiting_queue.pop(0)
            self.start_task(nxt, finish_time, event_queue)

    def handle_fail_task(self, tkey, current_time, event_queue):
        if "Fog" in self.name:
            debug_print(f"{tkey} => Fog fail => offload2Cloud")
            offp= self.sim.task_offload_point[tkey]
            data_sum= sum(EXTENDED_LAYERS_DATA[offp: EXTENDED_LAYERS])
            bw= self.sim.fog_cloud_bw
            t_up= data_sum/ bw
            arrival= current_time+ t_up
            e_id= self.sim.next_event_id()
            heapq.heappush(event_queue,(arrival,e_id,"arrive_task_cloud",self.parent,tkey))
        else:
            debug_print(f"{tkey} => Cloud fail => do BACKUP")
            finish_backup= current_time+ BACKUP_LAUNCH_TIME
            e_id= self.sim.next_event_id()
            heapq.heappush(event_queue,(finish_backup,e_id,"backup_done",self,tkey))

    def do_fail(self, fail_time, event_queue):
        debug_print(f"[Node={self.name}] do_fail => fail_time= {fail_time}")
        self.is_failed= True
        self.fail_recover_time= fail_time+ FAIL_REPAIR_TIME
        e_id= self.sim.next_event_id()
        heapq.heappush(event_queue,(self.fail_recover_time,e_id,"recover_node",self,None))

    def do_recover(self):
        self.is_failed= False
        debug_print(f"[Node={self.name}] do_recover => now is OK")

class FogNode(Node):
    pass

class CloudNode(Node):
    pass

class CloudGroup:
    def __init__(self, cloud_nodes):
        self.cloud_nodes = cloud_nodes
        self.next_idx = 0

    def add_task(self, tkey, current_time, event_queue):
        selected_cloud = self.cloud_nodes[self.next_idx]
        self.next_idx = (self.next_idx + 1) % len(self.cloud_nodes)
        selected_cloud.add_task(tkey, current_time, event_queue)

    def finish_task(self, tkey, finish_time, event_queue):
        selected_cloud = self.cloud_nodes[0]
        selected_cloud.finish_task(tkey, finish_time, event_queue)

    def __str__(self):
        return f"CloudGroup({', '.join(str(cloud) for cloud in self.cloud_nodes)})"

# Simulator
class Simulator:
    def __init__(self, seed=None, fog_fail_prob=FAIL_PROB_FOG_BASE, cloud_fail_prob=FAIL_PROB_CLOUD_BASE):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.fog_fail_prob= fog_fail_prob
        self.cloud_fail_prob= cloud_fail_prob

        self.global_event_id=0
        self.edge_compute_powers= [random.uniform(EDGE_POWER_MIN, EDGE_POWER_MAX) for _ in range(NUM_EDGES)]

        # Build network
        self.clouds = [CloudNode(f"Cloud_{i}", CLOUD_CAPACITY, CLOUD_COMPUTE_POWER, self) for i in range(NUM_CLOUDS)]
        self.cloud_group = CloudGroup(self.clouds)
        self.fogs = []
        edges_per_fog = NUM_EDGES // NUM_FOGS
        for i in range(NUM_FOGS):
            fog = FogNode(f"Fog_{i}", FOG_CAPACITY, FOG_COMPUTE_POWER, self, parent=self.cloud_group)
            self.fogs.append(fog)

        self.edge_to_fog_map={}
        for i in range(NUM_EDGES):
            idx= i// edges_per_fog
            self.edge_to_fog_map[i]= self.fogs[idx]

        self.logs=[]

        self.edge_fog_bw= (EDGE_FOG_BW_RANGE[0]+ EDGE_FOG_BW_RANGE[1])/2
        self.fog_cloud_bw= (FOG_CLOUD_BW_RANGE[0]+ FOG_CLOUD_BW_RANGE[1])/2
        self.fog_edge_bw= (FOG_EDGE_BW_RANGE[0]+ FOG_EDGE_BW_RANGE[1])/2
        self.cloud_edge_bw= (CLOUD_EDGE_BW_RANGE[0]+ CLOUD_EDGE_BW_RANGE[1])/2

        self.task_offload_point={}
        self.last_round_time=10.0

        self.memory= Memory()
        self.ppo= PPO(state_dim=3, action_dim=EXTENDED_LAYERS+1)
        self.update_timestep=0

        self.ft_memory= Memory()
        self.ft_ppo= FaultTolerantTransformerPPO(state_dim=9, action_dim=EXTENDED_LAYERS+1)
        self.ft_update_timestep=0

    def next_event_id(self):
        self.global_event_id+=1
        return self.global_event_id

    def log_event(self, time, etype, node, tkey, extra_info=""):
        edge_id, round_id= tkey if tkey else(-1,-1)
        item={
            "time": round(time,3),
            "event_type": etype,
            "node": str(node),
            "edge_id": edge_id,
            "round_id": round_id,
            "extra_info": extra_info
        }
        self.logs.append(item)
        print(f"[LOG] t={item['time']}, event={etype}, node={node}, edge={edge_id}, round={round_id}, info={extra_info}")

    def _fail_check(self, event_queue, cur_time):
        for fog in self.fogs:
            if not fog.is_failed:
                if random.random() < self.fog_fail_prob:
                    fog.do_fail(cur_time, event_queue)

        for cloud in self.cloud_group.cloud_nodes:
            if not cloud.is_failed:
                if random.random() < self.cloud_fail_prob:
                    cloud.do_fail(cur_time, event_queue)

    def _reset_servers(self):
        pass

    def simulate_no_offload(self, num_rounds):
        comp_sum= sum(EXTENDED_LAYERS_COMPUTE)
        cur_time=0.0
        for r in range(num_rounds):
            local_times=[]
            for i in range(NUM_EDGES):
                t_i= comp_sum/ self.edge_compute_powers[i]
                local_times.append(t_i)
            cur_time+= max(local_times)
        return cur_time

    def simulate_fedadapt_ppo(self, num_rounds):
        cur_time=0.0
        for r in range(num_rounds):
            event_queue=[]
            self._reset_servers()

            self.edge_fog_bw= random.uniform(*EDGE_FOG_BW_RANGE)
            self.fog_cloud_bw= random.uniform(*FOG_CLOUD_BW_RANGE)
            self.fog_edge_bw= random.uniform(*FOG_EDGE_BW_RANGE)
            self.cloud_edge_bw= random.uniform(*CLOUD_EDGE_BW_RANGE)

            feats=[]
            for i in range(NUM_EDGES):
                feats.append([ self.edge_compute_powers[i], self.last_round_time,
                               (self.edge_fog_bw+self.fog_edge_bw)/2 ])
            feats=np.array(feats)
            labels= kmeans_clustering(feats,k=2)

            groupA= (labels==0).sum()
            groupB= (labels==1).sum()
            centerA= feats[labels==0].mean(axis=0) if groupA>0 else np.array([1.0,self.last_round_time,5.0])
            centerB= feats[labels==1].mean(axis=0) if groupB>0 else np.array([1.0,self.last_round_time,5.0])

            actA= None
            actB= None
            act_count=0

            if groupA>0:
                actA= self.ppo.select_action(centerA, self.memory)
                act_count+=1
            if groupB>0:
                actB= self.ppo.select_action(centerB, self.memory)
                act_count+=1

            self._fail_check(event_queue, cur_time)

            finished_tasks=0
            last_event_time= cur_time
            for i in range(NUM_EDGES):
                tkey=(i,r)
                if labels[i]==0 and actA is not None:
                    offp= actA
                elif labels[i]==1 and actB is not None:
                    offp= actB
                else:
                    # fallback
                    offp= 0
                self.task_offload_point[tkey]= offp

                local_comp= sum(EXTENDED_LAYERS_COMPUTE[:offp])
                ctime= local_comp/ self.edge_compute_powers[i]
                finish_local= cur_time+ ctime
                data_up= sum(EXTENDED_LAYERS_DATA[:offp])
                t_up= data_up/ self.edge_fog_bw
                arrival_fog= finish_local+ t_up

                e_id= self.next_event_id()
                node= self.edge_to_fog_map[i]
                heapq.heappush(event_queue,(arrival_fog,e_id,"arrive_task_fog",node,tkey))

            while finished_tasks< NUM_EDGES:
                if not event_queue:
                    break
                etime,_, etype,node,tkey= heapq.heappop(event_queue)
                self.log_event(etime, etype,node,tkey)
                last_event_time= etime

                if etype=="arrive_task_fog":
                    node.add_task(tkey, etime, event_queue)
                elif etype=="finish_fog_task":
                    node.finish_task(tkey, etime,event_queue)
                elif etype=="arrive_task_cloud":
                    node.add_task(tkey, etime,event_queue)
                elif etype=="finish_cloud_task":
                    node.finish_task(tkey, etime,event_queue)
                elif etype=="result_arrive_edge":
                    finished_tasks+=1
                elif etype=="recover_node":
                    node.do_recover()
                elif etype=="backup_done":
                    finished_tasks+=1
                else:
                    debug_print("Unknown event", etype)

            round_time= last_event_time- cur_time
            cur_time= last_event_time

            for _ in range(act_count):
                self.memory.rewards.append(-round_time)
                self.memory.is_terminals.append(True)

            self.update_timestep+=1
            if self.update_timestep % UPDATE_TIMESTEP==0:
                self.ppo.update(self.memory)
                self.update_timestep=0

            self.last_round_time= round_time
        return cur_time

    def simulate_faulttransformer_ppo(self, num_rounds):
        cur_time=0.0
        for r in range(num_rounds):
            event_queue=[]
            self._reset_servers()

            self.edge_fog_bw= random.uniform(*EDGE_FOG_BW_RANGE)
            self.fog_cloud_bw= random.uniform(*FOG_CLOUD_BW_RANGE)
            self.fog_edge_bw= random.uniform(*FOG_EDGE_BW_RANGE)
            self.cloud_edge_bw= random.uniform(*CLOUD_EDGE_BW_RANGE)

            self._fail_check(event_queue, cur_time)

            edge_state= [ self.edge_compute_powers[0], self.last_round_time,
                          (self.edge_fog_bw+self.fog_edge_bw)/2,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            fog_state=  [ 0.0,0.0,0.0,
                          self.fog_fail_prob,self.cloud_fail_prob,
                          FAIL_REPAIR_TIME,BACKUP_LAUNCH_TIME,0.0,0.0]
            cloud_state=[ 0.0,0.0,0.0,
                          0.0,0.0,
                          FAIL_REPAIR_TIME,BACKUP_LAUNCH_TIME,0.0,0.0]
            state_tokens= np.array([edge_state, fog_state, cloud_state], dtype=np.float32)

            act= self.ft_ppo.select_action(state_tokens, self.ft_memory)
            act_count=1

            finished_tasks=0
            last_event_time= cur_time
            for i in range(NUM_EDGES):
                tkey=(i,r)
                self.task_offload_point[tkey]= act
                local_comp= sum(EXTENDED_LAYERS_COMPUTE[:act])
                ctime= local_comp/ self.edge_compute_powers[i]
                finish_local= cur_time+ ctime
                data_up= sum(EXTENDED_LAYERS_DATA[:act])
                t_up= data_up/ self.edge_fog_bw
                arrival_fog= finish_local+ t_up

                e_id= self.next_event_id()
                node= self.edge_to_fog_map[i]
                heapq.heappush(event_queue,(arrival_fog,e_id,"arrive_task_fog",node,tkey))

            while finished_tasks< NUM_EDGES:
                if not event_queue:
                    break
                etime,_, etype,node,tkey= heapq.heappop(event_queue)
                self.log_event(etime, etype,node,tkey)
                last_event_time= etime

                if etype=="arrive_task_fog":
                    node.add_task(tkey, etime,event_queue)
                elif etype=="finish_fog_task":
                    node.finish_task(tkey, etime,event_queue)
                elif etype=="arrive_task_cloud":
                    node.add_task(tkey, etime,event_queue)
                elif etype=="finish_cloud_task":
                    node.finish_task(tkey, etime,event_queue)
                elif etype=="result_arrive_edge":
                    finished_tasks+=1
                elif etype=="recover_node":
                    node.do_recover()
                elif etype=="backup_done":
                    finished_tasks+=1
                else:
                    debug_print("Unknown event", etype)

            round_time= last_event_time- cur_time
            cur_time= last_event_time

            for _ in range(act_count):
                self.ft_memory.rewards.append(-round_time)
                self.ft_memory.is_terminals.append(True)

            self.ft_update_timestep+=1
            if self.ft_update_timestep % UPDATE_TIMESTEP==0:
                self.ft_ppo.update(self.ft_memory)
                self.ft_update_timestep=0

            self.last_round_time= round_time
        return cur_time

    def export_logs_to_csv(self, filename="simulation_log.csv"):
        with open(filename,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f)
            w.writerow(["time","event_type","node","edge_id","round_id","extra_info"])
            for item in self.logs:
                w.writerow([
                    item["time"],
                    item["event_type"],
                    item["node"],
                    item["edge_id"],
                    item["round_id"],
                    item["extra_info"]
                ])
        debug_print(f"Log export to: {filename}")

# run_once_experiment
def run_once_experiment(num_rounds, seed=None):
    sim= Simulator(seed=seed)

    # 1) No offload
    t_no= sim.simulate_no_offload(num_rounds)
    sim.logs.clear()

    # 2) FedAdapt+PPO
    t_feda= sim.simulate_fedadapt_ppo(num_rounds)
    sim.logs.clear()

    # 3) FaultTolerantTransformerPPO(FTFormer)
    t_ft= sim.simulate_faulttransformer_ppo(num_rounds)
    sim.logs.clear()

    return (t_no,
            t_feda,
            t_ft)

# main_experiment
def main_experiment():
    N=3
    results_no=[]
    results_feda=[]
    results_ft=[]

    perc_feda_list=[]
    perc_ft_list=[]

    for i in range(N):
        seed_i= 999 + i
        print(f"\n=== Experiment {i+1}/{N}, seed={seed_i} ===")
        t_no, t_feda, t_ft = run_once_experiment(NUM_ROUNDS, seed=seed_i)

        results_no.append(t_no)
        results_feda.append(t_feda)
        results_ft.append(t_ft)

        p_feda = 100.0*(t_no - t_feda)/t_no
        p_ft = 100.0*(t_no - t_ft)/t_no

        perc_feda_list.append(p_feda)
        perc_ft_list.append(p_ft)

    arr_no= np.array(results_no)
    arr_fd= np.array(results_feda)
    arr_ft= np.array(results_ft)

    arr_pfeda= np.array(perc_feda_list)
    arr_pft= np.array(perc_ft_list)

    mean_no= arr_no.mean()
    std_no= arr_no.std()
    mean_fd= arr_fd.mean()
    std_fd= arr_fd.std()
    mean_ft= arr_ft.mean()
    std_ft= arr_ft.std()

    mean_pfeda= arr_pfeda.mean()
    std_pfeda= arr_pfeda.std()
    mean_pft= arr_pft.mean()
    std_pft= arr_pft.std()

    print("\n================ Experiment result ==================")
    print("Approach                                  | Average time(±std)   | Saved %(±std)")
    print("---------------------------------------------------------------------")
    print(f"No Offload                               | {mean_no:.3f} (±{std_no:.2f})  |  0.00% (±0.00)")
    print(f"FedAdapt+PPO (MLP)                       | {mean_fd:.3f} (±{std_fd:.2f})  | {mean_pfeda:.2f}% (±{std_pfeda:.2f})")
    print(f"FaultTolerantTransformerPPO (FTFormer)   | {mean_ft:.3f} (±{std_ft:.2f})  | {mean_pft:.2f}% (±{std_pft:.2f})")
    print("=====================================================================\n")

def main():
    main_experiment()

if __name__=="__main__":
    main()
