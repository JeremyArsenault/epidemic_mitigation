import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import copy

class ActorModel(nn.Module):
    """
    Model for learning agent movement behaviors
    
    Input: []
    Output: 3x3 movement nums used to calculate log-prob of movement.
    """
    def __init__(self, comm_layers=0):
        super().__init__()
        self.comm_layers = comm_layers
        
        self.l1 = torch.nn.Linear(8+9*(comm_layers+1),64)
        self.l2 = torch.nn.Linear(64,32)
        self.l3 = torch.nn.Linear(32,16)
        self.l4 = torch.nn.Linear(16,9)
        if self.comm_layers:
            self.lc = torch.nn.Linear(16,comm_layers)

    def forward(self, x):
        x = f.relu(self.l1(x))
        x = f.relu(self.l2(x))
        x = f.relu(self.l3(x))
        m = torch.clamp(self.l4(x), -3, 3)
        m = torch.exp(m)
        m = m / m.sum(dim=1, keepdim=True).repeat(1, m.shape[1])
        if self.comm_layers:
            c = torch.sigmoid(self.lc(x))
            return m, c
        else:
            return m
    
    def sample_move(self, p):
        p = p.detach().numpy()
        return [np.random.choice(9, p=x) for x in p]
    
class CriticModel(torch.nn.Module):
    """
    Model for scoring agent movement behaviors and communication
    
    Input: []
    Output: score scalar
    """
    def __init__(self, comm_layers=0):
        super().__init__()
        self.comm_layers = comm_layers
        
        self.l1 = torch.nn.Linear(8+9*(comm_layers+1),64)
        self.l2 = torch.nn.Linear(64,32)
        self.l3 = torch.nn.Linear(32,16)
        self.l4 = torch.nn.Linear(16,1)

    def forward(self, x):
        x = f.relu(self.l1(x))
        x = f.relu(self.l2(x))
        x = f.relu(self.l3(x))
        x = self.l4(x)
        return x
    
def print_progress(n, tot, score, bar_size=50):
    scale = bar_size / tot
    n = int(scale*n)
    completed = ''.join([u'\u2588' for _ in range(n)])
    todo = ''.join([' ' for _ in range(bar_size-n)])
    print('['+completed+todo+']'+' score:'+str(round(score,2)), "\r", end="")
    
def train(env, actor_model, critic_model, save_name, episodes=200, replay_bank_size=10, updates_per_epoch=200):
    """
    Training loop. Vanilla PGM
    """
    comm_b = env.communication_layers!=0
    
    actor_optim = torch.optim.Adam(actor_model.parameters(), lr=0.00005)
    # actor_loss = nn.CrossEntropyLoss(reduction='none')
    critic_optim = torch.optim.Adam(critic_model.parameters(), lr=0.0001)
    critic_loss = nn.MSELoss()
    
    hist = []
    
    # [[o1, act, o2], rew]
    replay_bank = []
    for episode in range(episodes):
        
        # do simulation
        actor_model.eval()
        env.reset()
        done = False
        os =  []
        while not done:
            o1s, acts, o2s = [], [], []
            for a in range(env.n_agents):
                o1 = env.observe(a)
                if comm_b:
                    act, comm = actor_model(torch.Tensor([o1]))
                    act = actor_model.sample_move(act)[0]
                    comm = comm.detach().numpy()
                    env.step(act, a, comm)
                    acts.append([act, np.sign(comm)])
                else:
                    act = actor_model.sample_move(actor_model(torch.Tensor([o1])))[0]
                    env.step(act, a)
                    acts.append(act)
                o1s.append(o1)
                o2s.append(env.observe(a))
            os.append([o1s, acts, o2s])
            done = env.timestep()
        scores = env.score()
        
        replay_bank.append([os, scores])
        replay_bank = replay_bank[-replay_bank_size:]
        
        # update value network
        critic_model.train()
        for i in np.random.choice(len(replay_bank), updates_per_epoch):
            o, scores = replay_bank[i]
            _, _, obs = o[np.random.randint(env.n_iter)]
            x, y = torch.tensor(obs), torch.tensor(scores)
            
            critic_optim.zero_grad()
            l = critic_loss(critic_model(x.float()).flatten(), y.float())
            l.backward()
            critic_optim.step()
            
        # update policy network
        actor_model.train()
        critic_model.eval()
        for i in np.random.choice(len(replay_bank), updates_per_epoch):
            o1, a, o2 = replay_bank[i][0][np.random.randint(env.n_iter)]
            if comm_b:
                o1, ao, co, o2 = torch.tensor(o1), torch.tensor([[x[0]] for x in a]), torch.tensor([x[1] for x in a]), torch.tensor(o2)
            else:
                o1, ao, o2 = torch.tensor(o1), torch.tensor(a), torch.tensor(o2)
            adv = critic_model(o1.float()) - critic_model(o2.float()) 
            
            actor_optim.zero_grad()
            if comm_b:
                a, c = actor_model(o1.float())
                l = (torch.log(a.gather(1,ao.long())) * adv.float()).mean()
                l += 0.5*(co * comm * adv.float()).mean()
            else:
                l = (torch.log(actor_model(o1.float()).gather(1,ao.unsqueeze(1))) * adv.float()).mean()
            l.backward()
            actor_optim.step()
        
        # print progress
        print_progress(episode, episodes, sum(scores) / len(scores))
        hist.append(sum(scores) / len(scores))
        
    torch.save(actor_model, f'models/{save_name}_actor.torch')
    torch.save(critic_model, f'models/{save_name}_critic.torch')
    
    return hist

    
if __name__=='__main__':
    pass