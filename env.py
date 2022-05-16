import numpy as np

class GridWorld:
    """
    OpenAI Gym - like torodial grid-world for infection.
    Agents can move one cell at a time.
    All agents updated individually.
    
    Objective is to minimize mutational distance reached in a population with one
    or more of the following pro-social objectives: proximity to a central location,
    number of distinct agent-agent interactions, total length of agent-agent interaction.
    
    Virus is transmitted to all succeptible agents in a cell each with probability
    *virulence* at each timestep. In transmission, point mutations bay be accrued 
    according to *mutation_rate* according to a poisson distribution. Effects of 
    backmutation considered negligible.
    
    After infection, the virus is infectious without symptoms for 5 days, then
    infectious with symptoms for 5 timesteps, then ceases to be infectious and
    the infected agent gains immunity.
    """
    def __init__(self,
                 n_iter = 100,
                 dim_x = 20,
                 dim_y = 20,
                 n_agents = 100,
                 mutation_rate = 0.1,
                 virulence = 0.8,
                 communication_layers = 0,
                 initial_infections = 2,
                 prox_incent = 0,
                 num_incent = 0,
                 len_incent = 0,
                 torodial = False,
                ):
        """
        :n_iter: (int) number of steps to simulate for
        :dim_x: (int) ([1,inf) number of simulation iterations. dimension of world
        :dim_x: (int) x dimension of world
        :dim_y: (int) y dimension of world
        :n_agents: (int) number of agents
        :mutation_rate: (float) (0,inf) expected number of point mutations per transmission.
        :virulence: (float) (0,1] Probability of transmission between agents in same cell.
        :communication_layers: (int) communication layers
        :initial_infections: (int) [1,n_agents] number of initial infections
        :prox_incent: (float) [0,1] incentive for l1 distance from world center (max of pop)
        :num_incent: (float) [0,1] incentive for number of agents interacted with (min of pop)
        :len_incent: (float) [0,1] incentive for length of interaction (min of pop)
        :torodial: (bool) True if borderless / torodial environment
        """
        self.n_iter = n_iter
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_agents = n_agents
        self.mutation_rate = mutation_rate
        self.virulence = virulence
        self.communication_layers = communication_layers
        self.initial_infections = initial_infections
        self.prox_incent = prox_incent
        self.num_incent = num_incent
        self.len_incent = len_incent
        self.torodial = torodial
        
        # rendering
        self.viewer = None
        
        # random
        self.poisson = np.zeros(100)
        for k in range(1, 10):
            p = np.exp(-mutation_rate)*sum([np.power(mutation_rate,i)/np.math.factorial(i) for i in range(k)])
            if p<1:
                self.poisson[int(p*100):] = k
        
        self.reset()
    
    def reset(self):
        """
        Reset world, virus, and agents
        """
        # create world
        self.t = 0
        self.agents = [[np.random.randint(self.dim_x), np.random.randint(self.dim_y)] for i in range(self.n_agents)] # [x,y]
        self.infections = [[-int(n>=self.initial_infections),-int(n>=self.initial_infections)] for n in range(self.n_agents)] # [days, num mutations]
        self.last_comm = np.zeros((self.dim_x, self.dim_y, self.communication_layers))
        self.comm = np.zeros((self.dim_x, self.dim_y, self.communication_layers))
        
        # scoring
        self.mutation_num = 0
        self.proximity = [0 for _ in range(self.n_agents)]
        self.contacts = [set() for _ in range(self.n_agents)]
        self.lengths = [0 for _ in range(self.n_agents)]
        self.wall_moves = [0 for _ in range(self.n_agents)]
        
    def observe(self, a):
        """
        observation: symptomatic, dist from center, time, pop virus status, number of contacts, best distance, length of contacts
                     neighbors (9), comm layers (9 each)
        
        :a: (int) agent number
        :returns: observation for each agent
        """
        # build population map
        loc = np.zeros((self.dim_x,self.dim_y))
        for x,y in self.agents:
            loc[x,y] += 1
        loc = 1 - np.power(0.5,loc)
        
        # collect observations
        observation = []
        # symptoms
        d = self.infections[a][0]
        if d<2:
            observation.append(-1)
        elif d<10:
            observation.append(0)
        else:
            observation.append(1)
        # pos
        observation.append((self.agents[a][0]/self.dim_x)-0.5)
        observation.append((self.agents[a][1]/self.dim_y)-0.5)
        # time
        observation.append(self.t/self.n_iter)
        # population status
        observation.append(self.mutation_num/self.n_agents)
        # contacts
        observation.append(len(self.contacts[a])/self.n_agents)
        # best distance
        observation.append(self.proximity[a])
        # lengths
        observation.append(self.lengths[a])
        # population location
        for x in [self.agents[a][0]+i for i in [-1,0,1]]:
            for y in [self.agents[a][1]+i for i in [-1,0,1]]:
                if self.torodial:
                    x, y = x % self.dim_x, y % self.dim_y
                if (x==-1) or (x==self.dim_x) or (y==-1) or (y==self.dim_y):
                    observation.append(-1)
                else:
                    observation.append(loc[x,y])
        # communication
        for x in [self.agents[a][0]+i for i in [-1,0,1]]:
            for y in [self.agents[a][1]+i for i in [-1,0,1]]:
                if self.torodial:
                    x, y = x % self.dim_x, y % self.dim_y
                if (x==-1) or (x==self.dim_x) or (y==-1) or (y==self.dim_y):
                    observation += [-1 for _ in range(self.communication_layers)]
                else:
                    observation += list(self.last_comm[x,y])            
                
        return observation
    
    def inf_status(self, i):
        """
        """
        d, n = self.infections[i]
        if d == -1:
            return -1
        elif d<10:
            return 0
        else:
            return 1
        
    def step(self, action, a, comms=None):
        """
        Move agents and infect
        
        :actios: movement int for each agent and comm floats
        :a: (int) agent number
        :returns: True if done, else False
        """
        # move agents
        x, y = self.agents[a][0] + (action//3)-1, self.agents[a][1] + (action%3)-1
        if self.torodial:
            x, y = x % self.dim_x, y % self.dim_y
        # wall hit
        if (x==-1) or (x==self.dim_x) or (y==-1) or (y==self.dim_y):
            self.wall_moves[a] += 2 / self.n_iter
        self.agents[a] = [min(max((x),0),self.dim_x-1), min(max((y),0),self.dim_y-1)]
        
        # communication
        if comms is not None:
            self.comm[self.agents[a][0], self.agents[a][1]] = np.array(comms)
            
        # increment infection counter
        if self.infections[a][0] >= 0:
            self.infections[a][0] += 1
                
        # create world
        world = [[] for _ in range(self.dim_x*self.dim_y)]
        cell = [i for i, (x,y) in enumerate(self.agents) if (x==self.agents[a][0] and y==self.agents[a][1])]
                
        # do infection
        if self.inf_status(a) == -1: # succeptible
            inf = None
            for i in cell:
                if self.inf_status(a) == -0:
                    inf = self.infections[i][1]
            if (inf is not None) and (np.random.sample() < self.virulence):
                self.infections[a] = [0, inf+self.poisson[np.random.randint(100)]]
        elif self.inf_status(a) == 0: # infectious
            for i in cell:
                if (i!=a) and (self.inf_status(i)==-1) and (np.random.sample() < self.virulence):
                    self.infections[i] = [0, self.infections[a][1]+self.poisson[np.random.randint(100)]]
                        
        # scoring update
        # mutation number
        mut = sum([n+1 for (d,n) in self.infections])
        self.mutation_num = mut
        
        # prox incent
        x,y = self.agents[a]
        _x, _y = abs(x - self.dim_x//2), abs(y - self.dim_y//2)
        dist = 1 - 2*(_x+_y)/(self.dim_x+self.dim_y)
        self.proximity[a] = max(dist, self.proximity[a])
            
        # num and length of contacts
        self.contacts[a].update(cell)
        self.lengths[a] += float(len(cell)>1) / self.n_iter
                
    def timestep(self):
        """
        Call after all agents have taken actions
        """
        for x in self.infections:
            if x[0] != -1:
                x[0] += 1
                
        self.last_comm = self.comm
        self.comm = np.zeros((self.dim_x, self.dim_y, self.communication_layers))
        
        # increment time
        self.t += 1
        return True if self.t>=self.n_iter else False
        
    def score(self):
        """
        Get cumulative score at current timestep. 
        :returns: (float) incentives since last time score fn was called minus
                    sum difference of mutation number since last time score was called
        """                
        return self.prox_incent*(np.array(self.proximity)**2) + \
               self.num_incent*np.array([len(x)/self.n_agents for x in self.contacts]) + \
               self.len_incent*np.array(self.lengths) - \
               self.mutation_num/self.n_agents - \
               np.array(self.wall_moves)
    
    def render(self, console=False):
        """
        Render environment as 2D image

        :returns: image as rgb np array
        """
        screen_size = 400

        def shade_cell(img, x, y, n_agents, mut_n):
            x = (img.shape[0]*x)//self.dim_x
            y = (img.shape[1]*y)//self.dim_y
            
            w = img.shape[0]//self.dim_x
            h = img.shape[1]//self.dim_y
            
            c1 = (0,min(255, n_agents*50),0)
            c2 = (0,0,min(255, mut_n*50))
            img[x:x+w//2, y:y+w] = c1
            img[x+w//2:x+w, y:y+w] = c2

        if self.viewer is None:
            import rendering
            self.viewer = rendering.SimpleImageViewer(maxwidth=screen_size)
            
        # create world
        world = [[] for _ in range(self.dim_x*self.dim_y)]
        for (x,y), (d,n) in zip(self.agents, self.infections):
            if d < 10:
                world[x*self.dim_x+y].append(n)
              
        if console:
            print()
            print(self.t, end='')
            for i, x in enumerate(world):
                if i%self.dim_x==0:
                    print()
                print(x, end='')
                
        # create rendering
        disp = np.zeros((screen_size,screen_size,3), dtype=np.uint8)
        for x in range(self.dim_x):
            for y in range(self.dim_y):
                c = world[x*self.dim_x+y]
                shade_cell(disp, x, y, len(c), max(c+[-1])+1)

        self.viewer.imshow(disp)
        return disp
    
    def close(self):
        self.viewer.close()
        self.viewer = None

if __name__=='__main__':
    import time
    
    n = 200
    env = GridWorld(dim_x=20, dim_y=20, n_agents=n)
    env.reset()

    done = False
    while not done:
        env.render(console=True)
        done = env.step(np.random.randint(0,9,n))
        time.sleep(0.05)
    env.close()
        