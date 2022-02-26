algorithm = 'PPO' # 'DQN

hidden_dim = 128 # number of variables in hidden layer
cuda_device = '0'
max_step = 500 # max timesteps in one episode
GAMMA = 0.99
n_episode = 1000000
i_episode = 0
capacity = 300000 
batch_size = 128
n_epoch = 5 # update policy using 1 trajectory for K epochs
epsilon = 0.9
score = 0
start_train = 100
log_interval = 1
num_agent = 100
update_interval = 5
random_seed = 123

### PPO config
render = False
solved_reward = -40         # stop training if avg_reward > solved_reward
log_interval = 2           # print avg reward in the interval

update_timestep = 1000 # 2000      # update policy every n timesteps
lr = 0.0001
betas = (0.9, 0.999)
eps_clip = 0.2              # clip parameter for PPO

