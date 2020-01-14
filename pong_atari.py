import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np
import gym 
#from torch.utils.tensorboard import SummaryWriter
from collections import deque
import random
import os
os.environ.setdefault('PATH', '')
from collections import deque
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)

# helpers
def init_weights(m):
	if isinstance(m,nn.Linear):
		torch.nn.init.normal_(m.weight,0.,0.1)

class ReplayBuffer:
	def __init__(self,size):
		self.size = size
		self.memory = deque([],maxlen=size)

	def push(self, x):
		self.memory.append(x)
	
	def sample(self, batch_size):
		batch = random.sample(self.memory,batch_size)
		state, action, reward, next_state, done = map(np.stack, zip(*batch))
		return state, action, reward, next_state, done

	def get_len(self):
		return len(self.memory)

class TimeLimit(gym.Wrapper):
	def __init__(self, env, max_episode_steps=None):
		super(TimeLimit, self).__init__(env)
		self._max_episode_steps = max_episode_steps
		self._elapsed_steps = 0

	def step(self, ac):
		observation, reward, done, info = self.env.step(ac)
		self._elapsed_steps += 1
		if self._elapsed_steps >= self._max_episode_steps:
			done = True
			info['TimeLimit.truncated'] = True
		return observation, reward, done, info

	def reset(self, **kwargs):
		self._elapsed_steps = 0
		return self.env.reset(**kwargs)

class ClipActionsWrapper(gym.Wrapper):
	def step(self, action):
		import numpy as np
		action = np.nan_to_num(action)
		action = np.clip(action, self.action_space.low, self.action_space.high)
		return self.env.step(action)

	def reset(self, **kwargs):
		return self.env.reset(**kwargs)

class NoopResetEnv(gym.Wrapper):
	def __init__(self, env, noop_max=30):
		"""Sample initial states by taking random number of no-ops on reset.
		No-op is assumed to be action 0.
		"""
		gym.Wrapper.__init__(self, env)
		self.noop_max = noop_max
		self.override_num_noops = None
		self.noop_action = 0
		assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

	def reset(self, **kwargs):
		""" Do no-op action for a number of steps in [1, noop_max]."""
		self.env.reset(**kwargs)
		if self.override_num_noops is not None:
			noops = self.override_num_noops
		else:
			noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
		assert noops > 0
		obs = None
		for _ in range(noops):
			obs, _, done, _ = self.env.step(self.noop_action)
			if done:
				obs = self.env.reset(**kwargs)
		return obs

	def step(self, ac):
		return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
	def __init__(self, env):
		"""Take action on reset for environments that are fixed until firing."""
		gym.Wrapper.__init__(self, env)
		assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
		assert len(env.unwrapped.get_action_meanings()) >= 3

	def reset(self, **kwargs):
		self.env.reset(**kwargs)
		obs, _, done, _ = self.env.step(1)
		if done:
			self.env.reset(**kwargs)
		obs, _, done, _ = self.env.step(2)
		if done:
			self.env.reset(**kwargs)
		return obs

	def step(self, ac):
		return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
	def __init__(self, env):
		"""Make end-of-life == end-of-episode, but only reset on true game over.
		Done by DeepMind for the DQN and co. since it helps value estimation.
		"""
		gym.Wrapper.__init__(self, env)
		self.lives = 0
		self.was_real_done  = True

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		self.was_real_done = done
		# check current lives, make loss of life terminal,
		# then update lives to handle bonus lives
		lives = self.env.unwrapped.ale.lives()
		if lives < self.lives and lives > 0:
			# for Qbert sometimes we stay in lives == 0 condition for a few frames
			# so it's important to keep lives > 0, so that we only reset once
			# the environment advertises done.
			done = True
		self.lives = lives
		return obs, reward, done, info

	def reset(self, **kwargs):
		"""Reset only when lives are exhausted.
		This way all states are still reachable even though lives are episodic,
		and the learner need not know about any of this behind-the-scenes.
		"""
		if self.was_real_done:
			obs = self.env.reset(**kwargs)
		else:
			# no-op step to advance from terminal/lost life state
			obs, _, _, _ = self.env.step(0)
		self.lives = self.env.unwrapped.ale.lives()
		return obs

class MaxAndSkipEnv(gym.Wrapper):
	def __init__(self, env, skip=4):
		"""Return only every `skip`-th frame"""
		gym.Wrapper.__init__(self, env)
		# most recent raw observations (for max pooling across time steps)
		self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
		self._skip       = skip

	def step(self, action):
		"""Repeat action, sum reward, and max over last observations."""
		total_reward = 0.0
		done = None
		for i in range(self._skip):
			obs, reward, done, info = self.env.step(action)
			if i == self._skip - 2: self._obs_buffer[0] = obs
			if i == self._skip - 1: self._obs_buffer[1] = obs
			total_reward += reward
			if done:
				break
		# Note that the observation on the done=True frame
		# doesn't matter
		max_frame = self._obs_buffer.max(axis=0)

		return max_frame, total_reward, done, info

	def reset(self, **kwargs):
		return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
	def __init__(self, env):
		gym.RewardWrapper.__init__(self, env)

	def reward(self, reward):
		"""Bin reward to {+1, 0, -1} by its sign."""
		return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
	def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
		"""
		Warp frames to 84x84 as done in the Nature paper and later work.
		If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
		observation should be warped.
		"""
		super().__init__(env)
		self._width = width
		self._height = height
		self._grayscale = grayscale
		self._key = dict_space_key
		if self._grayscale:
			num_colors = 1
		else:
			num_colors = 3

		new_space = gym.spaces.Box(
			low=0,
			high=255,
			shape=(self._height, self._width, num_colors),
			dtype=np.uint8,
		)
		if self._key is None:
			original_space = self.observation_space
			self.observation_space = new_space
		else:
			original_space = self.observation_space.spaces[self._key]
			self.observation_space.spaces[self._key] = new_space
		assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

	def observation(self, obs):
		if self._key is None:
			frame = obs
		else:
			frame = obs[self._key]

		if self._grayscale:
			frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
		frame = cv2.resize(
			frame, (self._width, self._height), interpolation=cv2.INTER_AREA
		)
		if self._grayscale:
			frame = np.expand_dims(frame, -1)

		if self._key is None:
			obs = frame
		else:
			obs = obs.copy()
			obs[self._key] = frame
		return obs

class FrameStack(gym.Wrapper):
	def __init__(self, env, k):
		"""Stack k last frames.
		Returns lazy array, which is much more memory efficient.
		See Also
		--------
		baselines.common.atari_wrappers.LazyFrames
		"""
		gym.Wrapper.__init__(self, env)
		self.k = k
		self.frames = deque([], maxlen=k)
		shp = env.observation_space.shape
		self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

	def reset(self):
		ob = self.env.reset()
		for _ in range(self.k):
			self.frames.append(ob)
		return self._get_ob()

	def step(self, action):
		ob, reward, done, info = self.env.step(action)
		self.frames.append(ob)
		return self._get_ob(), reward, done, info

	def _get_ob(self):
		assert len(self.frames) == self.k
		return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
	def __init__(self, env):
		gym.ObservationWrapper.__init__(self, env)
		self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

	def observation(self, observation):
		# careful! This undoes the memory optimization, use
		# with smaller replay buffers only.
		return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
	def __init__(self, frames):
		"""This object ensures that common frames between the observations are only stored once.
		It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
		buffers.
		This object should only be converted to numpy array before being passed to the model.
		You'd not believe how complex the previous solution was."""
		self._frames = frames
		self._out = None

	def _force(self):
		if self._out is None:
			self._out = np.concatenate(self._frames, axis=-1)
			self._frames = None
		return self._out

	def __array__(self, dtype=None):
		out = self._force()
		if dtype is not None:
			out = out.astype(dtype)
		return out

	def __len__(self):
		return len(self._force())

	def __getitem__(self, i):
		return self._force()[i]

	def count(self):
		frames = self._force()
		return frames.shape[frames.ndim - 1]

	def frame(self, i):
		return self._force()[..., i]

def make_atari(env_id, max_episode_steps=None):
	env = gym.make(env_id)
	assert 'NoFrameskip' in env.spec.id
	env = NoopResetEnv(env, noop_max=30)
	env = MaxAndSkipEnv(env, skip=4)
	if max_episode_steps is not None:
		env = TimeLimit(env, max_episode_steps=max_episode_steps)
	return env

def wrap_deepmind(env, episodic_life=True, clip_rewards=True, frame_stack=True, scale=False):
	"""Configure environment for DeepMind-style Atari.
	"""
	if episodic_life:
		env = EpisodicLifeEnv(env)
	if 'FIRE' in env.unwrapped.get_action_meanings():
		env = FireResetEnv(env)
	env = WarpFrame(env)
	if scale:
		env = ScaledFloatFrame(env)
	if frame_stack:
		env = FrameStack(env, 4)
	if clip_rewards:
		env = ClipRewardEnv(env)
   
	return env

class ImageToPyTorch(gym.ObservationWrapper):
	"""
	Image shape to num_channels x weight x height
	"""
	def __init__(self, env):
		super(ImageToPyTorch, self).__init__(env)
		old_shape = self.observation_space.shape
		self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8)

	def observation(self, observation):
		return np.swapaxes(observation, 2, 0)
	
def wrap_pytorch(env):
	return ImageToPyTorch(env)

# Hyperparameters
BATCH_SIZE = 64
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_UPDATE_INTERVAL = 100
TENSORBOARD_LOG = False
#TB_LOG_PATH = './runs/dqn/run2'
REPLAY_BUFFER_CAPACITY = 100
env = make_atari("PongNoFrameskip-v4")
env = wrap_deepmind(env)
env = wrap_pytorch(env)
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n

# network definition
class Flatten(torch.nn.Module):
	def forward(self, x):
		batch_size = x.shape[0]
		return x.view(batch_size, -1)       

class ConvDQN(nn.Module):
	def __init__(self):
		super(ConvDQN, self).__init__()

		self.c1 = nn.Conv2d(STATE_DIM , 32, kernel_size = 8, stride = 4)
		self.c2 = nn.Conv2d(32 , 64, kernel_size = 4, stride = 2)
		self.c3 = nn.Conv2d(64 , 64, kernel_size = 3, stride = 1)
		self.fc1 = nn.Linear(7*7*64, 512)
		self.fc2 = nn.Linear(512, ACTION_DIM)
	
		self.apply(init_weights)

		
	def forward(self,x):
		x = F.relu(self.c1(x))
		x = F.relu(self.c2(x))
		x = F.relu(self.c3(x))
		x = Flatten().forward(x)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		
		return x
  
class DQN(nn.Module):
	def __init__(self):
		super(DQN,self).__init__()

		self.fc1 = nn.Linear(STATE_DIM,50)
		self.fc2 = nn.Linear(50,ACTION_DIM)

		self.apply(init_weights)

	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = self.fc2(x)

		return x

class Agent(object):
	def __init__(self):
		self.dqn, self.target_dqn = ConvDQN(), ConvDQN()

		self.learn_step_counter = 0
		self.memory_counter = 0
		self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)
		self.optimizer = opt.Adam(self.dqn.parameters(),lr=LR)
		self.loss_fn = nn.MSELoss()
		

	def get_action(self, s):
		s = torch.unsqueeze(torch.FloatTensor(s),0)

		if np.random.uniform() < EPSILON:
			qs = self.dqn.forward(s)
			action = torch.max(qs,1)[1].data.numpy()
			action = action[0]
		else:
			action = env.action_space.sample()

		return action

	def update_params(self):
		# update target network
		if self.learn_step_counter % TARGET_UPDATE_INTERVAL == 0:
			self.target_dqn.load_state_dict(self.dqn.state_dict())
		self.learn_step_counter += 1

		# sample batch of transitions
		states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

		states = torch.FloatTensor(states)
		actions = torch.LongTensor(actions.astype(int).reshape((-1,1)))
		rewards = torch.FloatTensor(rewards).unsqueeze(1)
		next_states = torch.FloatTensor(next_states)
		dones = torch.FloatTensor(np.float32(dones)).unsqueeze(1)

		# get q values
		q_current = self.dqn(states).gather(1,actions)
		q_next = self.target_dqn(next_states).detach()
		q_target = rewards + GAMMA * q_next.max(1)[0].view(BATCH_SIZE,1)
		q_loss = self.loss_fn(q_current,q_target)
		
		# backpropagate
		self.optimizer.zero_grad()
		q_loss.backward()
		self.optimizer.step()

# create agent
agent = Agent()
#if TENSORBOARD_LOG:
#   writer = SummaryWriter(TB_LOG_PATH)

print('\nCollecting experience')
for ep in range(400):
	state = env.reset()
	episode_reward = 0

	while True:
		env.render()
		action = agent.get_action(state)

		# take action
		next_state, reward_orig, done, _ = env.step(action)

		agent.replay_buffer.push((state,action,reward_orig,next_state,done))
		agent.memory_counter += 1

		episode_reward += reward_orig

		if agent.memory_counter > REPLAY_BUFFER_CAPACITY:
			agent.update_params()

			if done:
				print("Episode: {}, Frames: {}, Reward: {}".format(ep,agent.memory_counter,round(episode_reward,2)))
		
		if done:
			break

		state = next_state
	if TENSORBOARD_LOG:
		writer.add_scalar('episode_reward',episode_reward,ep)
env.close()