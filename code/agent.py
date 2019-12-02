from sim_wrap import Wrapper
from simulator import State
import random
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import argparse

def check_start_state(curr_state):
	if (curr_state.hand_value == -1 and curr_state.value_type == -1 and curr_state.dealer == -1):
		return True
	return False

class Agent:
	def __init__(self):
		self.qsa_map = {}
		self.wrap = Wrapper()
		self.policy = {}

	def init_qsa (self):
		self.qsa_map[(State(-1, -1, -1), "DEAL")] = (0.0, 0)
		for action in ["HIT", "STICK"]:
			for i in xrange(32):
				for j in xrange(3):
					for k in xrange(1, 11):
						self.qsa_map[(State(i, j, k), action)] = (0.0, 0)

	def policy_25 (self, curr_state):
		if (check_start_state(curr_state)):
			action = "DEAL"
		elif (curr_state.hand_value < 25):
			action = "HIT"
		else:
			action = "STICK"
		return action

	def greedy_policy(self, curr_state, epsilon=0.1):
		if (check_start_state(curr_state)):
			return "DEAL"

		indicator = np.random.binomial(1,1 - epsilon + float(epsilon)/2)
		if (indicator == 1):
			return self.policy[curr_state]
		else:
			if (self.policy[curr_state] == "HIT"):
				return "STICK"
			else:
				return "HIT"

	def init_policy (self):
		self.policy[State(-1, -1, -1)] = "DEAL"
		for i in xrange(32):
			for j in xrange (3):
				for k in xrange (1,11):
					p1 = random.uniform(0,1)
					if (p1 < 0.5):
						self.policy[State(i, j, k)] = "HIT"
					else:
						self.policy[State(i, j, k)] = "STICK"

	def lambda_update (self, episode, online=True, alpha=0.1, la=0.5):

		if (len(episode) != 0):
			max_target = 0.0
			factor1 = 1
			factor2 = 1 - la
			for i in xrange(len(episode)):
				item = episode[i]
				max_target += factor1 * episode[i][2]
				factor1 *= la
				if (i != 0):
					max_target += factor2 * self.qsa_map[(item[0], item[1])][0]
					factor2 *= la

			factor2 = 1 - la
			for i in xrange(len(episode)):
				item = episode[i]
				if (i != 0):
					max_target -= episode[i-1][2]
					max_target -= factor2 * self.qsa_map[(item[0], item[1])][0]
					max_target /= la
				tup = self.qsa_map[(item[0], item[1])]
				self.qsa_map[(item[0], item[1])] = (tup[0] + alpha*(max_target - tup[0]), tup[1] + 1)

				other_action = "HIT"
				if (item[1] == "HIT"):
					other_action = "STICK"
				if (self.qsa_map[(item[0], item[1])][0] > self.qsa_map[(item[0], other_action)][0]):
					self.policy[item[0]] = item[1]
				else:
					self.policy[item[0]] = other_action

	def update_policy (self, episode, online=False, k=1, alpha=0.1):
		i = len(episode) - k - 1
		s1 = episode[i][0]
		a1 = episode[i][1]
		r = episode[i][2]
		s2 = episode[i+k][0]
		a2 = episode[i+k][1]

		tup1 = self.qsa_map[(s1, a1)]
		tup2 = self.qsa_map[(s2, a2)]
		
		if (online):
			self.qsa_map[(s1, a1)] = (tup1[0] + alpha*(tup2[0] - tup1[0]), tup1[1] + 1)
		else:
			other_action2 = "HIT"
			if (a2 == "HIT"):
				other_action2 = "STICK"
			other_q = self.qsa_map[(s2, other_action2)][0]
			self.qsa_map[(s1, a1)] = (tup1[0] + alpha*(max(tup2[0], other_q) - tup1[0]), tup1[1] + 1)

		other_action1 = "HIT"
		if (a1 == "HIT"):
			other_action1 = "STICK"

		if (self.qsa_map[(s1,a1)][0] > self.qsa_map[(s1, other_action1)][0]):
			self.policy[s1] = a1
		else:
			self.policy[s1] = other_action1

	def update_policy_full (self, episode, online=False, k=1, alpha=0.1):
		if (len(episode) != 0):
			r = episode[-1][2]
			for i in xrange (max(0, len(episode) - k), len(episode)):
				s1 = episode[i][0]
				a1 = episode[i][1]
				tup1 = self.qsa_map[(s1, a1)]
				self.qsa_map[(s1, a1)] = (tup1[0] + alpha*(r - tup1[0]), tup1[1] + 1)
				other_action1 = "HIT"
				if (a1 == "HIT"):
					other_action1 = "STICK"

				if (self.qsa_map[(s1, a1)][0] > self.qsa_map[(s1, other_action1)][0]):
					self.policy[s1] = a1
				else:
					self.policy[s1] = other_action1

	def test_episode(self, episodes=100):
		av_reward = 0.0
		for i in xrange(episodes):
			curr_state = self.wrap.start_game()
			EoE =False
			curr_state, reward, EoE = self.wrap.next_state_reward(curr_state, "DEAL")
			while(not EoE):
				curr_state, reward, EoE = self.wrap.next_state_reward(curr_state, self.policy[curr_state])
			av_reward += reward
		av_reward /= episodes
		return av_reward


	def play_episode (self, policy_25=False, control=False, online=False, td_lambda=False, decay=False, k=1, episodes=1, alpha=0.1, reward_av_num=10000):
		if (not control):
			episode = []
			next_state = self.wrap.start_game()
			EoE = False
			while (not EoE):
				curr_state = next_state
				if (policy_25):
					curr_action = self.policy_25(curr_state)
				else:
					# print curr_state
					curr_action = self.policy[curr_state]
				next_state, reward, EoE = self.wrap.next_state_reward(curr_state,curr_action)
				episode.append((curr_state, curr_action, reward))
			return episode
		else:
			rew = []
			if (not td_lambda):
				epsilon = 0.1
				updates = 1
				av_reward = 0.0
				for i in xrange(episodes):
					episode = []
					curr_state = self.wrap.start_game()
					curr_action = self.greedy_policy(curr_state)
					EoE = False
					next_state, reward, EoE = self.wrap.next_state_reward(curr_state, curr_action)
					epsilon_i = epsilon/updates
					if (not EoE):
						next_action = self.greedy_policy(next_state, epsilon=epsilon_i)
					episode_length = 0
					while(not EoE):
						curr_state = next_state
						curr_action = next_action
						next_state, reward, EoE = self.wrap.next_state_reward(curr_state,curr_action)
						epsilon_i = epsilon / updates
						next_action = self.greedy_policy(curr_state, epsilon_i)
						episode.append((curr_state, curr_action, reward))
						episode_length += 1
						if (episode_length >= k+1):
							self.update_policy(episode, online=online, k=k, alpha=alpha)
						if (decay):
							updates += 1
					self.update_policy_full(episode, online=online, k=k, alpha=alpha)
					av_reward += reward
					if (i%reward_av_num == 0):
						rew.append(av_reward / reward_av_num)
						av_reward = 0.0
				return rew
			else:
				epsilon = 0.1
				updates = 1
				av_reward = 0.0
				for i in xrange(episodes):
					episode = []
					curr_state = self.wrap.start_game()
					curr_action = self.greedy_policy(curr_state)
					EoE = False
					next_state, reward, EoE = self.wrap.next_state_reward(curr_state, curr_action)
					epsilon_i = epsilon/updates
					if (not EoE):
						next_action = self.greedy_policy(next_state, epsilon=epsilon_i)
					episode_length = 0
					while(not EoE):
						curr_state = next_state
						curr_action = next_action
						next_state, reward, EoE = self.wrap.next_state_reward(curr_state,curr_action)
						epsilon_i = epsilon / updates
						next_action = self.greedy_policy(curr_state, epsilon_i)
						episode.append((curr_state, curr_action, reward))
						episode_length += 1
						if (decay):
							updates += 1
					self.lambda_update(episode, online=True, alpha=alpha, la=0.5)
					av_reward += reward
					if (i%reward_av_num == 0):
						rew.append(av_reward / reward_av_num)
						av_reward = 0.0
				return rew


 
def print_qsa (agent):
	for action in ["HIT", "STICK"]:
		for i in xrange(32):
			for j in xrange(3):
				for k in xrange(1, 11):
					if (State(i, j, k), action) in agent.qsa_map:
						print str(i) + " | " + str(j) + " | " + str(k) + " | " + action + "\t ----------- " + str(agent.qsa_map[(State(i, j, k), action)][0]) + " | " + str(agent.qsa_map[(State(i, j, k), action)][1])
					else:
						print str(i) + " | " + str(j) + " | " + str(k) + " | " + action + "\t ----------- Not found ----------------------------------------"

def print_policy (agent):
	for i in xrange(32):
		for j in xrange(3):
			for k in xrange(1, 11):
				if (State(i, j, k) in agent.policy):
					print str(i) + " | " + str(j) + " | " + str(k) + "\t ----------- " + agent.policy[State(i,j,k)]
				else:
					print str(i) + " | " + str(j) + " | " + str(k) + "\t ----------- Not found ----------------------------------------"

def print_episode (episode):
	for i in episode:
		print str(i[0]) + " | " + i[1] + " ----------- " + str(i[2])

def print_episode_element (item):
	
	print str(item[0]) + " | " + item[1] + " ----------- " + str(item[2])

def print_dict (d):
	for i in d:
		print "State: ", str(i[0]), " Action: ", i[1], " --------- ", str(d[i])
	print

def average_runs (agent_master, agent):
	for i in agent_master.qsa_map:
		elem = agent_master.qsa_map[i]
		agent_master.qsa_map[i] = ((elem[0]*elem[1] + agent.qsa_map[i][0]) / (elem[1] + 1), elem[1]+1)

def plot_qsa (agent, file_path, figure_title):
	fig = plt.figure()
	fig.suptitle(figure_title)
	for j in [0, 1, 2]:
		s = "22" + str(j+1)
		ax = fig.add_subplot(int(s), projection='3d')
		X = np.array(range(0,32))
		Y = np.array(range(1,11))
		Z = np.zeros((10,32), dtype="float64")
		for i in xrange(31):
			for k in xrange(10):
				Z[k][i] = max(agent.qsa_map[State(X[i], j, Y[k]), "HIT"][0], agent.qsa_map[State(X[i], j, Y[k]), "STICK"][0])
		X,Y = np.meshgrid(X,Y)
		ax.plot_wireframe(X,Y,Z, rstride=1, cstride=1)
		ax.set_xlabel('Player hand value')
		ax.set_ylabel('Dealer hand value')
		ax.set_zlabel('q(s,a) value')
		ax.view_init(45, 260)
		if (j == 0):
			ax.set_title("Hand value - No special card")
		elif (j == 1):
			ax.set_title("Hand value - Special card (Used at lower value)")
		elif (j == 2):
			ax.set_title("Hand value - Special card (Used at higher value)")
	# plt.show()
	fig = plt.gcf()
	fig.set_size_inches(19.20,10.80)
	plt.savefig(file_path,dpi=80)

def plot_avg_reward(ax, reward_avg, interval=1000, linestyle='-', legend='legend', marker=','):
	y = np.array(reward_avg)
	n = len(reward_avg)
	x = []
	for i in xrange(n):
		x.append(interval*i)
	x = np.array(x)
	line, = ax.plot(x, y, linestyle=linestyle, marker=marker)
	line.set_label(legend)
	ax.set_xlabel('Episodes')
	ax.set_ylabel('Average Reward over previous ' + str(interval) + ' episodes')
	

parser = argparse.ArgumentParser()
parser.add_argument('-m', action="store_true", dest="monte_carlo_eval", help="execute Monte Carlo evaluation of policy_25")
parser.add_argument('-t', action="store_true", dest="td_eval", help="execute TD evaluation of policy_25")
parser.add_argument('-r', action="store_true", dest="avg_training_reward", help="execute average training reward vs epochs experiment")
parser.add_argument('-e', action="store_true", dest="avg_test_reward", help="execute average test reward vs alpha experiment")
parser.add_argument('-l', action="store_true", dest="lambda_policy", help="execute td lambda policy evaluation")

results = parser.parse_args()
monte_carlo_eval = results.monte_carlo_eval
td_eval = results.td_eval
avg_training_reward = results.avg_training_reward
avg_test_reward= results.avg_test_reward
lambda_policy = results.lambda_policy


############################################### M O N T E   C A R L O ###########################################
def monte_carlo_eval (agent, episode, first_visit=True):
	if (first_visit):
		states = {}
		for item in episode:
			if ((item[0], item[1]) not in states):
				states[(item[0], item[1])] = 0
			else:
				states[(item[0], item[1])] += 1

		_return = 0.0
		for i in xrange(len(episode) - 1, -1, -1):
			state = episode[i][0]
			action = episode[i][1]
			reward = float(episode[i][2])
			_return += reward
			if (states[(state, action)] == 0):
				tup = agent.qsa_map[(state, action)]
				agent.qsa_map[(state, action)] = ((tup[0] * tup[1] + _return) / (tup[1] + 1) , tup[1] + 1)
			else:
				states[(state, action)] -= 1
	else:
		_return = 0.0
		for i in xrange(len(episode) - 1, -1, -1):
			state = episode[i][0]
			action = episode[i][1]
			reward = float(episode[i][2])
			_return += reward
			tup = agent.qsa_map[(state, action)]
			agent.qsa_map[(state, action)] = ((tup[0] * tup[1] + _return) / (tup[1] + 1) , tup[1] + 1)

if (monte_carlo_eval):
	print "Running MC evaluation of policy 25"
	EPISODES = 1000
	for visit in [True, False]:
		sub_folder = 'every_visit'
		if (visit):
			sub_folder = 'first_visit'
		for runs in [1, 10, 100, 1000]:
			agent_master = Agent()
			agent_master.init_qsa()
			for i in xrange(runs):
				agent = Agent()
				agent.init_qsa()
				for ep in xrange(EPISODES):
					episode = agent.play_episode(policy_25=True, control=False, online=False)
					monte_carlo_eval (agent, episode, visit)
				average_runs(agent_master, agent)
			plot_qsa(agent_master, './graphs/Part1/monte_carlo/' + sub_folder + '/' + str(runs), 'Monte carlo ' + sub_folder + ' averaged over ' + str(runs) + ' runs')


###################################################### T D (N) ##################################################
def td_eval (agent, episode, n):
	alpha = 0.1
	_return = 0.0
	window_size = 0
	cum_reward = 0.0
	for i in xrange(len(episode) - 1, -1, -1):
		qs_next = 0.0
		state = episode[i][0]
		action = episode[i][1]
		reward = float(episode[i][2])
		cum_reward += reward
		_return = 0.0
		if (window_size == n):
			cum_reward -= float(episode[i+n][2])
			if (episode[i+n][0], episode[i+n][1]) in agent.qsa_map:
				qs_next = agent.qsa_map[(episode[i+n][0], episode[i+n][1])][0]
			else:
				agent.qsa_map[(episode[i+n][0], episode[i+n][1])] = (0, 0)
		else:
			window_size += 1
		_return = cum_reward + qs_next
		tup = agent.qsa_map[(state, action)]
		agent.qsa_map[(state, action)] = (tup[0] + alpha * (_return - tup[0]), tup[1] + 1)

if(td_eval):
	print "Running TD evaluation of policy 25"
	EPISODES = 1000
	for k in [1, 3, 5, 10, 20, 100, 1000]:
		sub_folder = str(k) + '/'
		for runs in [1, 10, 100, 1000]:
			agent_master = Agent()
			agent_master.init_qsa()
			for i in xrange(runs):
				agent = Agent()
				agent.init_qsa()
				for ep in xrange(EPISODES):
					episode = agent.play_episode(policy_25=True, control=False, online=False)
					td_eval (agent, episode, k)
				average_runs(agent_master, agent)
			plot_qsa(agent_master, './graphs/Part1/td/' + sub_folder + str(runs), 'TD(' + str(k) + ') averaged over ' + str(runs) + ' runs')
	

############################################# P A R T 2 #############################################################

# color = ['red', 'green', 'blue', 'cyan']
# marker = ['^', 'o', 'O']


linestyles = ['--', '-.']

if (avg_training_reward):
	print "Running Training reward vs Epochs experiment"
	k_arr = [1, 10, 100, 1000]
	for EPISODES in [1000, 10000, 100000]:
		REWARD_INTERVAL = EPISODES/10
		fig = plt.figure()
		fig.suptitle("TD(K) | TD(K) Decay | Q Learning | TD(Lambda) Reward averaged over " + str(REWARD_INTERVAL) + " episodes")
		ax = fig.add_subplot(111)

		print "Executing SARSA(k) for ", EPISODES, " episodes"
		for k in k_arr:
			for decay in [True, False]:
				linestyle = linestyles[0]
				if(decay):
					linestyle = linestyles[1]
				agent = Agent()
				agent.init_qsa()
				agent.init_policy()
				reward_avg = agent.play_episode(policy_25=False, control=True, online=True, td_lambda=False, episodes=EPISODES, decay=decay, k=k, reward_av_num=REWARD_INTERVAL)
				plot_avg_reward(ax, reward_avg, interval=REWARD_INTERVAL, linestyle=linestyle, legend='TD(' + str(k) + ') with decay=' + str(decay))

		print "Executing Q-Learning for ", EPISODES, " episodes"
		agent = Agent()
		agent.init_qsa()
		agent.init_policy()
		reward_avg = agent.play_episode(policy_25=False, control=True, online=False, td_lambda=False, episodes=EPISODES, decay=False, k=1, reward_av_num=REWARD_INTERVAL)
		plot_avg_reward(ax, reward_avg, interval=REWARD_INTERVAL, linestyle='-', legend='Q Learning', marker='o')

		print "Executing TD Lambda for ", EPISODES, " episodes"
		agent = Agent()
		agent.init_qsa()
		agent.init_policy()
		reward_avg = agent.play_episode(policy_25=False, control=True, online=False, td_lambda=True, episodes=EPISODES, decay=False, k=1, reward_av_num=REWARD_INTERVAL)
		plot_avg_reward(ax, reward_avg, interval=REWARD_INTERVAL, linestyle='-', legend='TD(Lambda)', marker='^')

		ax.legend()
		fig = plt.gcf()
		fig.set_size_inches(19.20,10.80)
		plt.savefig('./graphs/Part2/' + str(REWARD_INTERVAL),dpi=80)


######################################################## P A R T 3 ################################################################

linestyles = ['--', '-.']

if (avg_test_reward):
	EPISODES = 100000
	print "Running Test reward vs Alpha experiment for ", EPISODES, " episodes"
	fig = plt.figure()
	fig.suptitle("TD(K) | TD(K) Decay | Q Learning | TD(Lambda) Average reward after training for " + str(EPISODES) + " episodes")
	ax = fig.add_subplot(111)

	alpha_arr = [0.1, 0.2, 0.3, 0.4]

	print "Executing SARSA-k"
	k_arr = [1, 10, 100, 1000]
	for k in k_arr:
		for decay in [True, False]:
			linestyle = linestyles[0]
			if(decay):
				linestyle = linestyles[1]
			reward_arr = []
			for alpha in alpha_arr:
				agent = Agent()
				agent.init_qsa()
				agent.init_policy()
				agent.play_episode(policy_25=False, control=True, online=True, td_lambda=False, episodes=EPISODES, decay=decay, alpha=alpha, k=k)
				reward_avg = agent.test_episode(episodes=100)
				reward_arr.append(reward_avg)
			line, = ax.plot(alpha_arr, reward_arr, linestyle=linestyle)
			line.set_label('TD(' + str(k) + ') with decay=' + str(decay))

	print "Executing Q-Learning"
	reward_arr = []
	for alpha in alpha_arr:
		agent = Agent()
		agent.init_qsa()
		agent.init_policy()
		agent.play_episode(policy_25=False, control=True, online=False, td_lambda=False, episodes=EPISODES, decay=False, alpha=alpha, k=1)
		reward_avg = agent.test_episode(episodes=100)
		reward_arr.append(reward_avg)
	line, = ax.plot(alpha_arr, reward_arr, linestyle='-', marker='o')
	line.set_label('Q Learning')

	print "Executing TD Lambda"
	reward_arr = []
	for alpha in alpha_arr:
		agent = Agent()
		agent.init_qsa()
		agent.init_policy()
		agent.play_episode(policy_25=False, control=True, online=False, td_lambda=True, episodes=EPISODES, decay=False, alpha=alpha, k=1)
		reward_avg = agent.test_episode(episodes=100)
		reward_arr.append(reward_avg)
	line, = ax.plot(alpha_arr, reward_arr, linestyle='-', marker='^')
	line.set_label('TD(Lambda)')

	ax.legend()
	ax.set_xlabel('Learning Rate')
	ax.set_ylabel('Average reward over 100 test runs after training')
	fig = plt.gcf()
	fig.set_size_inches(19.20,10.80)
	# plt.savefig('./graphs/Part3/' + str(100),dpi=80)


##################################################### P A R T 4 ###########################################################

if (lambda_policy):
	EPISODES = 1000000
	print ("Training TD Lambda for ", EPISODES, " episodes")
	agent = Agent()
	agent.init_qsa()
	agent.init_policy()
	agent.play_episode(policy_25=False, control=True, online=False, td_lambda=False, episodes=EPISODES, decay=False, alpha=0.1, k=1)

	print ("Evaluating trained algorithm using Monte Carlo")
	agent.init_qsa()
	EPISODES = 1000000
	agent.policy[State(-1, -1, -1)] = "DEAL"
	for ep in xrange(EPISODES):
		episode = agent.play_episode(policy_25=False, control=False, online=False)
		monte_carlo_eval (agent, episode, False)
	plot_qsa(agent, './graphs/Part4/eval', 'TD(Lambda) trained for 1 Million episodes')
