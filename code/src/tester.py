from sim_wrap import Wrapper
from simulator import State

wrap = Wrapper()
# curr_state, reward, EoE = wrap.next_state_reward(curr_state, "HIT")

for i in xrange (100):
	curr_state = wrap.start_game()
	EoE = False
	print ("CurrState 		  NextState 		Reward")
	while(not EoE):
		prev_state = curr_state
		curr_state, reward, EoE = wrap.next_state_reward(curr_state,"HIT")
		print "(" + str(prev_state) + ")  --->  (" + str(curr_state) + ")  --->  " + str(reward)
	print
	print


# from simulator import Simulator
# sim = Simulator()
# v,t = sim.transition(26, 2, 3)
# print v
# print t
