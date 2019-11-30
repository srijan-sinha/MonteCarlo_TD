import numpy as np
import math
import random
from simulator import Simulator
from simulator import State

######################################################################################
######################################################################################
## 																					##
## Player Hand Value --> -1, [0 - 31]												##
## Dealer States --> -1, [1 - 10]													##
## Value Type --> (Hard No Sp | Hard Sp | Soft Sp) - (0 | 1 | 2)					##
## Total States = (Hand Value x Value Type x Dealer State) = 32 * 3 * 11 = 1056		##
## State = (Hand Value, Value Type, Dealer State)									##
##																					##
## Action --> 
## 																					##
######################################################################################
######################################################################################

class Wrapper:
	def __init__(self, p=2.0/3):
		self.sim = Simulator(p)
		self.start_state = State (-1, -1, -1)
	
	def start_game (self):
		return self.start_state

	def next_state_reward (self, curr_state, action):
		if (curr_state == self.start_state):
			player_card = self.sim.get_card()
			dealer_card = self.sim.get_card()
			# player_card = 8
			# dealer_card = 7
			# print "Initial cards: " + str(player_card) + " " + str(dealer_card)
			if (player_card < 0 and dealer_card < 0):
				next_state = State (-1, 0, -1)
				reward = 0
				EoE = True
			
			elif (player_card > 0 and dealer_card < 0):
				hand_value = player_card
				value_type = 0
				if (player_card >= 1 and player_card <= 3):
					hand_value += 10
					value_type = 2
				next_state = State (hand_value, value_type, -1)
				reward = 100
				EoE = True

			elif (player_card < 0 and dealer_card > 0):
				next_state = State (-1, 0, dealer_card)
				reward = -100
				EoE = True

			else:
				hand_value = player_card
				value_type = 0
				if (player_card >= 1 and player_card <= 3):
					hand_value += 10
					value_type = 2
				next_state = State (hand_value, value_type, dealer_card)
				reward = 0
				EoE = False

		else:
			if (action == "HIT"):
				# print "Hitting..."
				next_state = self.sim.hit_update(curr_state)
				if (next_state.hand_value != -1):
					reward = 0
					EoE = False
				else:
					reward = -100
					EoE = True
			else :
				next_state = curr_state
				reward = self.sim.get_reward_stick(curr_state)
				EoE = True
		# print "Next state: " + str(next_state)
		# print "Reward: " + str(reward)
		# print "EoE: " + str(EoE)
		return (next_state, reward, EoE)

