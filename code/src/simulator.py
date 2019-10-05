import numpy as np
import math
import random

######################################################################################
######################################################################################
## 																					##
## Player Hand Value --> -1, [0 - 31]												##
## Dealer States --> -1, [1 - 10]													##
## Value Type --> (Hard No Sp | Hard Sp | Soft Sp) - (0 | 1 | 2)					##
## Total States = (Hand Value x Value Type x Dealer State) = 32 * 3 * 11 = 1056		##
## State = (Hand Value, Value Type, Dealer State)									##
## 																					##
######################################################################################
######################################################################################

class BustError(Exception):
	def __init__(self, value):
		self.value = value
	def __str__(self):
		return repr(self.value)

class State:
	def __init__(self, hand_value=0, value_type=0, dealer=-1):
		self.hand_value = hand_value
		self.value_type = value_type
		self.dealer = dealer

	def __str__(self):
		s = "("
		s += str(self.hand_value)
		s += " | "
		s += str(self.value_type)
		s += " | "
		s += str (self.dealer)
		s += ")"
		return s

	def __eq__(self, other):
		if not isinstance(other, State):
			return NotImplemented
		else:
			return self.hand_value == other.hand_value and self.value_type == other.value_type and self.dealer == other.dealer

	# def __lt__(self, other):
	# 	if not isinstance(other, State):
	# 		return NotImplemented
	# 	else:
	# 		if (self.hand_value < other.hand_value):
	# 			return True
	# 		elif (self.hand_value == other.hand_value):
			


	def __hash__(self):
		return hash(tuple([self.hand_value, self.value_type, self.dealer]))

class Simulator:	
	def __init__ (self, black=float(2.0)/3):
		self.gen = Generator(black)
	
	def get_card (self):
		card = self.gen.get_card()
		return self.gen.get_card()

	def transition (self, hand_value_old, value_type_old, next_card):
		hand_value = 0
		value_type = 0
		if (hand_value_old == -1):
			raise BustError("Hit on a player/dealer bust!")
		
		if (next_card >= 1 and next_card <= 3):

			if (value_type_old == 0):
				hand_value = hand_value_old + next_card + 10
				if (hand_value >= 0 and hand_value <= 31):
					value_type = 2
				elif (hand_value > 31):
					hand_value -= 10
					value_type = 1
					if (hand_value > 31):
						hand_value = -1
						value_type = 0
				else:
					print "This case shouldn't happen ever (1)"

			elif (value_type_old == 1):
				hand_value = hand_value_old + next_card
				if (hand_value >= 0 and hand_value <= 31):
					if (hand_value <= 21):
						value_type = 2
						hand_value += 10
					else:
						value_type = 1
				else:
					if (hand_value < 0):
						print "This case shouldn't happen ever (2)"
						hand_value += 10
						value_type = 2
					hand_value = -1
					value_type = 0

			else:
				hand_value = hand_value_old + next_card
				if (hand_value >= 0 and hand_value <= 31):
					value_type = 2
				elif (hand_value > 31):
					hand_value -= 10
					value_type = 1
				else:
					hand_value = -1
					value_type = 0

		else:

			if (value_type_old == 0):
				value_type = 0
				hand_value = hand_value_old + next_card
				if (hand_value < 0 or hand_value > 31):
					hand_value = -1

			elif (value_type_old == 1):
				hand_value = hand_value_old + next_card
				if (hand_value >= 0 and hand_value <= 31):
					if (hand_value <= 21):
						value_type = 2
						hand_value += 10
					else:
						value_type = 1
				else:
					if (hand_value < 0):
						print "This case shouldn't happen ever (3)"
						hand_value += 10
						value_type = 2
					hand_value = -1
					value_type = 0

			else:
				hand_value = hand_value_old + next_card
				if (hand_value >= 0 and hand_value <= 31):
					value_type = 2
				elif (hand_value > 31):
					hand_value -= 10
					value_type = 1
				else:
					hand_value = -1
					value_type = 0
		return (hand_value, value_type)


	def update_state (self, curr_state, next_card):
		hand_value, value_type = self.transition(curr_state.hand_value, curr_state.value_type, next_card)
		dealer = curr_state.dealer
		state = State (hand_value, value_type, curr_state.dealer)

		return state

	def hit_update (self, curr_state):
		next_card = self.get_card()
		# print "Next Card: " + str(next_card)
		return self.update_state(curr_state, next_card)

	def get_reward_stick (self, curr_state):
		if (curr_state.hand_value == -1):
			print "This case shouldn't happen ever (4)"

		if (curr_state.dealer == -1):
			print "This case shouldn't happen ever (5)"			
			return 100
		dealer_hand = curr_state.dealer
		dealer_value_type = 0
		if (dealer_hand >= 1 and dealer_hand <= 3):
			dealer_value_type = 2
			dealer_hand += 10

		while (dealer_hand < 25 and dealer_hand >= 0):
			next_card = self.get_card()
			dealer_hand, dealer_value_type = self.transition(dealer_hand, dealer_value_type, next_card)

		if (dealer_hand == -1):
			return 100
		elif (dealer_hand < curr_state.hand_value):
			return 100
		elif (dealer_hand > curr_state.hand_value):
			return -100
		elif (dealer_hand == curr_state.hand_value):
			return 0;

class Generator:
	def __init__ (self, p=2.0/3):
		self.black_p = p

	def get_card (self):
		colour = int(2*(float(np.random.binomial(1,self.black_p)) - 0.5))
		value = random.sample(range(1,11), 1)
		return colour * value[0]
