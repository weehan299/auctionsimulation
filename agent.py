
from attrs import define, field, validators
import abc
import numpy as np
import itertools
import random
from typing import Dict,Tuple
import copy
import sys

from policy import TimeDecliningExploration,Policy
from validators import validate_learning_rate, validate_gamma 

@define
class Agent(metaclass=abc.ABCMeta):

    @abc.abstractclassmethod
    def pick_strategy(self, state:np.array, action_space:np.array, t:int):
        raise NotImplementedError
        
    @abc.abstractclassmethod
    def learn(
        self, 
        old_state: np.array,
        curr_state:np.array,
        new_state: np.array,
        action_space:np.array,
        prev_reward:float,
        reward: float,
        prev_action: float,
        action:float
            ):
     
        raise NotImplementedError

    def get_name(self):
        return type(self).__name__

        
@define
class ConstantBidder(Agent):
    stable_status: bool = field(default=True)
    constant_bid: bool=field(init=False)
    def learn(
        self, 
        old_state: np.array,
        curr_state:np.array,
        new_state: np.array,
        action_space:np.array,
        prev_reward:float,
        reward: float,
        prev_action: float,
        action:float
            ):
        pass

    def pick_strategy(self, state: np.array, action_space: np.array, t:int) -> float:
        self.constant_bid = action_space[len(action_space)//2]
        return self.constant_bid
    
    def get_parameters(self) -> str:
        return "Constant bid at: {}".format(round(self.constant_bid,5))


    
    
@define
class QLearning(Agent):

    """ Q learning that stores Q in dictionary form"""

    Q: Dict = field(default=None)
    policy: Policy = field(factory = TimeDecliningExploration)

    old_action_value: float = field(init=False)
    curr_action_value: float= field(init=False)

    learning_rate: float = field(default = 0.05,validator=[validators.instance_of(float), validate_learning_rate]) #learning rate
    gamma: float = field(default = 0.99,validator=[validators.instance_of(float), validate_gamma]) # discount rate
    
    stable_status: bool = field(default=False)


    def pick_strategy(self, state: np.array, action_space: np.array, t:int) -> float:
        if not self.Q:
            self.Q = self.initQ(len(state), action_space)

        Q_value_array =list(self.Q[tuple(state)].values())
        prob_weights = self.policy.give_prob_weights_for_each_action(Q_value_array,t)
        return random.choices(action_space,weights=prob_weights,k=1)[0]

    
    def learn(
        self, 
        old_state: np.array,
        curr_state:np.array,
        new_state: np.array,
        action_space:np.array,
        prev_reward:float,
        reward: float,
        prev_action: float,
        action:float
            ):

        old_action_value_array = copy.deepcopy(list(self.Q[tuple(curr_state)].values()))
        old_action_value = copy.deepcopy(self.Q[tuple(curr_state)][action])
        new_action_value = self.Q[tuple(new_state)][self.get_argmax(new_state)]
        self.Q[tuple(curr_state)][action] = (1-self.learning_rate) * old_action_value +  self.learning_rate * (reward + self.gamma * new_action_value )

        self.old_action_value = old_action_value
        self.curr_action_value = self.Q[tuple(curr_state)][action]
        
        #check convergence
        new_action_value_array = list(self.Q[tuple(curr_state)].values())
        #self.stable_status = 1 if all(np.absolute(np.subtract(old_action_value_array , new_action_value_array)) < 1e-6) else 0
        #self.stable_status = 1 if abs(old_action_value - self.Q[tuple(curr_state)][action]) < 1e-6 else 0
        #print(old_action_value_array, new_action_value_array, self.old_action_value, self.curr_action_value)

        self.stable_status = (np.argmax(old_action_value_array) == np.argmax(new_action_value_array))

    
    def get_argmax(self, state: np.array) -> float:
        highest_qvalue_actions = [
            action for action, value in self.Q[tuple(state)].items() if value == max(self.Q[tuple(state)].values())
        ]
        return random.choice(highest_qvalue_actions) 
    
    
    def get_parameters(self) -> str:
        return ":  learning_rate={}, gamma={}, policy = {} " .format(
             self.learning_rate, self.gamma, self.policy.get_name()
        )

    @staticmethod
    def initQ(num_agents:int, action_space:np.array) -> Dict:
        Q = {}
        for state in itertools.product(action_space, repeat=num_agents):
            Q[state] = dict((price,40) for price in action_space)  #35 good to ensure exploration
        return Q
    

@define
class QLearningWithMemory(Agent):

    """ Q learning that stores Q in dictionary form"""
    # a agent state is the set of all past prices. 

    Q: Dict = field(default=None)
    policy: Policy = field(factory = TimeDecliningExploration)
    memory_length: int = field(default = 2)
    memory: list = field(default = None)

    old_action_value: float = field(init=False)
    curr_action_value: float= field(init=False)

    learning_rate: float = field(default = 0.1,validator=[validators.instance_of(float), validate_learning_rate]) #learning rate
    gamma: float = field(default = 0.95,validator=[validators.instance_of(float), validate_gamma]) # discount rate
    
    stable_status: bool = field(default=False)


    def pick_strategy(self, state: np.array, action_space: np.array, t:int) -> float:
        if not self.Q:
            self.memory = self.init_memory(len(state), self.memory_length,action_space)
            self.Q = self.init_Q(len(state), self.memory_length, action_space)
        
        state_with_memory = self.append_memory_to_state(state) 
        
        Q_value_array =list(self.Q[tuple(state_with_memory)].values())
        prob_weights = self.policy.give_prob_weights_for_each_action(Q_value_array,t)
        return random.choices(action_space,weights=prob_weights,k=1)[0]

    
    def learn(
        self, 
        old_state: np.array,
        curr_state:np.array,
        new_state: np.array,
        action_space:np.array,
        prev_reward:float,
        reward: float,
        prev_action: float,
        action:float
            ):


        #print("old memory", self.memory)
        self.update_memory(old_state)
        #print("new memory", self.memory)
        #print("curr state: ", curr_state, "new state:", new_state)
        curr_state_with_memory = self.append_memory_to_state(curr_state)
        #print("curr state with memory:",curr_state_with_memory)
        new_state_with_memory = np.concatenate([new_state,np.array(curr_state_with_memory[:self.memory_length])])
        #print("new state with memory:",new_state_with_memory)

    
        old_action_value_array = copy.deepcopy(list(self.Q[tuple(curr_state_with_memory)].values()))
        old_action_value = copy.deepcopy(self.Q[tuple(curr_state_with_memory)][action])
        new_action_value = self.Q[tuple(new_state_with_memory)][self.get_argmax(new_state_with_memory)]
        self.Q[tuple(curr_state_with_memory)][action] = (1-self.learning_rate) * old_action_value +  self.learning_rate * (reward + self.gamma * new_action_value )

        self.old_action_value = old_action_value
        self.curr_action_value = self.Q[tuple(curr_state_with_memory)][action]
        
        #check convergence
        new_action_value_array = list(self.Q[tuple(curr_state_with_memory)].values())
        #self.stable_status = 1 if all(np.absolute(np.subtract(old_action_value_array , new_action_value_array)) < 1e-6) else 0
        #self.stable_status = 1 if abs(old_action_value - self.Q[tuple(curr_state)][action]) < 1e-6 else 0
        self.stable_status = (np.argmax(old_action_value_array) == np.argmax(new_action_value_array))

    def append_memory_to_state(self, state:np.array) -> np.array:
        temp = copy.deepcopy(self.memory.flatten())
        result = np.concatenate((state,temp))
        return result
        
    def update_memory(self, curr_state:np.array):
        self.memory[1:] = self.memory[:-1]
        self.memory[0] = curr_state

    def get_argmax(self, state: np.array) -> float:
        optimal_actions = [
            action for action, value in self.Q[tuple(state)].items() if value == max(self.Q[tuple(state)].values())
        ]
        return random.choice(optimal_actions) 
    
    

    def get_parameters(self) -> str:
        return ":  learning_rate={}, gamma={}, policy = {} " .format(
             self.learning_rate, self.gamma, self.policy.get_name()
        )

    @staticmethod
    def init_Q(num_agents:int, memory_length:int, action_space:np.array) -> Dict:
        Q = {}
        for state in itertools.product(action_space, repeat=num_agents*(memory_length+1)):
            Q[state] = dict((price,0) for price in action_space)
        return Q

    @staticmethod
    def init_memory(num_agents:int, memory_length:int,action_space:np.array) -> np.array:
        #initialise with random memory from the start
        memory = np.array([random.choices(action_space, k=num_agents) for i in range(memory_length)]).flatten()
        return memory
    
    
    
    
@define
class TitforTat(Agent):
    #only can be used when agent knows the state of other agents. 
    # tit for tat agent placed as the second player. 
    stable_status: bool = field(default=True)
    max_price: float = field(default = 0)
    def learn(
        self, 
        old_state: np.array,
        curr_state:np.array,
        new_state: np.array,
        action_space:np.array,
        prev_reward:float,
        reward: float,
        prev_action: float,
        action:float
            ):
        pass

    def pick_strategy(self, state: np.array, action_space: np.array, t:int) -> float:
        if not self.max_price:
            self.max_price = max(action_space)

        if state[0] <= state[1]:
            return state[0]
        else:
            return self.max_price
    
    def get_parameters(self) -> str:
        return "Tit for Tat Price at: {}".format(round(self.max_price,5))