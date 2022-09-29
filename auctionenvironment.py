from attrs import define, field, validators
from typing import List
import numpy as np
import random
from tqdm import tqdm
from bidcalculator import BidCalculator

from validators import validate_action_space_num, validate_total_periods
from agent import Agent

#convex combination alpha, alpha = 1: FPA, alpha=2, SPA
#m: number of intervals in actions space
#bid design to calculate payoffs

@define
class AuctionEnvironment:
    
    agents: List[Agent] = field(factory=list)

    action_space: np.array = field(init=False)
    alpha: float  = field(default = 1) #alpha = 1: FPA, 2:SPA

    bid_calculator: BidCalculator = field(factory=BidCalculator)

    action_space_num: int = field(default = 19, validator=[validators.instance_of(int), validate_action_space_num] )
    total_periods: int = field(default=1000000, validator=[validators.instance_of(int), validate_total_periods])

    tstable:int = 10000
    tscore:int = 0
    

    bid_history: list = field(factory=list)
    payoff_history: list = field(factory=list)

    def __attrs_post_init__(self):
        self.action_space = self.init_action_space(0.05,0.95, self.action_space_num)
        self.bid_calculator = BidCalculator(alpha = self.alpha)

    @staticmethod
    def init_action_space(low:float, high: float, step: int) -> np.array:
        return np.linspace(low , high, step+1)
        
    def run_simulation(self):
        prev_state = random.choices(self.action_space, k=len(self.agents))
        curr_state = np.array(
            [agent.pick_strategy(prev_state, self.action_space, 0) for agent in self.agents]
        )
        prev_reward_array = self.bid_calculator.calculate_payoff(curr_state)

        for t in tqdm(range(self.total_periods)):
            
            next_state = np.array(
                [agent.pick_strategy(curr_state, self.action_space, t) for agent in self.agents]
            )

            if self.tscore == self.tstable:
                break


            reward_array = self.bid_calculator.calculate_payoff(next_state)

            for agent, action, prev_action, reward, prev_reward in zip(
                self.agents, next_state, curr_state, reward_array, prev_reward_array):

                agent.learn(
                    old_state = prev_state,
                    curr_state= curr_state,
                    new_state= next_state,
                    action_space= self.action_space,
                    prev_reward = prev_reward,
                    reward= reward,
                    prev_action = prev_action,
                    action = action,
                        )


            prev_state = curr_state
            curr_state = next_state
            prev_reward_array = reward_array
            

            self.bid_history.append(prev_state)
            self.payoff_history.append(reward_array)

            #check for convergence
            self.check_stable()


        
    def run_simulation_dont_provide_other_players_info(self):

        prev_state_of_all_agents = random.choices(self.action_space, k=len(self.agents))
        curr_state_of_all_agents = np.array(
            [self.agents[i].pick_strategy(np.array([prev_state_of_all_agents[i]]), self.action_space, 0) for i in range(len(self.agents))]
        )

        prev_reward_array = self.bid_calculator.calculate_payoff(curr_state_of_all_agents)

        for t in tqdm(range(self.total_periods)):
            
            next_state_of_all_agents = np.array(
                [self.agents[i].pick_strategy(np.array([curr_state_of_all_agents[i]]), self.action_space, t) for i in range(len(self.agents))]
            )

            if self.tscore == self.tstable:
                break


            reward_array = self.bid_calculator.calculate_payoff(next_state_of_all_agents)

            for agent, prev_state, next_state, curr_state, reward, prev_reward in zip(
                self.agents, prev_state_of_all_agents, next_state_of_all_agents, curr_state_of_all_agents, reward_array, prev_reward_array):

                agent.learn(
                    old_state = np.array([prev_state]),
                    curr_state = np.array([curr_state]),
                    new_state = np.array([next_state]),
                    action_space= self.action_space,
                    prev_reward = prev_reward,
                    reward= reward,
                    prev_action = curr_state,
                    action = next_state,
                        )


            prev_state_of_all_agents = curr_state_of_all_agents
            curr_state_of_all_agents = next_state_of_all_agents
            prev_reward_array = reward_array
            

            self.bid_history.append(prev_state_of_all_agents)
            self.payoff_history.append(reward_array)

            #check for convergence
            self.check_stable()
                
    def check_stable(self):
            #print([agent.stable_status == 1 for agent in self.agents])
            if False in [agent.stable_status == 1 for agent in self.agents]:
                self.tscore = 0
            else:
                self.tscore += 1


    def check_stable2(self):
            if False in [abs(agent.old_action_value - agent.curr_action_value) < 1e-6 for agent in self.agents]:
                self.tscore = 0
            else:
                self.tscore += 1

