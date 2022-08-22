from attrs import define, field, validators
from typing import List
import numpy as np
import random
from tqdm import tqdm

@define
class BidCalculator:
    alpha: float = field(default = 1)
    def calculate_payoff(self, bid_array:np.array):
        payoff_array = np.zeros(len(bid_array))
        winning_index = random.choice(np.where(bid_array==max(bid_array))[0])
        #index = np.argmax(bid_array)
        payoff_array[winning_index] = 1
        for i in range(len(bid_array)):
            if i == winning_index: 
                payoff_array[winning_index] -=  (2-self.alpha) * bid_array[i]
            else:
                payoff_array[winning_index] -=  (self.alpha - 1) * bid_array[i]
        #print(payoff_array)
        return payoff_array

    def divisible_object_calculate_payoff(self, bid_array:np.array):
        payoff_array = np.zeros(len(bid_array))
        winning_indices = np.where(bid_array==max(bid_array))[0]
        payoff_array[winning_indices] = 1/len(bid_array)
        for winning_index in winning_indices:
            for i in range(len(bid_array)):
                if i ==  winning_index:
                    payoff_array[winning_index] -=  (2-self.alpha) * bid_array[i]
                else:
                    payoff_array[winning_index] -=  (self.alpha-1) * bid_array[i]
        return payoff_array
