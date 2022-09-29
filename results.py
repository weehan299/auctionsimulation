from typing import Tuple
from attrs import define, field, validators

import numpy as np
from tabulate import tabulate

from auctionenvironment import AuctionEnvironment


@define
class Results:

    env: AuctionEnvironment = field(factory=AuctionEnvironment)
    competitive_profits: np.array = field(init=False)
    last_bid: np.array = field(init=False)
    average_bid: np.array = field(init=False)
    average_payoff: np.array = field(init=False)

    bid_history: list = field(factory=list)
    payoff_history: list = field(factory=list)


    def __attrs_post_init__(self):
        self.last_bid = np.array(self.env.bid_history)[-1]
        self.average_bid = np.array(self.env.bid_history)[-2500:].mean(axis=0)
        self.average_payoff = np.array(self.env.payoff_history)[-2500:].mean(axis=0)
        #self.competitive_profits = self.competitive_profits_compute()
        self.bid_history = self.env.bid_history
        self.payoff_history = self.env.payoff_history

    def print_results(self):
        
        name = [agent.get_name() for agent in self.env.agents]
        desc = [agent.get_parameters() for agent in self.env.agents]

        print("alpha = ", self.env.alpha, " (1 is FPA, 2 is SPA) ")
        print(tabulate({"Name": name,
                    "Last_bid": self.last_bid,
                    #"Bertrand-Nash Price": self.env.competitive_prices_array,
                    "Average Bid (last 2500 bids)": self.average_bid,
                    #"Bertrand-Nash Profit": self.competitive_profits,
                    "Average Payoff": self.average_payoff,
                }, 
                headers="keys"))
            
        print(tabulate({"Name":name, "Description":desc },headers="keys"))
        print("\n")