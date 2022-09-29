from auctionenvironment import AuctionEnvironment
from agent import QLearning, ConstantBidder, QLearningWithMemory, TitforTat
from policy import Boltzmann, TimeDecliningExploration
from results import Results

def run(*args, **kwargs):
    

    agent1 = QLearning(learning_rate= kwargs.get("learning_rate",0.05), gamma = kwargs.get("gamma", 0.99),
                        #policy=Boltzmann(temp_max=kwargs.get("temp_max",3), temp_min=kwargs.get("temp_min",0.0001),tot_steps = kwargs.get("total_periods", 1000000)))
                        policy=TimeDecliningExploration(beta = kwargs.get("beta", 0.0002)))
    agent2 = QLearning(learning_rate= kwargs.get("learning_rate",0.05), gamma = kwargs.get("gamma", 0.99), 
                        #policy=Boltzmann(temp_max=kwargs.get("temp_max",3),temp_min=kwargs.get("temp_min",0.0001),tot_steps = kwargs.get("total_periods", 1000000)))
                        policy=TimeDecliningExploration(beta = kwargs.get("beta", 0.0002)))
    constant_agent = ConstantBidder()
    titfortat_agent = TitforTat()

    agent3 = QLearningWithMemory(learning_rate= kwargs.get("learning_rate",0.05), gamma = kwargs.get("gamma", 0.99), 
                        #policy=Boltzmann(temp_max=kwargs.get("temp_max",3),temp_min=kwargs.get("temp_min",0.0001),tot_steps = kwargs.get("total_periods", 1000000)))
                        policy=TimeDecliningExploration(beta = kwargs.get("beta", 0.0002)))
    
    agent4 = QLearningWithMemory(learning_rate= kwargs.get("learning_rate",0.05), gamma = kwargs.get("gamma", 0.99), 
                        #policy=Boltzmann(temp_max=kwargs.get("temp_max",3),temp_min=kwargs.get("temp_min",0.0001),tot_steps = kwargs.get("total_periods", 1000000)))
                        policy=TimeDecliningExploration(beta = kwargs.get("beta", 0.0002)))

    env = AuctionEnvironment(
        total_periods=kwargs.get("total_periods",1000000),
        action_space_num=kwargs.get("action_space_num",15),
        alpha = kwargs.get("alpha",1),
        agents = [agent1, titfortat_agent]
    )

    env.run_simulation()
    #env.run_simulation_dont_provide_other_players_info()

    results = Results(env)
    return results

if __name__ == "__main__":
    results = run(num_agent=2, alpha=2)
    results.print_results()
    print(results.bid_history[-100:])