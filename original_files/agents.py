"""Agents for Fishing Season (MALIS, Project 1).

This file is a skeleton. Write all your agents in it.

Do NOT modify the classes Agent and ContextualAgent, nor the function play():
they define how every agent interacts with the simulator, and they are used
to run your agents during grading.

How an agent works 
------------------
For each season, play(agent, bay) calls:

    agent.reset()                     once, before the first day
    spot = agent.choose()             every day: return a spot, 0 to 4
    agent.update(spot, catch)         every day: the catch you observed

An agent only learns through update(). It must not look at the bay itself.

Contextual agents (Part IV, optional) follow the same pattern, but receive the
morning conditions x (numpy array [wind_kmh, waves_m, tide_m]):

    spot = agent.choose(x)
    agent.update(x, spot, catch)

Example
-------
    from fishing import FishingBay
    from agents import RandomAgent, play

    bay = FishingBay(season=1, seed=0)
    total = play(RandomAgent(seed=0), bay)
"""
import numpy as np
from fishing import ContextualBay, FishingBay
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# Provided: do not modify
# ---------------------------------------------------------------------------


class Agent:
    """Base class for the agents of Parts II and III.

    Parameters
    ----------
    n_spots : int
        Number of fishing spots (5 in the game).
    seed : int or None
        Seed of the agent's own random generator, self.rng. Use self.rng for
        any random decision, so that results can be reproduced.
    """

    def __init__(self, n_spots=5, seed=None):
        self.n_spots = n_spots
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Called once at the start of every season. Clear what the agent learned."""

    def choose(self):
        """Return the spot to fish today, an int between 0 and n_spots - 1."""
        raise NotImplementedError

    def update(self, spot, catch):
        """Called after each day with the spot chosen and the catch observed (kg)."""


class ContextualAgent:
    """Base class for the agents of Part IV (contextual bay).

    Same as Agent, except that choose() and update() also receive x, the
    conditions of the day: numpy array [wind_kmh, waves_m, tide_m].
    """

    def __init__(self, n_spots=5, seed=None):
        self.n_spots = n_spots
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """Called once at the start of every season. Clear what the agent learned."""

    def choose(self, x):
        """Return the spot to fish today given the conditions x."""
        raise NotImplementedError

    def update(self, x, spot, catch):
        """Called after each day with the conditions, the spot chosen and the catch observed."""


def play(agent, bay):
    """Play one full season of bay with agent. Returns the total catch (kg).

    Works with FishingBay and Agent, or with ContextualBay and ContextualAgent.
    The bay is reset first, so the same bay can be played by several agents.
    """
    contextual = isinstance(bay, ContextualBay)
    if contextual and not isinstance(agent, ContextualAgent):
        raise TypeError("a ContextualBay must be played by a ContextualAgent")
    if not contextual and not isinstance(agent, Agent):
        raise TypeError("a FishingBay must be played by an Agent")
    bay.reset()
    agent.reset()
    while not bay.done:
        if contextual:
            x = bay.observe_array()
            spot = int(agent.choose(x))
            catch = bay.fish(spot)
            agent.update(x, spot, catch)
        else:
            spot = int(agent.choose())
            catch = bay.fish(spot)
            agent.update(spot, catch)
    return bay.total


# ---------------------------------------------------------------------------
# Provided example agents (baselines)
# ---------------------------------------------------------------------------


class RandomAgent(Agent):
    """Chooses a spot uniformly at random every day. Learns nothing."""

    def choose(self):
        return int(self.rng.integers(self.n_spots))


class RandomContextualAgent(ContextualAgent):
    """Ignores the conditions and chooses a spot uniformly at random every day."""

    def choose(self, x):
        return int(self.rng.integers(self.n_spots))


# ---------------------------------------------------------------------------
# Your agents
# ---------------------------------------------------------------------------


class HumanRuleAgent(Agent):
    """Task 4: the best human strategy identified in Part II, written as a rule.

    Describe the rule here, precisely enough that someone could apply it by hand.
    Add constructor parameters if your rule has any (call super().__init__ first).
    """

    def reset(self):
        raise NotImplementedError("Task 4")

    def choose(self):
        raise NotImplementedError("Task 4")

    def update(self, spot, catch):
        raise NotImplementedError("Task 4")


# Task 4: add below at least two agents implementing methods from the reading
# (Sutton and Barto, Chapter 2), as subclasses of Agent. Document each class:
# what it keeps in memory, how it chooses, what its parameters mean.
class EpsilonGreedyAgent(Agent):
    # historic {bay_num 1: [catch 1, catch 2 ...catch n ],... bay_num k : [catch 1 ,...,catch l]}
    def __init__(self, n_spots=5, seed=None,epsilon=0.5):
            self.n_spots = n_spots
            self.rng = np.random.default_rng(seed)
            self.epsilon=epsilon
            self.historic={}

    def reset(self):
        for i in range(self.n_spots):
            self.historic[i]=[]
    
    def choose(self):
        # take greedy action
        if(self.rng.random() <1-self.epsilon):
            each_average=[]
            for i in range(5):
                mean=np.array(self.historic[i]).mean() if self.historic[i]!=[] else float('inf')
                each_average.append(mean)
            
            return int(np.array(each_average).argmax())
        # take random action
        else:
            return int(self.rng.integers(self.n_spots))
        
    def update(self,spot,catch):
        self.historic[spot].append(catch)
    

# Task 6 (optional): add your contextual agents below, as subclasses of
# ContextualAgent.
class ContextualAgent(Agent):
    # self.training_days : number of days used to traing KNN/Linear Regresssion model
    def __init__(self, n_spots=5, seed=None,training_prop=0.5):
        self.n_spots = n_spots
        self.rng = np.random.default_rng(seed)
        self.training_days=int(300*training_prop) 
        self.passed_days=0
        self.passed_days_on_spot=0
        self.training_spot=0
        self.training_time_per_spot=int(self.training_days/self.n_spots)
        self.Xdata={}
        self.Ydata={}
        self.models={}
        for i in range(self.n_spots):
            self.Xdata[i]=[]
            self.Ydata[i]=[]
        
    def reset(self):
        self.passed_days=0
        self.passed_days_on_spot=0
        self.training_spot=0
        for i in range(self.n_spots):
            self.Xdata[i] = []
            self.Ydata[i] = []

    def choose(self,data):
        # data = ["wind_kmh", "waves_m", "tide_m"]
        # we continue to train our model
        if(self.passed_days<self.training_days):
            self.Xdata[self.training_spot].append(data)
            self.passed_days+=1
            self.passed_days_on_spot+=1
            chosen_spot=int(self.training_spot)

            # we change spot
            if(self.passed_days_on_spot>=self.training_time_per_spot):
                self.training_spot+=1
                self.passed_days_on_spot=0
                return chosen_spot
            return self.training_spot
        # predict the best spot 
        else:
            prediction_per_spot=[0 for i in range(self.n_spots)]
            for i in range(self.n_spots):
                prediction_per_spot[i]=self.models[i].predict([data])
            best_spot=(np.array(prediction_per_spot)).argmax()
            self.passed_days+=1
            return int(best_spot)


    def update(self,spot,catch):
            if(self.passed_days<=self.training_days):
                self.Ydata[spot].append(catch)
                # create the model with the data
                if(self.passed_days==self.training_days):
                    for i in range(self.n_spots):
                        Xdata=np.array(self.Xdata[i])
                        Ydata=np.array(self.Ydata[i])
                        self.models[i]=LinearRegression()
                        self.models[i].fit(Xdata,Ydata)
