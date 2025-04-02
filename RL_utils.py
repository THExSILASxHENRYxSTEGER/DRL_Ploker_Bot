import torch
from torch import nn
import numpy as np

class Agent(nn.Module): # is one player in a card game of texas hold em 

    def __init__(self):
        super(Agent, self).__init__()
        self.cash = 1 # the player sees the percentage of the starting bid
        self.big_blind = False
        
    def set_params(self, total_cash, total_blind):
        self.total_cash = total_cash
        self.total_blind = total_blind
        self.blind = total_blind/total_cash
        self.current_stake = 0 # its the stake already invested at the *current* preflop, flop, turn or river, not the entire round

    def add_cash(self, cash):
        self.cash += cash/self.total_cash

    def take_cash(self, cash):
        self.cash -= cash/self.total_cash

    def take_big_blind(self, blind):
        self.big_blind = True
        self.take_cash(2*blind)
        self.current_stake += 2*blind

    def take_small_blind(self,blind):
        self.take_cash(blind)
        self.current_stake += blind

    def random_bet(self, blind, min_stake, history, first_bid=True, p_fold=0.2):
        if min_stake == self.current_stake or np.random.rand() > p_fold: # players are forced to check if they dont have to call or raise else possible fold
            rais = first_bid * blind * np.random.randint(0,4)
            stake = (min_stake - self.current_stake) + rais
            self.current_stake += stake
            self.take_cash(stake)
            return self.current_stake, stake, False # last variable is whether player folds
        else:
            self.current_stake = 0
            return 0, 0, True

    def preflop(self, hand, blind, min_stake, history, first_bid=True, p_fold=0.2): # the agent bets during this stage
        return self.random_bet(blind, min_stake, history, first_bid, p_fold)

    def flop(self, flop, blind, min_stake, history, first_bid=True, p_fold=0.1):
        return self.random_bet(blind, min_stake, history, first_bid, p_fold)

    def turn(self, turn, blind, min_stake, history, first_bid=True, p_fold=0.1):
        return self.random_bet(blind, min_stake, history, first_bid, p_fold)

    def river(self, river, blind, min_stake, history, first_bid=True, p_fold=0.1):
        return self.random_bet(blind, min_stake, history, first_bid, p_fold)






































# !!!!! have a structure such that there is for sure 4 lstm cells i.e. one per each round and 
# in between all these lstm cells are one more lstm cells that is conditional on whether or not there is a second bidding round 