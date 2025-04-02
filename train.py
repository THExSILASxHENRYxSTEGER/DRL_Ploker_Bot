from Poker_env_utils import Tournament
from RL_utils import Agent
import numpy as np

np.random.seed(1)

n_players = 4

if n_players < 2 or 10 < n_players:
    Exception(ValueError("Texas Hold em can be played by minimally 2 and maximally 10 players"))

init_player_cash = 5000
blind_size = 25
double_intervl = 10
n_tournaments = 100

players = [Agent() for _ in range(n_players)]

for i in range(n_tournaments):
    tournament = Tournament(players, init_player_cash, blind_size, double_intervl)
    while tournament.ongoing:
        tournament.play_round()