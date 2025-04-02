import numpy as np
from itertools import product
from copy import deepcopy

class Card: # class of a normal card
    
    def __init__(self, num, suit):
        self.num = num
        self.suit = suit
        self.num_id = Deck.nums_ind[num]
        self.suit_id = Deck.suits_ind[suit]

    def __str__(self):
        return f"{self.suit}{self.num}"

    def get_card_id(self):
        return self.num_id, self.suit_id

    def __lt__(self, other):
        if self.num_id == other.num_id:
            return self.suit_id < other.suit_id
        else:
            return self.num_id < other.num_id


class Deck: # this class gives out a deck of 52 cards for texas hold em

    suits = ["♦", "❤", "♠", "♣"]
    nums = [*[str(i) for i in range(2,11)], "J", "Q", "K", "A"]

    suits_ind = {suit:i for (i,suit) in enumerate(suits)}
    nums_ind = {num:i for (i,num) in enumerate(nums)}

    def __init__(self):
        self.reset()

    def reset(self):
        self.avail_cards = [Card(num, suit) for (num, suit) in  list(product(*[Deck.nums, Deck.suits]))]    
        self.avail_cards = self.shuffle(self)

    @staticmethod
    def shuffle(deck):
        indcs = np.random.permutation(len(deck.avail_cards))
        return [deck.avail_cards[i] for i in indcs]
    
    def draw(self):
        return self.avail_cards.pop(0)

class Hand: # class of a hand that a player can have is supposed to show if one hand is bigger than another
    
    def __init__(self, card1, card2):
        self.card1, self.card2 = sorted([card1, card2]) # sort cards so NNs dont need to memorize multiple combinations
        self.hand_id = [*card1.get_card_id(), *card2.get_card_id()]

    def __str__(self):
        return f"{str(self.card1)} {str(self.card2)}"

# !!!!!!!!!! order cards always from lowest to highest or vice versa so model has not to learn order of cards dont matter 

class Flop:
    def __init__(self, card1, card2, card3):
        self.card1, self.card2, self.card3 = sorted([card1, card2, card3])
        self.hand_id = [*card1.get_card_id(), *card2.get_card_id(), *card3.get_card_id()]

    def __str__(self):
        return f"{str(self.card1)} {str(self.card2)} {str(self.card3)}"

class Round:

    draw_hand = lambda deck, n: [deck.draw() for _ in range(n)]

    def __init__(self, players, blind, start_ind):
        self.pot = 0
        self.blind = blind
        self.active_players = players[start_ind:]+players[:start_ind]
        self.folder_indcs = list()
        self.history = { # collect over betting rounds
            "blind" : blind,
            "player_hands" : None, # hands of all players in betting order
            "crds_rnd" : [], # all cards in order of the round
            "sub_rnds" : {
                "preflop" : [], # preflop history of betting order
                "flop" : [],  # flop history
                "turn" : [],   # turn history
                "river" : [],   # river history
            },
            "winner_id": -1
        } # sample batches to train

    def erase_current_stake(self):
        for ap in self.active_players:
            ap.current_stake = 0

    def betting_round(self, cardsets, stake, method_name, history, first_bid=True): # betting rounds (2 betting rounds per preflop,flop,turn and river) i.e. iterations around the table after each time cards are dealt/shown
        round_over = False
        for i, player in enumerate(self.active_players):
            if i+1 == len(self.active_players) and i == len(self.folder_indcs): # in case all players fold the last player wins all by default
                self.active_players[i].add_cash(self.pot)
                return _, True # True marks the game is over
            player_bet = getattr(player, method_name, None)
            if len(cardsets) > 1: 
                relevant_cds = cardsets[i]
            else: 
                relevant_cds = cardsets[0]
            p_stake, add_pot, fold = player_bet(relevant_cds, self.blind, stake, history, first_bid=first_bid)
            history.append((p_stake, add_pot, fold))
            if first_bid:
                stake = np.max([stake, p_stake])
            if fold:
                self.folder_indcs.append(i)
                continue
            self.pot += add_pot
        for i in reversed(self.folder_indcs):
            self.active_players.pop(i)   
            if len(cardsets) > 1:
                cardsets.pop(i)
        if len(self.active_players) == 1:
            self.active_players[0].add_cash(self.pot)
            round_over = True
        self.folder_indcs.clear()
        return stake, round_over

    def preflop(self, deck):
        n_pl = len(self.active_players)
        hands = [Hand(c1,c2) for (c1,c2) in zip(Round.draw_hand(deck,n_pl), Round.draw_hand(deck,n_pl))]
        self.history["player_hands"] = deepcopy(hands)
        self.pot = 3 * self.blind
        self.active_players[-1].take_big_blind(self.blind)
        self.active_players[-2].take_small_blind(self.blind)
        stake = 2 * self.blind           # the current minimum stake all players need to pay in to participate
        round_func = "bet_preflop"
        #first betting round
        stake, over = self.betting_round(hands, stake, round_func, self.history["pfl"], first_bid=True)
        #second betting round
        if not over:
            _, over = self.betting_round(hands, stake, round_func, self.history["pfl"], first_bid=False)
        return over

    def subround(self, deck, sub_rnd):
        stake = 0
        if sub_rnd == "preflop":
            n_pl = len(self.active_players)
            cardsets = [Hand(c1,c2) for (c1,c2) in zip(Round.draw_hand(deck,n_pl), Round.draw_hand(deck,n_pl))]
            self.history["player_hands"] = deepcopy(cardsets)
            self.pot = 3 * self.blind
            self.active_players[-1].take_big_blind(self.blind)
            self.active_players[-2].take_small_blind(self.blind)
            stake = 2 * self.blind           # the current minimum stake all players need to pay in to participate
        else:
            deck.draw() # one card is burned if its not preflop
            if sub_rnd == "flop":
                cardsets = [Flop(deck.draw(), deck.draw(), deck.draw())]    
            else:
                cardsets = [deck.draw()]
            self.history["crds_rnd"] += cardsets
        #first betting round along the table
        stake, over = self.betting_round(cardsets, stake, sub_rnd, self.history["sub_rnds"][sub_rnd],first_bid=True)
        #second betting round along the table
        if not over:
            _, over = self.betting_round(cardsets, stake, sub_rnd, self.history["sub_rnds"][sub_rnd],first_bid=False)
        if not over and sub_rnd == "river":
            self.showdown()
        return over, self.history
        
# !!!!!!! program showdown and also save winner id in history, but beware the winner can also be decided when everyone
# else folds so also needed to manipulate betting rounds !!!!!!!!! 
    def showdown(self): 
        pass

class Tournament:
    
    def __init__(self, players, init_player_cash, blind_size, double_intervl=30):
        self.players = players
        for player in self.players:
            player.set_params(init_player_cash, blind_size)
        self.blind = blind_size
        self.intrvl = double_intervl
        self.n_rounds = 0
        self.deck = Deck()
        self.ongoing = True
        self.start_ind = np.random.randint(len(players)) # the index of the player next to the big blind

    def play_round(self):
        self.deck.reset()
        self.n_rounds += 1
        if self.n_rounds % self.intrvl == 0:
            self.blind *= 2
        round = Round(self.players, self.blind, self.start_ind)
        self.start_ind = self.start_ind + 1 if self.start_ind + 1 < len(self.players) else 0        
        for sub_rnd in round.history["sub_rnds"].keys():
            over, history = round.subround(self.deck, sub_rnd)
            round.erase_current_stake()
        return history

        

        