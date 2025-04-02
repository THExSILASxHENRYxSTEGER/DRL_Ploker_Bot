import numpy as np
from itertools import product
from copy import deepcopy
from enum import Enum, auto

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

    def get_cards(self):
        return self.card1, self.card2

class Flop:
    
    def __init__(self, card1, card2, card3):
        self.card1, self.card2, self.card3 = sorted([card1, card2, card3])
        self.flop_id = [*card1.get_card_id(), *card2.get_card_id(), *card3.get_card_id()]

    def __str__(self):
        return f"{str(self.card1)} {str(self.card2)} {str(self.card3)}"

    def get_cards(self):
        return self.card1, self.card2, self.card3

class Combinations:
    
    def __init__(self, hand, tbl_crds): # tbl_crds are the cards on the table every player shares
        self.hand = hand
        self.tbl_crds = tbl_crds
        combntn_names = [
            "royal_flush",
            "straight_flush",
            "four_of_a_kind",
            "full_house",
            "flush",
            "straight",
            "three_of_a_kind",
            "two_pairs",
            "pair",
            "high_card",
            ]
        self.cmbntns = self.get_cmbnts(combntn_names)
    
    def get_cmbnts(self, combntn_names):
        cmbntns = dict()
        for name in combntn_names:
            funcl = getattr(self, name, None)
            comb = funcl()
            cmbntns[name] = comb

    def royal_flush(self): # each of the functions returns a list of the individual players hands categories sorted descendingly by strength for example in pairs all individual pairs the player has are accumulated
        pass

    def straight_flush(self):
        pass

    def four_of_a_kind(self):
        pass
    
    def full_house(self):
        pass
    
    def flush(self):
        pass
    
    def straight(self):
        pass
    
    def three_of_a_kind(self):
        pass

    def two_pairs(self):
        pass

    def pair(self):
        pass

    def high_card(self):
        return reversed([*self.hand.get_cards()])

    def __lt__(self, other): # go down the self.cmbntns lists and see which player has highest individual combiation
        pass # !!!! consider split pot

class Round:

    draw_hand = lambda deck, n: [deck.draw() for _ in range(n)]

    def __init__(self, players, blind, start_ind):
        self.pot = 0
        self.blind = blind
        self.active_players = players[start_ind:]+players[:start_ind]
        self.folder_indcs = list()
        self.history = { # collect over betting rounds
            "ap_indcs" : list(range(len(self.active_players))), # initial indeces of active players removed after flop
            "blind" : blind,
            "player_hands" : None, # hands of all players in betting order
            "tbl_crds" : [], # all table cards in order of the round
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
            self.history["ap_indcs"].pop(i)
            if len(cardsets) > 1:
                cardsets.pop(i)
        if len(self.active_players) == 1:
            self.active_players[0].add_cash(self.pot)
            round_over = True
        self.folder_indcs.clear()
        return stake, round_over

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
            self.history["tbl_crds"] += cardsets
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
        player_cmbnts = list()
        for idx in self.history["ap_indcs"]: 
            cmbnts = Combinations(self.history["player_hands"][idx], self.history["tbl_crds"])
            player_cmbnts.append(cmbnts)

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

        

        