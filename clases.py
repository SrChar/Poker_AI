import random
import funciones as f
from collections import Counter
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PPOPolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        """
        Initializes a Proximal Policy Optimization (PPO) Policy Network.

        :param num_inputs: Number of input features to the network.
        :param num_actions: Number of possible actions or output features.
        """
        super(PPOPolicyNetwork, self).__init__()
        # Arquitectura de la red neuronal
        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_actions)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        :param x: The input tensor.
        :return: A tuple containing action probabilities and state values.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.fc3(x), dim=0)
        state_values = self.fc4(x)
        return action_probs, state_values

class PPO:
    def __init__(self, num_inputs, num_actions, lr=0.002, gamma=0.99, eps_clip=0.2):
        """
        Initializes the Proximal Policy Optimization (PPO) algorithm.

        :param num_inputs: Number of input features.
        :param num_actions: Number of possible actions.
        :param lr: Learning rate for the optimizer.
        :param gamma: Discount factor for future rewards.
        :param eps_clip: Clipping parameter for PPO.
        """
        self.policy = PPOPolicyNetwork(num_inputs, num_actions)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def select_action(self, state):
        """
        Selects an action based on the current state.

        :param state: The current state represented as a numpy array.
        :return: A tuple of the chosen action and its log probability.
        """
        state = torch.from_numpy(state).float().unsqueeze(0) # Se convierte el estado a un tensor de torch
        action_probs, _ = self.policy(state)

        # Convierte action_probs en un tensor si no lo es
        if not isinstance(action_probs, torch.Tensor):
            action_probs = torch.tensor(action_probs)
        distribution = Categorical(action_probs)
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)

    def update(self, rewards, states, actions, log_probs):
        """
        Updates the policy network based on the received batch of experience.

        :param rewards: List of rewards from the batch.
        :param states: List of states from the batch.
        :param actions: List of actions taken in each state.
        :param log_probs: Log probabilities of each action.
        """
        # Calcular recompensas descontadas
        discounts = [self.gamma ** i for i in range(len(rewards) + 1)]
        R = []
        for step in range(len(rewards)):
            R.append(sum([a * b for a, b in zip(discounts[:len(rewards)-step], rewards[step:])]))

        old_states = torch.stack(states)
        old_actions = torch.tensor(actions, dtype=torch.long).view(-1, 1)
        old_log_probs = torch.stack(log_probs)

        # Convertir recompensas descontadas en un tensor
        R = torch.tensor(R, dtype=torch.float32)

        # Normalizar las recompensas
        R = (R - R.mean()) / (R.std() + 1e-5)

        # Obtener los nuevos log_probs, valores y estados de ventaja
        action_probs, state_values = self.policy(old_states)
        V = state_values.squeeze()

        new_log_probs = action_probs.gather(1, old_actions).squeeze(1)
        advantage = R - V

        # Calcular la función de pérdida y realizar una actualización de gradiente
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
        loss = -torch.min(surr1, surr2).mean()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

class Player():
    def __init__(self, id, budget, n_simulations):
        """
        Initializes a player with a unique ID, budget, and number of simulations for decision-making.

        :param id: Unique identifier for the player.
        :param budget: Starting budget or money the player has.
        :param n_simulations: Number of simulations to run for probability calculations.
        """
        self.id = id
        self.hand = []
        self.initial_budget = budget
        self.budget = budget
        self.hand_calculators = [self.straight_type, self.emparejar, self.flush, self.high_card]
        self.hand_names = ["ROYAL FLUSH", "STRAIGHT FLUSH", "FOAK", "FULL HOUSE", "FLUSH", "STRAIGHT", "TOAK", "DOUBLE PAIR", "PAIR", "HIGH CARD"]
        self.apuesta_actual = 0
        self.n_simulations = n_simulations

    def set_cards(self, card_list):
        """
        Sets the player's hand based on a given list of cards.

        :param card_list: List of Card objects to be set as the player's hand.
        """
        self.hand = sorted(card_list, key=f.get_card_value, reverse=True)

    def reset_info(self):
        """
        Resets the player's budget, bet, and hand to their initial state.
        """
        self.budget = self.initial_budget
        self.apuesta_actual = 0
        self.better_hand = []
        self.probability = 0
        self.EV = 0
        self.hand = []

    def bet(self, hand, round=1):
        """
        Executes a betting action for the player.

        :param hand: The current Hand object being played.
        :param round: Indicates the betting round number.
        :return: The amount bet by the player.
        """
        if self.apuesta_actual > self.budget: # Si self.apuesta_actual > self.budget, se hace all in
            self.apuesta_actual = self.budget

        if round == 2: # En el round 2, como mucho, se puede hacer call
            if self.apuesta_actual >= hand.max_bet - hand.find_player_bet(self):
                self.apuesta_actual = hand.max_bet - hand.find_player_bet(self)

        hand.add_bet(self, self.apuesta_actual)
        self.budget -= self.apuesta_actual
        return self.apuesta_actual

    def act(self, hand, n_hands, print_values=False):
        """
        Implements the action decision for a Player playing style.

        :param hand: Current hand of the game.
        :param n_hands: Number of hands played.
        :param print_values: Flag to print the decision-making process.
        :return: Integer indicating the action taken.
        """
        pass

    def get_probabilities(self, hand, print_values=False):
        """
        Calculates the win probability for the player in the current hand.

        :param hand: The current Hand object being played.
        :param print_values: Flag to print additional information for debugging or analysis.
        """
        cards_to_remove = self.hand + hand.community_cards
        victories = 0
        player_list = [Player(id=i + 1, budget=0, n_simulations=self.n_simulations) for i in
                       range(len(hand.active_players))] # Se crean los jugadrores ficticios de cada simulacion
        for i in range(self.n_simulations):
            [player.reset_info() for player in player_list] # Reseteamos los parámetros de cada player
            player_list[0].set_cards(self.hand.copy()) # Se establecen las cartas del jugador
            simulated_hand = Hand(player_list)
            simulated_hand.community_cards = hand.community_cards.copy()
            simulated_hand.remove_cards(cards_to_remove) # Se eliminan las cartas que ya tiene el jugador
            simulated_hand.deal_cards(restriction=player_list[0].id)
            simulated_hand.set_community_cards(5-len(simulated_hand.community_cards))
            for player in simulated_hand.active_players:
                if print_values:
                    print(
                        f"Player {player.id} --- {player.hand[0].number} {player.hand[0].suit}, {player.hand[1].number} {player.hand[1].suit}")
                player.get_best_hand(simulated_hand, print_values)
                if print_values:
                    print("-" * 100)

            simulated_hand.get_winners()
            if print_values:
                for winner in simulated_hand.winners:
                    print("WINNER", winner.id)
                    winner.print_hand()
                    print(winner.hand_names[winner.better_hand[0]], winner.better_hand[1])

            for winner in simulated_hand.winners:
                if winner.id == player_list[0].id:
                    victories += 1

            if print_values:
                print("\n\n")

        self.probability = victories / self.n_simulations

    def get_EV(self, hand):
        """
        Calculates the Expected Value (EV) for the player in the current betting scenario.

        :param hand: The current Hand object being played.
        """
        self.EV = (self.probability * (hand.get_pot() + hand.max_bet)) - ((1-self.probability) * hand.max_bet)

    def add_budget(self, value):
        """
        Adds a specified value to the player's budget.

        :param value: The amount to be added to the budget.
        """
        self.budget += value

    def print_hand(self):
        """
        Prints the player's hand.
        """
        for card in self.hand:
            print(card.number, card.suit)

    def get_best_hand(self, hand, print_values=False):
        """
        Determines the best poker hand that the player can make from their hand and community cards.

        :param hand: The current Hand object being played.
        :param print_values: Flag to print additional information for debugging or analysis.
        """
        self.total_cards = hand.community_cards + self.hand
        self.total_cards.sort(key=f.get_card_value, reverse=True)
        self.total_cards_numbers = [card.number for card in self.total_cards]
        self.total_cards_suits = [card.suit for card in self.total_cards]
        self.count_cards_number = Counter(self.total_cards_numbers)

        if print_values:
            for card in self.total_cards:
                print(card.number, card.suit)

        jugadas = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for fun in (self.hand_calculators):
            fun(jugadas)
        jugadas_validas = [i for i in jugadas if i != 0]
        result = list(jugadas_validas[-1])
        if result != None:
            # print(self.hand_names[i], result)
            if print_values:
                print(self.hand_names[result[0]], result[1])
            self.better_hand = result
            return

    def high_card(self, arr):
        """
        Evaluates and records if the player's best hand is a high card.

        :param arr: An array to store the results of the hand evaluation.
        """
        arr[0] = (9, [self.hand[0].number, self.hand[1].number])

    def emparejar(self, arr): # metemos el TOAK, FOAK y FULLHOUSE
        """
        Evaluates and records pairs, three of a kinds, full houses in the player's hand.

        :param arr: An array to store the results of the hand evaluation.
        """
        pairs = []
        trios = []
        for element, count in self.count_cards_number.items():
            if count == 4:
                arr[7] = (2, [element])
                #return (2, [element])
            if count == 3:
                trios.append(element)
                #return (6, [element])
            if count == 2:
                pairs.append(element)
                #return (8, [element])

        if len(pairs)>0:
            arr[1] = (8, [max(pairs)])
            if len(pairs) > 1:
                arr[2] = (7, pairs[:2])

        if len(trios) >= 1:
            arr[3] = (6, [max(trios)])
            if len(pairs) >= 1 and len([i for i in pairs if i not in trios]) > 0:
                arr[6] = (3, [max(trios), max([i for i in pairs if i not in trios])])

    def straight_type(self, arr):
        """
        Evaluates and records if the player's best hand is a straight or straight flush.

        :param arr: An array to store the results of the hand evaluation.
        """
        for i in range(len(self.total_cards) - 4):

            same_suit = all(self.total_cards_suits[j] == self.total_cards_suits[j + 1] for j in range(i, i + 4))

            # Manejar el caso especial A-2-3-4-5
            is_special_straight = (
                    f.get_card_value(self.total_cards[i]) == 14 and
                    f.get_card_value(self.total_cards[i + 1]) == 2 and
                    f.get_card_value(self.total_cards[i + 2]) == 3 and
                    f.get_card_value(self.total_cards[i + 3]) == 4 and
                    f.get_card_value(self.total_cards[i + 4]) == 5
            )

            if is_special_straight:
                if same_suit:
                    arr[8] = (1, [self.total_cards[i].number])  # Identificar como escalera de color especial
                else:
                    arr[4] = (5, [self.total_cards[i].number])  # Identificar como escalera normal especial

            for j in range(i, i + 4):
                # print("I, J", i, j)
                if f.get_card_value(self.total_cards[j]) != f.get_card_value(self.total_cards[j + 1]) + 1:
                    break
                if j == i + 3:
                    arr[4] = (5, [self.total_cards[i].number])
                    if same_suit:
                        if self.total_cards[i].number == 'A':
                            arr[9] = (0, [self.total_cards[i].number])
                        else:
                            arr[8] = (1, [self.total_cards[i].number])

    def flush(self, arr):
        """
        Evaluates and records if the player's best hand is a flush.

        :param arr: An array to store the results of the hand evaluation.
        """
        counts = Counter(self.total_cards_suits)
        for suit, count in counts.items():
            if count >= 5:
                arr[5] = (4, [card.number for card in self.total_cards if card.suit == suit])

class Tight_Agresivo(Player):
    # Este jugador será muy efecivo ya que jugará sólo cunado disponga de una muy buena mano.
    # En ese caso apostará una gran cantidad de dinero. Sin embargo, jugará pocas partidas.
    def act(self, hand, n_hands, print_values=False):

        # Calcula la probabilidad y el valor de ganancia esperado
        self.get_probabilities(hand, print_values=print_values)
        self.get_EV(hand)

        # Estrategia Tight Agresiva
        ghost_bet = self.budget * self.probability * 0.2
        min_bet_to_go = hand.get_max_bet() - hand.find_player_bet(self)

        if self.probability > 0.25:
            if ghost_bet >= min_bet_to_go:  # Por ejemplo, si la probabilidad de ganar es mayor al 25%
                self.apuesta_actual = ghost_bet  #
                a = 1
            else:  # Cuando el 20% del presupuesto sea menor que la apuesta mínima, iguala
                self.apuesta_actual = min_bet_to_go
                a = 1
        elif self.probability > 0.08:  # Si la probabilidad de ganar es mayor al 8%
            if ghost_bet/2 >= min_bet_to_go:
                self.apuesta_actual = ghost_bet/2
                a = 1

            else:
                self.apuesta_actual = min_bet_to_go
                a = 1
        else:
            self.apuesta_actual = 0  # Si la probabilidad de ganar es baja, no apuesta
            a = 0

        return a

class Bluffing(Player):
    def act(self, hand, n_hands, print_values=False):
        a = 0
        # Calcula la probabilidad y el valor de ganancia esperado
        self.get_probabilities(hand, print_values=print_values)
        self.get_EV(hand)

        # Estrategia de Bluffing
        ghost_bet = self.budget * self.probability
        min_bet_to_go = hand.get_max_bet() - hand.find_player_bet(self)

        if self.probability > 0.35:  # Si la probabilidad de ganar es alta
            if ghost_bet * 0.13 > min_bet_to_go:
                self.apuesta_actual = ghost_bet * 0.13 # Apuesta un 13% del presupuesto
                a = 1

            else:
                self.apuesta_actual = min_bet_to_go
                a = 1

        elif self.probability > 0.15:  # Si la probabilidad de ganar es moderada
            if ghost_bet * 0.08 >= min_bet_to_go:
                self.apuesta_actual = min_bet_to_go  # Apuesta un 8% del presupuesto
                a = 1
            else:
                self.apuesta_actual = min_bet_to_go
                a = 1

        # Estrategia de Bluffing basada en valor de ganancia esperado y ocasional farol
        if self.EV < hand.get_max_bet() and random.random() < 0.2:  # Realiza un farol con un 20% de probabilidad
            if ghost_bet * 0.5 > min_bet_to_go:
                self.apuesta_actual = ghost_bet * 0.5  # Aumenta la apuesta para intentar engañar a los oponentes
            else:
                self.apuesta_actual = min_bet_to_go
            a = 1
        return a

class JuegoPasivo(Player):
    def act(self, hand, n_hands, print_values=False):

        # Calcula la probabilidad y el valor de ganancia esperado
        self.get_probabilities(hand, print_values=print_values)
        self.get_EV(hand)
        min_bet_to_go = hand.get_max_bet() - hand.find_player_bet(self)

        # Estrategia de Juego Pasivo
        if self.probability > 0.3 and self.EV >= min_bet_to_go:  # Si la probabilidad de tener la mejor mano es alta
            ghost_bet = (self.probability*self.budget)/10 # Iguala la apuesta actual (call)
            if ghost_bet > min_bet_to_go:
                self.apuesta_actual = ghost_bet
                return 1
            else:
                return 0
        return 0

class NotTrainedModel(Player):
    def __init__(self, id, budget, n_simulations,):
        """
        Initializes a player with a pre-trained model for decision making.

        :param id: Player's ID.
        :param budget: Starting budget for the player.
        :param n_simulations: Number of simulations for calculating probabilities.
        """
        super().__init__(id, budget, n_simulations)
        self.states = []
        self.actions = []
        self.recompensas = []
        self.log_probs = []

        # Parámetros de la red neuronal y PPO
        self.num_inputs = 5  # Número de variables que influyen en el estado
        self.num_actions = 3  # Número de acciones posibles
        self.ppo = PPO(self.num_inputs, self.num_actions)

    def act(self, hand, n_hands, print_values=False):
        self.get_probabilities(hand, print_values=print_values)
        self.get_EV(hand)
        #n_pasive = self.count_pasive_players(hand, n_hands)

        state = np.array([self.probability, self.EV, self.hand[0].get_value(), self.hand[1].get_value(), self.budget])  # Se establece el state actual
        state = (state - state.mean()) / state.std() # Normalizamos los valores del state

        action, log_prob = self.ppo.select_action(state) # seleccionamos la acción más probable

        state_tensor = torch.from_numpy(state).float()
        action_tensor = torch.tensor([action], dtype=torch.long)

        # Almacenamos cada variable a la lista correspondiente
        self.states.append(state_tensor)
        self.actions.append(action_tensor)
        self.log_probs.append(log_prob)

        min_bet_to_go = hand.get_max_bet() - hand.find_player_bet(self) #Apuesta mínima para call
        if action == 0:
            return 0
        elif action == 1:
            self.apuesta_actual = min_bet_to_go
        elif action == 2:
            self.apuesta_actual = self.budget*0.15
            if not self.apuesta_actual > min_bet_to_go:
                self.apuesta_actual = min_bet_to_go
        return 1

    def add_recompensa(self, recompensa):
        """
        Adds a specified reward to the player's list of rewards, ensuring that the list of rewards aligns with the list of states.

        :param recompensa: The reward value to be added.
        """
        for i in range(len(self.states) - len(self.recompensas)):
            self.recompensas.append(recompensa)

    def update(self):
        """
        Updates the underlying Proximal Policy Optimization (PPO) model with the accumulated states, actions, rewards, and log probabilities.
        After updating, it resets these lists for the next iteration of learning.
        """
        # Actualizamos el modelo con la informacion obtenida y reseteamos las listas
        self.ppo.update(self.recompensas, self.states, self.actions, self.log_probs)
        self.states = []
        self.actions = []
        self.recompensas = []
        self.log_probs = []

class TrainedModel(Player):
    def __init__(self, id, budget, n_simulations, model_path, num_inputs, num_actions):
        """
        Initializes a player with a specified trained model for decision making.

        :param id: Player's ID.
        :param budget: Starting budget for the player.
        :param n_simulations: Number of simulations for calculating probabilities.
        :param model_path: Path to the trained model.
        :param num_inputs: Number of input features for the model.
        :param num_actions: Number of possible actions in the model.
        """
        super().__init__(id, budget, n_simulations)
        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.model = PPOPolicyNetwork(num_inputs=5, num_actions=3)
        self.model_state = torch.load(model_path, map_location=torch.device('cpu')) # Cargamos el modelo
        self.model.load_state_dict(self.model_state)
        self.model.eval() # Se establece que el modelo será usado para testing y no se entrenará

    def act(self, hand, n_hands, print_values=False):
        self.get_probabilities(hand, print_values=print_values)
        self.get_EV(hand)
        # n_pasive = self.count_pasive_players(hand, n_hands)
        state = np.array([self.probability, self.EV, self.hand[0].get_value(), self.hand[1].get_value(), self.budget])  # Convierte el estado del juego en un tensor
        state = (state - state.mean()) / state.std()
        state_tensor = torch.from_numpy(state).float()
        action_probs, state_value = self.model(state_tensor)
        action = torch.argmax(action_probs, dim=0).item()

        min_bet_to_go = hand.get_max_bet() - hand.find_player_bet(self) #Apuesta mínima para call
        if action == 0:
            return 0
        elif action == 1:
            self.apuesta_actual = min_bet_to_go
        elif action == 2:
            self.apuesta_actual = self.budget*0.15
            if not self.apuesta_actual > min_bet_to_go:
                self.apuesta_actual = min_bet_to_go
        return 1

class Card():
    def __init__(self, number, suit):
        self.number = number
        self.suit = suit
        self.values = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
              "J": 11, "Q": 12, "K": 13, "A": 14}

    def get_number(self):
        return self.number

    def get_suit(self):
        return self.suit

    def get_value(self):
        return self.values[self.number]

numbers = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
global_deck = [Card(number, suit) for number in numbers for suit in suits] # Definimos el deck

class Hand():
    def __init__(self, active_players):
        """
        Initializes a poker hand with a list of active players.

        :param active_players: List of Player objects participating in the hand.
        """
        self.blinds = []
        self.winners = []
        self.price = 0
        self.active_players = active_players.copy()
        self.aviable_cards = global_deck.copy()#[Card(number, suit) for number in numbers for suit in suits]
        random.shuffle(self.aviable_cards)
        self.community_cards = []
        self.betting_template = {player: 0 for player in self.active_players}
        self.players_bets = self.betting_template.copy()
        self.max_bet = 0

    def set_community_cards(self, num_cards):
        """
        Sets a specified number of community cards for the hand.

        :param num_cards: The number of community cards to be set.
        """
        for _ in range(num_cards):
            self.community_cards.append(self.aviable_cards.pop())

    def distribute_pot(self):
        """
        Distributes the pot among the winners at the end of a hand.
        """
        # Se toman todos los jugadores activos o no, cuya apuesta haya sido > 0
        players_bets_copy = {k: v for k, v in sorted(self.players_bets.copy().items(), key=lambda item: item[1]) if v > 0}

        # Se toman todos os jugadores activos, que serán los que tendrán derecho a ganar los sidepots
        active_players_bets = {k: v for k, v in sorted(self.players_bets.copy().items(), key=lambda item: item[1]) if k in self.active_players}
        pots_dict = {}

        # Se toma la apuesta mínima de los jugadores activos. En caso de haber más de 1, significa que 1 o más jugadores hicieron all in
        # ya que si no, todos tendrían la misma apuesta.
        min_ap = min(list(active_players_bets.values()))

        while len(players_bets_copy) > 0:
            contribuyentes = [] # Almacenará los contribuyentes de cada sidepot
            players = list(players_bets_copy.keys())
            loosers_pot = 0 # Será el dinero que añaden los jugadores no activos a cada sidepot

            for player in players:
                if players_bets_copy[player] > 0:
                    if player in self.active_players: # Si la apuesta encontrada es de un jugador activo, se añade como contribuyente
                        contribuyentes.append(player)
                    else: # Si no, se añade el valor de la apuesta de dicho player en el loosers_pot
                        if min_ap > players_bets_copy[player]:
                            loosers_pot += players_bets_copy[player]
                        else:
                            loosers_pot += min_ap
                    players_bets_copy[player] -= min_ap # Se resta la apuesta mínima encontrada a todos los jugadores. Y se repite el ciclo
                    if player in active_players_bets:
                        active_players_bets[player] -= min_ap

                # Eliminamos a los jugadores que ya hemos revisado
                if players_bets_copy[player] <= 0:
                    del players_bets_copy[player]
                    if player in active_players_bets:
                        del active_players_bets[player]

            pots_dict[tuple(contribuyentes)] = min_ap*len(contribuyentes) + loosers_pot
            if len(active_players_bets) > 0:
                min_ap = min(list(active_players_bets.values()))
            else:
                break

        for contrib, pot in pots_dict.items(): # Evaluamos cada sidepot por separado
            winners = self.get_sp_winners(contrib) # Esta funcion devuelve los ganadores de un determinado sidepot
            for winner in winners:
                # Se reparte el dinero entre los winners de cada sidepot. En caso de empate se divide entre el numero de jugadores empatados
                winner.add_budget(pot / len(winners))

    def add_community_cards(self, card):
        """
        Adds a card to the community cards and removes it from the available deck.

        :param card: Card object to be added to the community cards.
        """
        self.remove_cards([card])
        self.community_cards.append(card)

    def find_player_bet(self, player):
        """
        Finds the current bet amount of a given player.

        :param player: The player whose bet is to be found.
        :return: The amount of the bet of the given player.
        """
        return self.players_bets[player]

    def add_bet(self, player, value):
        """
        Adds a bet for a specific player.

        :param player: The player making the bet.
        :param value: The amount being bet.
        """
        self.players_bets[player] += value

    def remove_cards(self, card_list):
        """
        Removes a list of cards from the available deck.

        :param card_list: List of Card objects to be removed.
        """
        deck_copy = self.aviable_cards.copy()
        for card in card_list:
            for d_card in deck_copy:
                if card.number == d_card.number and card.suit == d_card.suit:
                    self.aviable_cards.remove(d_card)
                    break

    def remove_players(self, players):
        """
        Removes players from the active players list.

        :param players: List of players to be removed.
        """
        for player in players:
            self.active_players.remove(player)

    def set_price(self, price):
        """
        Sets the price for the hand.

        :param price: The price to be set for the hand.
        """
        self.price = price

    def get_max_bet(self):
        """
        Gets the maximum bet made by any player in the hand.

        :return: Maximum bet amount.
        """
        self.max_bet = max(self.players_bets.values())
        return self.max_bet

    def get_pot(self):
        """
        Calculates the total pot value from all bets.

        :return: Total pot value.
        """
        self.pot = sum(self.players_bets.values())
        return self.pot

    def find_player(self, id):
        """
        Finds a player in the active players list by their ID.

        :param id: The ID of the player to be found.
        :return: The Player object with the matching ID.
        """
        for player in self.active_players:
            if player.id == id:
                return player

    def print_community_cards(self):
        """
        Prints the community cards to the console.
        """
        for card in self.community_cards:
            print(card.number, card.suit)

    def deal_cards(self, restriction=-1):
        """
        Deals cards to the players in a random manner.

        :param restriction: Restriction based on the round (optional).
        """
        for player in self.active_players:
            if not player.id == restriction:
                card1 = random.choice(self.aviable_cards)
                self.aviable_cards.remove(card1)
                card2 = random.choice(self.aviable_cards)
                self.aviable_cards.remove(card2)
                player.set_cards([card1, card2])

    def evaluate_hand(self, player1, player2):
        """
        Compares two players' hands to determine which one is stronger.

        :param player1: First player for comparison.
        :param player2: Second player for comparison.
        :return: Positive value if player1's hand is stronger, negative if player2's hand is stronger, zero if hands are equal.
        """
        res = player1.better_hand[0] - player2.better_hand[0]
        if res == 0:
            for c1, c2 in zip(player1.better_hand[1], player2.better_hand[1]):
                res = f.get_num_value(c2) - f.get_num_value(c1)
                if res != 0:
                    break
        return res

    def tie_breaking(self, player1, player2):
        """
        Breaks a tie between two players' hands using high cards.

        :param player1: First player involved in the tie.
        :param player2: Second player involved in the tie.
        :return: Positive value if player1's high card wins, negative if player2's high card wins, zero if still a tie.
        """
        res = 0
        for c1, c2 in zip(player1.hand, player2.hand):
            res = c1.get_value() - c2.get_value()
            if res != 0:
                break
        return res

    def get_sp_winners(self, players):
        """
        Determines the winners among a subset of players in a side pot.

        :param players: List of players eligible to win the side pot.
        :return: List of players who won the side pot.
        """
        winners = [players[0]]
        for i in range(1, len(players)):
            res = self.evaluate_hand(players[0], players[i])
            if res > 0:
                winners = [players[i]]
            elif res == 0: #Si las manos que empatan son HIGH CARD, PAIR, TOAK, DOUBLE PAIR. Entonces se desempata por carta alta de las que estén en la mano
                if winners[0].better_hand[0] in [9,8,7,6]:
                    res = self.tie_breaking(winners[0], players[i])
                    if res < 0:
                        winners = [players[i]]
                    elif res == 0:
                        winners.append(players[i])
                else:
                    winners.append(players[i])
        return winners

    def get_winners(self):
        """
        Determines the winners of the hand.

        :return: Updates the 'winners' attribute with the list of players who won the hand.
        """
        self.winners = [self.active_players[0]]
        for i in range(1, len(self.active_players)):
            res = self.evaluate_hand(self.winners[0], self.active_players[i])
            if res > 0:
                self.winners = [self.active_players[i]]
            elif res == 0: #Si las manos que empatan son HIGH CARD, PAIR, TOAK, DOUBLE PAIR. Entonces se desempata por carta alta de las que estén en la mano
                if self.winners[0].better_hand[0] in [9,8,7,6]:
                    res = self.tie_breaking(self.winners[0], self.active_players[i])
                    if res < 0:
                        self.winners = [self.active_players[i]]
                    elif res == 0:
                        self.winners.append(self.active_players[i])
                else:
                    self.winners.append(self.active_players[i])

class Game():
    def __init__(self, game_players, big_blind):
        """
        Initializes a poker game with a set of players and blinds.

        :param game_players: List of Player objects participating in the game.
        :param big_blind: The amount of the big blind for the game.
        """
        self.hands = []
        self.n_hands = 0
        self.game_players = game_players.copy()
        self.n_players = len(self.game_players)
        self.ranking = []
        self.big_blind = big_blind
        self.small_blind = big_blind/2
        self.D = self.n_players - 2

    def get_ranking(self):
        """
        Calculates and returns the ranking of players based on their performance.
        """
        pass

    def add_hand(self, hand):
        """
        Adds a completed hand to the game's history.

        :param hand: The Hand object representing a completed hand.
        """
        self.hands.append(hand)
        self.n_hands += 1

    def remove_loosers(self):
        """
        Removes players from the game who have lost all their budget.
        """
        p_copy = self.game_players.copy()
        for player in p_copy:
            if player.budget == 0:
                self.game_players.remove(player)
                self.ranking.insert(0, player)

    def set_start_blinds(self, hand):
        """
        Sets the starting small and big blinds for a hand.

        :param hand: The Hand object for which blinds are to be set.
        """
        self.game_players.insert(0, self.game_players.pop())
        sb_player = self.game_players[-1]
        bb_player = self.game_players[0]
        sb_player.apuesta_actual = self.small_blind
        bb_player.apuesta_actual = self.big_blind

        sb_player.bet(hand)
        bb_player.bet(hand)

    def play(self, print_values=False):
        """
        Plays out the poker game.

        :param print_values: Boolean flag to print details of the game while playing.
        """
        start_time = time.time()
        while len(self.game_players) > 1:

            if print_values:
                print("NEW HAND")
            stop = False
            hand = Hand(self.game_players)
            hand.deal_cards()
            self.set_start_blinds(hand)
            for player in hand.active_players:
                if print_values:
                    print(
                        f"Player {player.id} --- {player.hand[0].number} {player.hand[0].suit}, {player.hand[1].number} {player.hand[1].suit}")

            for i in range(1, 5):
                if i == 2:
                    hand.set_community_cards(3)

                elif i > 2:
                    hand.set_community_cards(1)

                if print_values:
                    print("\nCOMUNITY CARDS")
                    hand.print_community_cards()
                    print("*"*40)

                #PRIMERA RONDA DE CADA CIEGA
                if print_values:
                    print("\n---------------- FIRST ROUND ----------------\n")
                remaining_players = hand.active_players.copy()

                for player in remaining_players:
                    if len(hand.active_players) == 2 and (hand.active_players[0].budget == 0 or hand.active_players[1].budget == 0):
                        break # Si solo quedan 2 jugadores y 1 de ellos ha hecho all in, no tiene sentido seguir apostando.

                    if player.budget > 0: # Si el jugador ha hecho all in no tiene sentido preguntarle si va a apostar
                        decision = player.act(hand, self.n_hands, print_values=False)
                        if print_values:
                            print("Decision", decision)

                        if decision == 0:
                            hand.remove_players([player])
                            if len(hand.active_players) == 1:
                                stop=True
                                break
                        else:
                            player.bet(hand)
                        hand.get_max_bet()

                    if print_values:
                        print(
                            f"Player {player.id} --- {round(player.probability*100, 2)} --- {round(player.budget,2)} --- {round(hand.find_player_bet(player), 2)}")
                if stop:
                    break
                hand.get_max_bet()

                # Se volvera a realizar el player.act solamente de aquellos jugadores que aun no hayan igualado la apuesta.
                #SEGUNDA RONDA DE CADA CIEGA

                if print_values:
                    print("\n---------------- SECOND ROUND ----------------\n")
                remaining_players = hand.active_players.copy()
                for player in remaining_players:
                    if len(hand.active_players) == 2 and (hand.active_players[0].budget == 0 or hand.active_players[1].budget == 0):
                        break # Si solo quedan 2 jugadores y 1 de ellos ha hecho all in, no tiene sentido seguir apostando.

                    if (player.budget > 0 and hand.find_player_bet(player) < hand.max_bet):
                        decision = player.act(hand, self.n_hands)
                        if print_values:
                            print("Decision", decision)
                        if decision == 0:
                            hand.remove_players([player])
                            if len(hand.active_players) == 1:
                                break
                        else:
                            player.bet(hand, round=2)

                    if print_values:
                        print(
                            f"Player {player.id} --- {round(player.probability*100, 2)} --- {round(player.budget,2)} --- {round(hand.find_player_bet(player), 2)}")

                #si todos los jugadores se retiran
                if len(hand.active_players) == 1:
                    break

            #EVALUAR LOS GANADORES
            for player in hand.active_players:
                player.get_best_hand(hand, print_values=False)

            hand.get_pot()

            # REPARITMOS EL BOTE ENTRE LOS GANADORES
            if print_values:
                print("POT", hand.pot)
            old_budgets = {player: player.budget for player in self.game_players}
            hand.distribute_pot()
            for player in self.game_players:
                if isinstance(player, NotTrainedModel):
                    player.add_recompensa(player.budget - old_budgets[player])

            #ELIMINAMOS A LOS ARRUINADOS
            self.remove_loosers()

            if print_values:
                print("NEW BUDGETS: ")
                for player in self.game_players:
                    print(player.id, round(player.budget,2))
                print("\n\n")

            self.add_hand(hand)
            if self.n_hands >= 15:
                break

        final_budgets = [player.budget for player in self.game_players]
        while len(self.game_players) > 1:
            min_budget = min(final_budgets)
            self.ranking.insert(0, self.game_players.pop(final_budgets.index(min_budget)))
            final_budgets.pop(final_budgets.index(min_budget))

        self.ranking.insert(0, self.game_players[0])
        match_time = round(time.time()-start_time, 3)
        print(f"Duration: {match_time}")
        print(f"Hands played: {self.n_hands}")

class PlayGround():
    def __init__(self, n_reps, n_games, n_simulations, n_players, initial_budget, big_blind):
        """
        Initializes a playground for simulating multiple games.

        :param n_reps: Number of repetitions for each game.
        :param n_games: Number of games to be played.
        :param n_simulations: Number of simulations for decision-making in games.
        :param n_players: Number of players in each game.
        :param initial_budget: The starting budget for each player.
        :param big_blind: The big blind amount for the games.
        """
        self.n_games = n_games
        self.n_simulations = n_simulations
        self.n_players = n_players
        self.initial_budget = initial_budget
        self.big_blind = big_blind
        self.players_type = [1,2,3]
        self.n_reps = n_reps
        self.total_iterations = self.n_reps * self.n_games
        self.model = NotTrainedModel(id=1, budget=self.initial_budget, n_simulations=self.n_simulations)
        self.model_wins = 0
        self.ranking = {}
        self.models_folder = "models/"

    def generate_players(self, n_players):
        """
        Generates a list of players with different playing styles.

        :param n_players: Number of players to generate.
        """
        self.players = [self.model]

        for i in range(1,n_players+1):

            p_type = random.choice(self.players_type)
            if p_type == 1:
                player = JuegoPasivo(id=i + 1, budget=self.initial_budget, n_simulations=self.n_simulations)
            elif p_type == 2:
                player = Tight_Agresivo(id=i + 1, budget=self.initial_budget, n_simulations=self.n_simulations)
            else:
                player = Bluffing(id=i + 1, budget=self.initial_budget, n_simulations=self.n_simulations)
            self.players.append(player)

        random.shuffle(self.players)

    def train(self):
        """
        Trains the models by playing a series of games.
        """
        self.players = [NotTrainedModel(id=i+1, budget=self.initial_budget, n_simulations=self.n_simulations) for i in range(self.n_players)]
        self.ranking = {player.id: 0 for player in self.players}

        for j in range(self.total_iterations):
            print(f"Iter {j + 1}/{self.total_iterations}")
            game = Game(self.players, self.big_blind)
            game.play(print_values=False)
            self.ranking[game.ranking[0].id] += 1
            print(self.ranking)
            [player.reset_info() for player in self.players]

            if (j+1) % self.n_games == 0:
                for player in self.players:
                    player.update()
                print(f"UPDATE {j + 1}")

        for player in self.players:
            self.save_model(player)

    def test(self, model_path, num_inputs, num_actions, n_players):
        """
        Tests a trained model by playing a series of games.

        :param model_path: Path to the trained model file.
        :param num_inputs: Number of input features for the model.
        :param num_actions: Number of actions possible for the model.
        :param n_players: Number of players in the test games.
        """
        self.model = TrainedModel(id=1, budget=self.initial_budget, n_simulations=self.n_simulations, model_path=model_path, num_inputs=num_inputs, num_actions=num_actions)
        self.generate_players(n_players)
        self.ranking = {player.id: 0 for player in self.players}
        for player in self.players:
            print(player.id, player.__class__.__name__)

        for j in range(self.n_games):
            print(f"Iter {j + 1}/{self.n_games}")
            game = Game(self.players, self.big_blind)
            game.play(print_values=False)
            self.ranking[game.ranking[0].id] += 1
            print(self.ranking)
            if game.ranking[0] is self.model:
                self.model_wins += 1

            [player.reset_info() for player in self.players]

        print(f'Total wins: {self.model_wins}', f'Win rate: {self.model_wins/self.n_games}')

    def save_model(self, model):
        """
        Saves the trained model to a file.

        :param model: The model to be saved.
        """
        torch.save(model.ppo.policy.state_dict(), f'{self.models_folder}modelo{model.id}.pth')