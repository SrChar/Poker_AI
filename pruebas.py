import clases as cl
import random
import torch

n_players = 5
initial_budget = 100000
n_simulations = 1000
big_blind = 5000

player_type = [1,2,3,4]
player_list = []

for i in range(n_players):
    p_type = random.choice(player_type)
    if p_type == 1:
        player = cl.JuegoPasivo(id=i + 1, budget=initial_budget, n_simulations=n_simulations)
    elif p_type == 2:
        player = cl.Tight_Agresivo(id=i + 1, budget=initial_budget, n_simulations=n_simulations)
    elif p_type == 3:
        player = cl.Bluffing(id=i + 1, budget=initial_budget, n_simulations=n_simulations)
    else:
        player = cl.NotTrainedModel(id=i + 1, budget=initial_budget, n_simulations=n_simulations)
    player_list.append(player)


for j in range(10):
    for player in player_list:
        print(player.id, isinstance(player, cl.NotTrainedModel))
    game = cl.Game(player_list, big_blind)

    game.play(print_values=False)

    for player in player_list:
        if isinstance(player, cl.NotTrainedModel):
            print(len(player.recompensas), len(player.states), len(player.actions), len(player.log_probs))
            player.update()
        player.reset_info()
    print(f"UPDATE {j+1}")

# Guardar el modelo
    # Guardar el modelo
for player in player_list:
    if isinstance(player, cl.NotTrainedModel):
        torch.save(player.ppo.policy.state_dict(), f'modelo{i}.pth')

