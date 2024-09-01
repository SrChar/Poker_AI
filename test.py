import clases as cl
import os

N_GAMES = 10
N_SIMULATIONS = 3492
N_PLAYERS = 5
INITIAL_BUDGET = 100000
BIG_BLIND = 5000
N_REPS = 5
MODEL_FOLDER = "models/"

for model in sorted(os.listdir(MODEL_FOLDER)):
    print("MODEL", model)
    model_path = MODEL_FOLDER + model
    pg = cl.PlayGround(N_REPS, N_GAMES, N_SIMULATIONS, N_REPS, INITIAL_BUDGET, BIG_BLIND)
    pg.test(model_path=model_path, num_inputs=5, num_actions=3, n_players=N_PLAYERS)


#model_path = "models/modelo4.pth"


