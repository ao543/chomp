import os
import random

import h5py

from chomp.OnePlane import OnePlane
from chomp.chomp_board import GameState
from chomp.chomp_types import Player
from chomp.agent.pg import PolicyAgent
from chomp.rl.experience import ExperienceCollector, combine_experience, load_experience
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Conv2D, Flatten, MaxPooling2D
from chomp.agent.pg import PolicyAgent
from tensorflow import set_random_seed

def simulate_game(alice, bob, BOARD_WIDTH, BOARD_HEIGHT, first_move=False):
    #black= 1st, white = second
    #BOARD_SIZE = 2
    game = GameState.new_game(row_size = BOARD_HEIGHT, col_size = BOARD_WIDTH)
    agents = {Player.alice: alice, Player.bob: bob}

    #Test
    #print("test")
    #print(game.board.grid)
    i = 0

    while not game.is_over():

        #test
        #print(game.next_player)

        if i % 2 == 0:
            next_move = agents[Player.alice].select_move(game)
        else:
            next_move = agents[Player.bob].select_move(game)

        #if i == 0:
            #first_move = next_move
        #game = game.apply_move(next_move)
        game.apply_move(next_move)
        i = i + 1

    if first_move:
        return game.get_winner(), first_move

    if i % 2 == 0:
        return Player.alice
    else:
        return Player.bob
    #return game.get_winner()

def base_model(BOARD_WIDTH, BOARD_HEIGHT):

    input_shape = (BOARD_HEIGHT, BOARD_WIDTH,  1)
    model = Sequential()
    model.add(Conv2D(48, kernel_size=(3,3),  activation='relu', padding = 'same',input_shape = input_shape ) )
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(48, kernel_size=(3,3), padding='same', activation='relu') )
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(BOARD_WIDTH * BOARD_HEIGHT, activation='softmax'))
    #model.summary()
    return model

def generate_experience(iteration, BOARD_WIDTH, BOARD_HEIGHT, num_games = 1000):
    #Test filename
    # bots = {Player.alice: naive.RandomBot(), Player.bob: naive.RandomBot()}

    if iteration != 0:
        agent_filename = 'agent' + str(iteration - 1) + '.hdf5'
        f = h5py.File(agent_filename, 'a')
        agent1 = PolicyAgent.load_policy_agent(f)
        agent2 = PolicyAgent.load_policy_agent(f)
        #agent2 = naive.RandomBot()
    else:
        game_encoder = OnePlane(BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT)
        agent1 = PolicyAgent(game_encoder, base_model(BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT))
        agent2 = PolicyAgent(game_encoder, base_model(BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT))


    collector1 = ExperienceCollector()
    collector2 = ExperienceCollector()

    agent1.set_collector(collector1)
    agent2.set_collector(collector2)

    for i in range(num_games):
        collector1.begin_episode()
        collector2.begin_episode()

        #Test
        #print("lion")
        #agent1.model.summary()

        game_record = simulate_game(agent1, agent2, BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT)

        if game_record == Player.alice:
            collector1.complete_episode(reward = 1)
            collector2.complete_episode(reward=-1)
        else:
            collector2.complete_episode(reward = 1)
            collector1.complete_episode(reward=-1)

    #Test
    #print('hello')
    #print(collector1.states)
    #print(len(collector1.states))
    #print(len(collector1.states[1]))

    experience = combine_experience([collector1, collector2])
    experience_filename = 'experience' + str(iteration) + '.hdf5'
    with h5py.File(experience_filename, 'w') as experience_outf:
        experience.serialize(experience_outf)

def learn_from_experience(iteration, BOARD_WIDTH, BOARD_HEIGHT):

    #Sets defaults here
    batch_size = 32
    #batch_size = 224
    #learning_rate = .00001
    learning_rate = .0001
    #learning_rate = .001


    clipnorm = 1.0
    updated_agent_filename = 'agent' + str(iteration) + '.hdf5'
    exp_filename = '/Users/andrew/Desktop/obrien_rl_project/chomp_proj/chomp/agent/experience' + str(iteration) +'.hdf5'

    if iteration == 0:
        learning_agent = PolicyAgent(OnePlane(BOARD_WIDTH, BOARD_HEIGHT), base_model(BOARD_WIDTH, BOARD_HEIGHT))
    else:
        agent_filename = 'agent' + str(iteration - 1) + '.hdf5'
        learning_agent = PolicyAgent.load_policy_agent(h5py.File(agent_filename))
    #for exp_filename in experience_files:
    expr_file = h5py.File(exp_filename)
    #Test
    #print("green")
    #print(expr_file)

    exp_buffer = load_experience(expr_file)

    #print("vegetables")
    #print(exp_buffer.states.shape)
    exp_buffer.states = np.squeeze(exp_buffer.states, axis = 1)
    #print(exp_buffer.states.shape)


    learning_agent.train(exp_buffer, lr=learning_rate, batch_size=batch_size)
    with h5py.File(updated_agent_filename, 'w') as updated_agent_outf:
        learning_agent.serialize(updated_agent_outf)

def compute_self_play_stats(iteration, BOARD_WIDTH, BOARD_HEIGHT,  num_games = 10000):
    agent_filename = 'agent' + str(iteration) + '.hdf5'


    if iteration == 0:
        learning_agent = PolicyAgent(OnePlane(BOARD_WIDTH, BOARD_HEIGHT), base_model(BOARD_WIDTH, BOARD_HEIGHT))
    else:
        learning_agent = PolicyAgent.load_policy_agent(h5py.File(agent_filename))

    win_record = {Player.alice: 0, Player.bob: 0}
    first_move_count = 0

    for i in range(num_games):

        game_record = simulate_game(learning_agent,learning_agent, BOARD_WIDTH = BOARD_WIDTH,
                                                BOARD_HEIGHT = BOARD_HEIGHT, first_move=False)

        if game_record == Player.alice:
            win_record[Player.alice] += 1
        else:
            win_record[Player.bob] += 1

        #if (BOARD_HEIGHT - 2 == first_move.row and first_move.col ==1):
            #first_move_count = first_move_count + 1

    #print("test")
    #print(win_record[Player.alice])
    #print(win_record[Player.bob])

    return (win_record[Player.alice], win_record[Player.bob])
    #, first_move_count
    #print("Alice wins: " + str(win_record[Player.alice]))
    #print("Bob wins: " + str(win_record[Player.bob]))

def learning_cycle(BOARD_WIDTH, BOARD_HEIGHT, cycles, output_file = 'exp1.txt'):
    results = {}

    for i in range(cycles):
        if i == 0:
            generate_experience(BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT, iteration = i)
            learn_from_experience(BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT, iteration = i)
            results[i] = compute_self_play_stats(iteration = i, BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT)
        else:
            generate_experience(BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT, iteration = i)
            learn_from_experience(BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT, iteration = i)
            results[i] = compute_self_play_stats(iteration = i, BOARD_WIDTH = BOARD_WIDTH, BOARD_HEIGHT = BOARD_HEIGHT)

    print(results)
    #for i in range(len(results)):
        #print(str(i) + ": ", end="")
        #(results[i][2]).print_mov()


    myfile = output_file
    with open(myfile, 'w') as f:
        #for key, value in a.items():
        f.write(str(results))



def clean_directory():
    path = '/Users/andrew/Desktop/obrien_rl_project/chomp_proj/chomp/agent'
    dir = os.listdir(path)
    i = 0
    for file in dir:
        if file.endswith('hdf5'):
            os.remove(file)

def experiment_1():
    #Tests for random convergence
    dim_list = [(random.randint(1,10), random.randint(1,10)), (random.randint(1,10),random.randint(1,10)),(random.randint(1,10), random.randint(1,10))]

    dim1 = dim_list[0]
    dim2 = dim_list[1]
    dim3 = dim_list[2]
    #print(dim1)
    #print(dim2)
    #print(dim3)
    f = open("exp1.txt", "r+")
    f.truncate()
    learning_cycle(BOARD_WIDTH= dim1[0], BOARD_HEIGHT=dim1[1], cycles=2, output_file='exp1.txt')
    clean_directory()
    learning_cycle(BOARD_WIDTH=dim2[0], BOARD_HEIGHT=dim2[1], cycles=2, output_file='exp1.txt')
    clean_directory()
    learning_cycle(BOARD_WIDTH=dim3[0], BOARD_HEIGHT=dim3[1], cycles=2, output_file='exp1.txt')


if __name__ == '__main__':
    random.seed(34)
    np.random.seed(34)
    set_random_seed(34)
    #learning_cycle(BOARD_WIDTH=4, BOARD_HEIGHT=5, cycles=50)
    #learning_cycle(BOARD_WIDTH=2, BOARD_HEIGHT=2, cycles=50)
    #learning_cycle(BOARD_WIDTH=2, BOARD_HEIGHT=2, cycles=2, output_file = 'exp1.txt')
    #clean_directory()
    experiment_1()
    #f = open("exp1.txt", "r+")
    #f.truncate()
