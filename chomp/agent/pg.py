from keras.optimizers import SGD

from chomp.OnePlane import OnePlane
import numpy as np
#from fastai.vision.all import *
from chomp import kerasutil

from chomp.agent.base import Agent

class PolicyAgent(Agent):

    def __init__(self, encoder, model = None):
        self.model = model
        self.encoder = encoder
        self.collector = None


    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self.encoder.name()
        h5file['encoder'].attrs['board_width'] = self.encoder.board_width
        h5file['encoder'].attrs['board_height'] = self.encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self.model, h5file['model'])


    def set_collector(self, collector):
        self.collector = collector


    @staticmethod
    def load_policy_agent(h5file):
        model = kerasutil.load_model_from_hdf5_group(h5file['model'])
        encoder_name = h5file['encoder'].attrs['name']
        board_width = h5file['encoder'].attrs['board_width']
        board_height = h5file['encoder'].attrs['board_height']
        #encoder = OnePlane(2)
        encoder = OnePlane(board_width, board_height)
        return PolicyAgent(encoder, model)


    def clip_probs(self, original_probs):
        #Think may already be numpy array
        #original_probs = original_probs.numpy()
        min_p = 1e-5
        max_p = 1 - min_p
        clipped_probs = np.clip(original_probs, min_p, max_p)
        clipped_probs = clipped_probs/np.sum(clipped_probs)
        return clipped_probs


    def select_move(self, game_state):


        board_tensor = self.encoder.encode(game_state)

        board_tensor = np.expand_dims(board_tensor, -1)



        move_probs = (self.model.predict(board_tensor))[0]
        move_probs = self.clip_probs(move_probs)
        num_moves = self.encoder.board_height * self.encoder.board_width
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(candidates, num_moves, replace=False, p=move_probs)

        for point_idx in ranked_moves:
            move = self.encoder.decode_move_int(point_idx)
            if game_state.is_valid_move(move):
                if self.collector is not None:
                    self.collector.record_decision(state = board_tensor, action = point_idx)
                return move

    #Rewritten to write output with negative of label
    def prepare_experience_data(self, experience, board_width, board_height):
        experience_size = experience.actions.shape[0]
        target_vectors = np.zeros((experience_size, board_width * board_height))
        for i in range(experience_size):
            action = experience.actions[i]
            reward = experience.rewards[i]
            target_vectors[i][action] = reward

        return target_vectors



    def train(self, experience, lr, batch_size):
        #clipnorm=clipnorm
        self.model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=lr, clipnorm=1.0))
        target_vectors = self.prepare_experience_data(experience, self.encoder.board_width, self.encoder.board_height)
        self.model.fit(experience.states, target_vectors, batch_size=batch_size, epochs=10)

