import numpy as np
from chomp.chomp_board import Move


class OnePlane:

    def __init__(self, BOARD_WIDTH, BOARD_HEIGHT):
        #self.board_width, self.board_height = board_size, board_size
        self.board_width = BOARD_WIDTH
        self.board_height = BOARD_HEIGHT
        self.num_planes = 1

    def name(self):
        return 'OnePlane'

    def encode(self, game_state):
        #Test
        #print("test-v")
        #print(self.shape())
        #print(self.board_height)
        #print(self.board_width)

        board_matrix = np.zeros(self.shape())
        for i in range(self.board_height):
            for j in range(self.board_width):
                board_matrix[0, i, j] = game_state.board.grid[i][j]

        return board_matrix

    def num_points(self):
        return self.board_width * self.board_height

    def encode_move(self, move):
        return (self.board_width * (move.row )) + (move.col)

    def shape(self):
        return 1, self.board_height, self.board_width

    def decode_move(self, one_hot):
        hot_index = np.argmax(one_hot)
        row = (hot_index//self.board_width)
        col = (hot_index % self.board_width)
        decoded_move = Move(row, col)
        return decoded_move

    def decode_move_int(self, mov):
        hot_index = mov
        row = (hot_index//self.board_width)
        col = (hot_index % self.board_width)
        decoded_move = Move(row, col)
        return decoded_move


