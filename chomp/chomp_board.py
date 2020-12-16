from chomp.chomp_types import Player

class Move:
	def __init__(self, row, col):
		self.row = row
		self.col = col

	def print_mov(self):
		print("(" + str(self.row) + "," + str(self.col) + ")" )


class Board():

	def __init__(self, num_rows = 2, num_cols = 2):
		#1 is board, 2 critical, 0 taken
		self.num_rows = num_rows
		self.num_cols = num_cols
		self.grid = [[1 for i in range(num_cols)] for j in range(num_rows)]

		self.grid[num_rows - 1][0] = 2



	def clone(self):
		num_rows = self.num_rows
		num_cols = self.num_cols
		new_Board = Board(num_rows, num_cols)
		#new_grid = [[1 for i in range(num_cols)] for j in range(num_rows)]
		for i in range(num_rows):
			for j in range(num_cols):
				new_Board.grid[i][j] = self.grid[i][j]

		return new_Board





	def make_move(self, move):
		for i in range(0, move.row + 1):
			for j in range(move.col, self.num_cols):
				self.grid[i][j] = 0









class GameState():
	def __init__(self, board, next_player, previous_player):
		self.board = board
		self.next_player = next_player
		self.previous_player = previous_player


	def get_winner(self):
		return self.next_player


	def apply_move(self, move):
		self.board.make_move(move)
		#self.previous_player = self.previous_player.other
		#self.next_player = self.next_player.other

	def clone(self):
		#not sure about correct enum handling here
		return GameState(self.board.clone(), self.next_player, self.previous_player)

	def is_over(self):
		return (self.board.grid[self.board.num_rows][0] == 0)

	@classmethod
	def new_game(self, row_size = 2, col_size = 2):
		board = Board(row_size, col_size)
		return GameState(board, Player.alice, Player.bob)


	def is_over(self):
		return len(self.get_valid_moves()) == 0

	def is_valid_move(self, move):

		return not (move.row < 0 or move.col < 0 or move.col >= self.board.num_cols or move.row >= self.board.num_rows or
					self.board.grid[move.row][move.col] == 0)

	def get_valid_moves(self):
		return [Move(r, c) for r in range(self.board.num_rows) for c in range(self.board.num_cols) if self.is_valid_move(Move(r, c))]


