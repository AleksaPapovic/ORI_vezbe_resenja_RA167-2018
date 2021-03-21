import copy
import random

class State(object):
    """
    Klasa koja opisuje stanje table.
    """

    def __init__(self, board, parent=None):
        """
        :param board: Board (tabla)
        :param parent: roditeljsko stanje
        :return:
        """
        self.board = board  # sahovska tabla koja opisuje trenutno stanje
        self.parent = parent  # roditeljsko stanje
        self.value = 0.  # "vrednost" stanja - racuna ga evaluaciona funkcija calculate_value()

    def generate_next_states(self, max_player):
        """
        Generise moguca sledeca stanja (table) na osnovu svih mogucih poteza (u zavisnosti koji je igrac na potezu).
        :param max_player: bool. Da li je MAX igrac (crni)?
        :return: list. Lista mogucih sledecih stanja.
        """
        if max_player:
            color = 'b'
        else:
            color = 'w'
        next_states = []
        for row in range(self.board.rows):
            for col in range(self.board.cols):
                piece = self.board.determine_piece(row, col)  # odredi koja je figura
                if piece is None:
                    continue
                # generisi za crne ako je max igrac na potezu, generisi za bele ako je min igrac na potezu
                if (max_player and piece.side == 'b') or (not max_player and piece.side == 'w'):
                    legal_moves = piece.get_legal_moves()  # svi moguci potezi za figuru
                    for legal_move in legal_moves:
                        new_board = copy.deepcopy(self.board)
                        new_board.move_piece(row, col, legal_move[0], legal_move[1])

                        # provera da li je sah

                        if not new_board.sah(color):
                            next_state = State(new_board, self)
                            next_states.append(next_state)

        # TODO 5: Izmesati listu moguca sledeca stanja (da ne budu uvek u istom redosledu)
        # random.shuffle(next_states)

        niz = next_states
        for i in range(len(niz) - 1):
            rand = random.randint(0, (len(niz) - 1))
            val1 = niz[rand]
            val2 = niz[rand - 1]

            niz[rand - 1] = val1
            niz[rand] = val2
        next_states = niz
        return next_states

    def calculate_value(self):
        """
        Evaluaciona funkcija za stanje.
        :return:
        """
        # TODO 3: Implementirati jednostavnu evaluacionu funkciju (suma vrednosti svih figura na tabli)
        for row in range(self.board.rows):
            for col in range(self.board.cols):
                piece = self.board.determine_piece(row, col)
                if piece is not None:
                    self.value += piece.get_value()
        return self.value
