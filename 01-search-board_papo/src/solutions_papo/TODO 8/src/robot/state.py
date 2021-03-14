from abc import *
from collections import deque


class State(object):
    """
    Apstraktna klasa koja opisuje stanje pretrage.
    """

    @abstractmethod
    def __init__(self, board, parent=None, position=None, goal_position=None, old_goal_position=None):
        """
        :param board: Board (tabla)
        :param parent: roditeljsko stanje
        :param position: pozicija stanja
        :param goal_position: pozicija krajnjeg stanja
        :return:
        """

        self.board = board
        self.parent = parent  # roditeljsko stanje
        if self.parent is None:  # ako nema roditeljsko stanje, onda je ovo inicijalno stanje
            self.position = board.find_position(self.get_agent_code())  # pronadji pocetnu poziciju
            self.goal_position = board.find_position(self.get_agent_goal_code())  # pronadji krajnju poziciju
            self.old_goal_position = self.goal_position


        else:  # ako ima roditeljsko stanje, samo sacuvaj vrednosti parametara
            self.position = position
            self.goal_position = goal_position
            self.old_goal_position = old_goal_position

        self.depth = parent.depth + 1 if parent is not None else 1  # povecaj dubinu/nivo pretrage

    def get_next_states(self):
        new_positions = self.get_legal_positions()  # dobavi moguce (legalne) sledece pozicije iz trenutne pozicije
        next_states = []
        # napravi listu mogucih sledecih stanja na osnovu mogucih sledecih pozicija
        for new_position in new_positions:
            next_state = self.__class__(self.board, self, new_position, self.goal_position, self.food,
                                        self.old_goal_position)
            next_states.append(next_state)
        return next_states

    @abstractmethod
    def get_agent_code(self):
        """
        Apstraktna metoda koja treba da vrati kod agenta na tabli.
        :return: str
        """
        pass

    @abstractmethod
    def get_agent_goal_code(self):
        """
        Apstraktna metoda koja treba da vrati kod agentovog cilja na tabli.
        :return: str
        """
        pass

    @abstractmethod
    def get_legal_positions(self):
        """
        Apstraktna metoda koja treba da vrati moguce (legalne) sledece pozicije na osnovu trenutne pozicije.
        :return: list
        """
        pass

    @abstractmethod
    def is_final_state(self):
        """
        Apstraktna metoda koja treba da vrati da li je treuntno stanje zapravo zavrsno stanje.
        :return: bool
        """
        pass

    def is_pfood_state(self):
        pass

    @abstractmethod
    def unique_hash(self):
        """
        Apstraktna metoda koja treba da vrati string koji je JEDINSTVEN za ovo stanje
        (u odnosu na ostala stanja).
        :return: str
        """
        pass

    @abstractmethod
    def get_cost(self):
        """
        Apstraktna metoda koja treba da vrati procenu cene
        (vrednost heuristicke funkcije) za ovo stanje.
        Koristi se za vodjene pretrage.
        :return: float
        """
        pass

    @abstractmethod
    def get_current_cost(self):
        """
        Apstraktna metoda koja treba da vrati stvarnu trenutnu cenu za ovo stanje.
        Koristi se za vodjene pretrage.
        :return: float
        """
        pass

    def get_pfood_code(self):
        pass

    def get_zfood_code(self):
        pass

    def change_goal(self):
        pass


class RobotState(State):
    def __init__(self, board, parent=None, position=None, goal_position=None,old_goal_position=None,niz=None):
        super(self.__class__, self).__init__(board, parent, position, goal_position, old_goal_position)
        # posle pozivanja super konstruktora, mogu se dodavati "custom" stvari vezani za stanje
        # TODO 6: prosiriti stanje sa informacijom da li je robot pokupio kutiju

        if self.parent is None:  # ako nema roditeljsko stanje, onda je ovo inicijalno stanje
            self.niz = deque([])
            for rows in range(board.rows):
                for cols in range(board.cols):
                    if board.data[rows][cols] == 'b':
                        print(rows, cols)
                        self.niz.append((rows, cols))
            self.final_goal = board.find_position(self.get_agent_goal_code())

            # b, b = self.box_position[0]
            if len(self.niz) <= 0:
                self.goal_position = board.find_position(self.get_agent_goal_code())
            else:
                self.goal_position = self.niz[0]
        else:
            self.niz = niz
            self.final_goal = old_goal_position
            self.goal_position = goal_position



    def get_next_states(self):
        new_positions = self.get_legal_positions()  # dobavi moguce (legalne) sledece pozicije iz trenutne pozicije
        next_states = []
        # napravi listu mogucih sledecih stanja na osnovu mogucih sledecih pozicija
        for new_position in new_positions:
            next_state = self.__class__(self.board, self, new_position, self.goal_position,self.old_goal_position, self.niz )
            next_states.append(next_state)
        return next_states

    def get_agent_code(self):
        return 'r'

    def get_agent_goal_code(self):
        return 'g'

    def get_pfood_code(self):
        return 'b'

    def get_zfood__code(self):
        return 'z'

    def get_legal_positions(self):
        # d_rows (delta rows), d_cols (delta columns)
        # moguci smerovi kretanja robota (desno, levo, dole, gore)
        # d_rows = [0, 0, 1, -1]
        # d_cols = [1, -1, 0, 0]

        d_rows = [0, 0, 1, -1, 1, -1, -1, 1]
        d_cols = [1, -1, 0, 0, 1, 1, -1, -1]

        row, col = self.position  # trenutno pozicija
        new_positions = []
        for d_row, d_col in zip(d_rows, d_cols):  # za sve moguce smerove
            new_row = row + d_row  # nova pozicija po redu
            new_col = col + d_col  # nova pozicija po koloni
            # ako nova pozicija nije van table i ako nije zid ('w'), ubaci u listu legalnih pozicija
            if 0 <= new_row < self.board.rows and 0 <= new_col < self.board.cols and self.board.data[new_row][
                new_col] != 'w':
                new_positions.append((new_row, new_col))
        return new_positions

    def is_final_state(self):
        return self.position == self.goal_position

    def is_pfood_state(self):
        # if self.position == self.bfood_position:
        print("pre izmene")
        print(self.position)

        print(self.goal_position)
        rp, cp = self.position
        if (self.position == self.goal_position and self.board.data[rp][cp] == 'b'):
            print("hrana")
            self.goal_position = self.old_goal_position
            return True
        return False

    def change_goal(self):
        self.goal_position = self.board.find_position(self.get_agent_goal_code())
        print("izmena")
        print(self.position)
        print(self.food)

    def unique_hash(self):
        return str(self.position)

    def get_current_cost(self):
        return self.depth

    def get_cost(self):
        sr, sc = self.position
        gr, gc = self.goal_position
        return ((gr - sr) ** 2 + (gc - sc) ** 2) ** 0.5

    def get_cost_m(self):
        sr, sc = self.position
        gr, gc = self.goal_position
        return abs(gr - sr) + abs(gc - sc)

    def change(self):

        if len(self.niz) > 0:
            self.position = self.niz.popleft()

            if len(self.niz) > 0:
                self.goal_position = self.niz[0]
            else:
                self.goal_position = self.old_goal_position
        else:
            self.goal_position = self.old_goal_position