from pieces import *

class Board:
    """
    Klasa koja implementira strukturu table.
    """

    def __init__(self, rows=20, cols=20):
        self.rows = rows  # broj redova
        self.cols = cols  # broj kolona
        self.elems = ['.',   # prazno polje
                      'bp',  # crni pijun
                      'br',  # crni top
                      'bn',  # crni konj
                      'bb',  # crni lovac
                      'bk',  # crni kralj
                      'bq',  # crna kraljica
                      'wp',  # beli pijun
                      'wr',  # beli top
                      'wn',  # beli konj
                      'wb',  # beli lovac
                      'wk',  # beli kralj
                      'wq']  # beli kraljica

        self.data = [['.'] * cols for _ in range(rows)]

        self.previous_positions = [-7, -7]

        #provera belih figura da li su koriscene u partiji
        self.kralj_beli_koriscen = False
        self.top_beli_levi_koriscen = False
        self.top_beli_desni_koriscen = False

        # provera crnih figura da li su koriscene u partiji
        self.kralj_crni_koriscen = False
        self.top_crni_levi_koriscen = False
        self.top_crni_desni_koriscen = False



    def load_from_file(self, file_path):
        """
        Ucitavanje table iz fajla.
        :param file_path: putanja fajla.
        """
        board_f = open(file_path, 'r')
        row = board_f.readline().strip('\n')
        self.data = []
        while row != '':
            self.data.append(list(row.split()))
            row = board_f.readline().strip('\n')
        board_f.close()

    def save_to_file(self, file_path):
        """
        Snimanje table u fajl.
        :param file_path: putanja fajla.
        """
        if file_path:
            f = open(file_path, 'w')
            for row in range(self.rows):
                f.write(''.join(self.data[row]) + '\n')
            f.close()

    def move_piece(self, from_row, from_col, to_row, to_col):
        """
        Pomeranje figure.
        :param from_row: prethodni red figure.
        :param from_col: prethodna kolona figure.
        :param to_row: novi red figure.
        :param to_col: nova kolona figure.
        """
        if to_row < len(self.data) and to_col < len(self.data[0]):
            t = self.data[from_row][from_col]
            self.data[from_row][from_col] = '.'
            self.data[to_row][to_col] = t

            if (from_row == 7 and from_col == 4):
                self.kralj_beli_koriscen = True
            elif (from_row == 7 and from_col == 7):
                self.top_beli_desni_koriscen = True
            elif (from_row == 7 and from_col == 0):
                self.top_beli_levi_koriscen = True
            elif (from_row == 0 and from_col == 4):
                self.kralj_crni_koriscen = True
            elif (from_row == 0 and from_col == 7):
                self.top_crni_desni_koriscen = True
            elif (from_row == 0 and from_col == 0):
                self.top_crni_levi_koriscen = True
            self.previous_positions = [to_row, to_col]

    def clear(self):
        """
        Ciscenje sadrzaja cele table.
        """
        for row in range(self.rows):
            for col in range(self.cols):
                self.data[row][col] = '.'

    def find_position(self, element):
        """
        Pronalazenje specificnog elementa unutar table.
        :param element: kod elementa.
        :returns: tuple(int, int)
        """
        for row in range(self.rows):
            for col in range(self.cols):
                if self.data[row][col] == element:
                    return row, col
        return None, None

    def determine_piece(self, row, col):
        """
        Odredjivanje koja je figura na odredjenoj poziciji na tabli.
        :param row: red.
        :param col: kolona.
        :return: objekat figure (implementacija klase Piece).
        """
        elem = self.data[row][col]
        if elem != '.':
            side = elem[0]  # da li je crni (b) ili beli (w)
            piece = elem[1]  # kod figure
            if piece == 'p':
                return Pawn(self, row, col, side)
            # TODO: dodati za ostale figure
            if piece == 'n':
                return  Knight(self,row,col,side)
            if piece == 'b':
                return  Bishop(self,row,col,side)
            if piece == 'r':
                return  Rook(self,row,col,side)
            if piece == 'q':
                return  Queen(self,row,col,side)
            if piece == 'k':
                return King(self, row, col, side)
    def rokadaM(self, color):
        """
        Mala rokada kada pozicije menjaju kralj i top sa desne strane.
        """
        if(color == 'w'):
            self.data[7][5] = 'wr'
            self.data[7][6] = 'wk'
            self.data[7][4] = '.'
            self.data[7][7] = '.'
            self.kralj_beli_koriscen = True
            self.previous_positions = [7, 6]
        else:
            self.data[0][5] = 'br'
            self.data[0][6] = 'bk'
            self.data[0][4] = '.'
            self.data[0][7] = '.'
            self.kralj_crni_koriscen = True
            self.previous_positions = [0, 6]

    def rokadaV(self, color):
        """
        Velika rokada kada pozicije menjaju kralj i top sa leve strane.
        """
        if(color == 'w'):
            self.data[7][3] = 'wr'
            self.data[7][2] = 'wk'
            self.data[7][4] = '.'
            self.data[7][0] = '.'
            self.kralj_beli_koriscen = True
            self.previous_positions = [7, 2]

        else:
            self.data[0][3] = 'br'
            self.data[0][2] = 'bk'
            self.data[0][4] = '.'
            self.data[0][0] = '.'
            self.kralj_crni_koriscen = True
            self.previous_positions = [0, 2]



    def en_passant(self, from_row, from_col, to_row, to_col):
            """
            En passant
            """

            t = self.data[from_row][from_col]
            self.data[from_row][from_col] = '.'
            self.data[to_row][to_col] = t
            self.data[from_row][to_col] = '.'





    def sah(self, side, king_position=None):
        """
            Provera da li je napadnut kralj ako se moguce pozicije protivnika poklapaju sa pozicijom kralja
                    """
        if king_position is None:
            king_position = self.find_position(str(side) + 'k')


        if side == 'w':
            napadac = 'b'
        else:
            napadac = 'w'

        for row in range(self.rows):
            for col in range(self.cols):
                if self.data[row][col] != '.' and (not self.data[row][col].startswith(side)) and self.data[row][
                    col] != napadac + 'k':
                    piece = self.determine_piece(row, col)
                    positions = piece.get_legal_moves()

                    if king_position in positions:
                        return True

        return False




    def napadnuta_pozicija(self, side,figure_postion):
        """
            Provera da li je napadnut kralj ako se moguce pozicije protivnika poklapaju sa pozicijom kralja
                    """
        if side == 'w':
            napadac = 'b'
        else:
            napadac = 'w'

        for row in range(self.rows):
            for col in range(self.cols):
                if self.data[row][col] != '.' and (not self.data[row][col].startswith(side)) and self.data[row][
                    col] != napadac + 'k':
                    piece = self.determine_piece(row, col)
                    positions = piece.get_legal_moves()

                    if figure_postion in positions:
                        return True
        return False





