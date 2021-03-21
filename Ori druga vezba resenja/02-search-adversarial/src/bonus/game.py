from __future__ import print_function

import tkinter as tk
from tkinter import filedialog
import os
import copy
import time
from PIL import Image, ImageTk  # pip install --upgrade Pillow==3.1.1

from board import Board
from state import *
from search import *


def load_board_from_file(filename=None):
    if filename is None:
        filename = filedialog.askopenfilename(defaultextension='.brd',
                                                filetypes=(('board files', '*.brd'), ('All files', '*.*')))
    board.load_from_file(filename)
    return filename


def save_board_to_file():
    filename = filedialog.asksaveasfilename(defaultextension='.brd',
                                              filetypes=(('board files', '*.brd'), ('All files', '*.*')))
    board.save_to_file(filename)


def load_board(from_file=None):      # filename passed when reopening (resetting) same file
    load_board_from_file(from_file)
    display_board()


def reset():
    global history, board ,sah
    # while len(history) > 1:
    #   board = history[1]
    #    history = history[0:-1]
    #   display_board()
    load_board('boards/board.brd')
    history =[copy.deepcopy(board)]
    # restart svih koriscenja
    board.kralj_beli_koriscen = False
    board.top_beli_levi_koriscen = False
    board.top_beli_desni_koriscen = False
    board.kralj_crni_koriscen = False
    board.top_crni_levi_koriscen = False
    board.top_crni_desni_koriscen = False
    sah = False
sah = False
selected = None
moves = None
start, end = time.perf_counter(), None


def replace_piece(figura):
    global  board
    for i in range(7):
        if board.data[0][i] == "wp":
            if figura.startswith('w'):
                board.data[0][i] = figura
                update_board(0, i)
        elif board.data[7][i] == "bp":
            if figura.startswith('b'):
                board.data[7][i] = figura
                update_board(7,i)

def replace():
    newWindow = tk.Toplevel()
    icon1 = icons[board_to_icons["wq"]]
    icon2 = icons[board_to_icons["wr"]]
    icon3 = icons[board_to_icons["wb"]]
    icon4 = icons[board_to_icons["wn"]]
    btn1 = tk.Button(newWindow, text='Click me !',image= icon1, bd='5',
                 command= lambda :replace_piece("wq"))
    btn2 = tk.Button(newWindow, text='Click me !', image=icon2, bd='5',
                     command= lambda :replace_piece("wr"))
    btn3 = tk.Button(newWindow, text='Click me !', image=icon3, bd='5',
                     command= lambda :replace_piece("wb"))
    btn4 = tk.Button(newWindow, text='Click me !', image=icon4, bd='5',
                     command= lambda :replace_piece("wn"))

    icon5 = icons[board_to_icons["bq"]]
    icon6 = icons[board_to_icons["br"]]
    icon7 = icons[board_to_icons["bb"]]
    icon8 = icons[board_to_icons["bn"]]
    btn5 = tk.Button(newWindow, text='Click me !', image=icon5, bd='5',
                     command= lambda : replace_piece("bq"))
    btn6 = tk.Button(newWindow, text='Click me !', image=icon6, bd='5',
                     command= lambda :replace_piece("br"))
    btn7 = tk.Button(newWindow, text='Click me !', image=icon7, bd='5',
                     command= lambda :replace_piece("bb"))
    btn8 = tk.Button(newWindow, text='Click me !', image=icon8, bd='5',
                     command= lambda :replace_piece("bn"))

    # Set the position of button on the top of window.
    btn1.pack()
    btn2.pack()
    btn3.pack()
    btn4.pack()
    btn5.pack()
    btn6.pack()
    btn7.pack()
    btn8.pack()
    # sets the title of the
    # Toplevel widget
    newWindow.title("New Window")

def move_piece(event, row=None, col=None):
    global selected, moves, board, history, start, end ,sah
    if sah is not True:
        if row is None and col is None:
            cx = event.x
            cy = event.y
            col = cx // cell_size  # column
            row = cy // cell_size  # row

        if selected is None:
            # select piece
            if board.data[row][col] != '.':
                # determine which piece is selected
                piece = board.determine_piece(row, col)
                if piece is None:
                    return
                selected = row, col
                # generate legal moves
                moves = piece.get_legal_moves()
                for move in moves:
                    update_board(move[0], move[1])
                update_board(row, col)
        else:
            # move piece
            from_row, from_col = selected
            if (row, col) != selected and (row, col) not in moves:
                return
            selected = None
            moves_ = copy.copy(moves)
            moves = None
            for move in moves_:
                update_board(move[0], move[1])

            if from_row != row or from_col != col:
                history.append(copy.deepcopy(board))
                if (from_row == 7 and from_col == 4 and row == 7 and col == 6
                        and board.data[7][4].endswith('k') and board.data[7][7].endswith('r')):
                    board.rokadaM(board.data[7][4][0])
                    update_board(7, 4)
                    update_board(7, 5)
                    update_board(7, 6)
                    update_board(7, 7)
                elif (from_row == 7 and from_col == 4 and row == 7 and col == 2
                      and board.data[7][4].endswith('k') and board.data[7][0].endswith('r')):
                    board.rokadaV(board.data[7][4][0])
                    update_board(7, 0)
                    update_board(7, 2)
                    update_board(7, 3)
                    update_board(7, 4)
                elif (board.data[from_row][from_col].endswith('p') and board.data[row][col] == '.'
                      and abs(row - from_row) == 1 and (abs(row - from_row) + abs(col - from_col)) == 2):
                    # ako pijun ide po dijagonali na prazno polje to je znak da je en passant
                    board.en_passant(from_row, from_col, row, col)
                    update_board(from_row, col)
                elif (from_row == 0 and from_col == 4 and row == 0 and col == 6
                            and board.data[0][4].endswith('k') and board.data[0][7].endswith('r')):
                        board.rokadaM(board.data[0][4][0])
                        update_board(from_row, col)
                        update_board(0, 4)
                        update_board(0, 5)
                        update_board(0, 6)
                        update_board(0, 7)
                elif (from_row == 0 and from_col == 4 and row == 0 and col == 2
                      and board.data[0][4].endswith('k') and board.data[0][0].endswith('r')):
                    board.rokadaV(board.data[0][4][0])
                    update_board(0, 0)
                    update_board(0, 2)
                    update_board(0, 3)
                    update_board(0, 4)
                else:
                    board.move_piece(from_row, from_col, row, col)

            update_board(from_row, from_col)
            update_board(row, col)

            end = time.perf_counter()
            duration = end - start
            print('--- {0} was thinking for {1} seconds ---'.format('White', duration))

            # --------------------------------
            # AI (black player) should do a move here
            # --------------------------------
            if from_row != row or from_col != col:
                start = time.perf_counter()
                search = Minimax(board, 1)  # ovde promeniti koji se algoritam koristi i koja je dubina pretrage
                next_state = search.perform_adversarial_search()  # izvrsi pretragu
                end = time.perf_counter()
                duration = end - start
                print('--- {0} was thinking for {1} seconds ---'.format('Black', duration))
                if next_state is None:
                    print('--- {0} has no moves ---'.format('Black'))
                    sah = True
                else:
                    board = next_state.board
                    display_board()
                    start = time.perf_counter()
    else:
        print("sah-mat")


def update_board(row, col):
    global selected
    data = board.data
    delete_elems(row, col)
    elem = data[row][col]

    if selected is not None:
        color = 'salmon'
        if (row, col) in moves:
            color = 'light green'
    else:
        color = 'white' if (row + col) % 2 == 0 else 'gray'
    draw_rectangle(row, col, color)

    if elem in board_to_icons:
        icon = icons[board_to_icons[elem]]
        draw_icon(row, col, icon)


def get_cell_rectangle(row, col):
    return col * cell_size, row * cell_size, (col + 1) * cell_size, (row + 1) * cell_size


def draw_rectangle(row, col, color, width=1):
    rect = get_cell_rectangle(row, col)
    elem_id = canvas.create_rectangle(rect, width=width, fill=color, outline='gray')
    save_elem_id(elem_id, row, col)


def draw_icon(row, col, icon):
    rect = get_cell_rectangle(row, col)
    elem_id = canvas.create_image(rect[0]+2, rect[1]+2, image=icon, anchor=tk.NW)
    canvas.icons[elem_id] = icon
    save_elem_id(elem_id, row, col)


def save_elem_id(elem_id, row, col):
    if len(grid_elem_ids[row][col]) == 0:
        grid_elem_ids[row][col] = []
    grid_elem_ids[row][col].append(elem_id)


def delete_elems(row, col):
    if 0 <= row < rows and 0 <= col < cols:
        for elem_id in grid_elem_ids[row][col]:
            canvas.delete(elem_id)
            if elem_id in canvas.icons:
                del canvas.icons[elem_id]
        grid_elem_ids[row][col] = []


def display_board():
    canvas.delete(tk.ALL)
    for row in range(len(board.data)):
        for col in range(len(board.data[0])):
            update_board(row, col)


def make_menu(win):
    top = tk.Menu(win)  # win=top-level window
    win.config(menu=top)  # set its menu option
    file_menu = tk.Menu(top)
    file_menu.add_command(label='Open...', command=load_board, underline=0)
    file_menu.add_command(label='Save...', command=lambda: save_board_to_file(), underline=0)
    file_menu.add_command(label='Quit', command=sys.exit, underline=0)
    top.add_cascade(label='File', menu=file_menu, underline=0)
    edit = tk.Menu(top, tearoff=False)
    edit.add_separator()
    top.add_cascade(label='Edit', menu=edit, underline=0)


def undo():
    global history, board,sah
    if len(history) > 1:
        board = history[-1]
        history = history[0:-1]
        display_board()
    sah = False

#  main program #
rows = 8  # broj redova table
cols = 8  # broj kolona table
cell_size = 80  # velicina celije

board = Board(rows=rows, cols=cols)

history = [copy.deepcopy(board)]

grid_elem_ids = [[[]] * cols for _ in range(rows)]

# mapiranje sadrzaja table na boju celije
board_to_colors = {'.': 'white',
                   'w': 'gray',
                   'g': 'orangered'}
# mapiranje sadrzaja table na ikonicu
board_to_icons = {'bp': 'Chess_pdt60.png',
                  'br': 'Chess_rdt60.png',
                  'bn': 'Chess_ndt60.png',
                  'bb': 'Chess_bdt60.png',
                  'bk': 'Chess_kdt60.png',
                  'bq': 'Chess_qdt60.png',
                  'wp': 'Chess_plt60.png',
                  'wr': 'Chess_rlt60.png',
                  'wn': 'Chess_nlt60.png',
                  'wb': 'Chess_blt60.png',
                  'wk': 'Chess_klt60.png',
                  'wq': 'Chess_qlt60.png'}


root = tk.Tk()
root.title('ORI - Pretrage')
make_menu(root)  # make window menu
ui = tk.Frame(root, bg='white')  # main UI
ui2 = tk.Frame(root, bg='white')

# define the user interaction widgets
canvas = tk.Canvas(root, width=cols * cell_size + 1, height=rows * cell_size + 1,
                   highlightthickness=0, bd=0, bg='white')

# load icons
canvas.icons = dict()
icons = dict()
for f in os.listdir('icons'):
    icon = Image.open(os.path.join('icons', f))
    icon = icon.resize((cell_size - 2, cell_size - 2), Image.ANTIALIAS)  # resize icon to fit cell
    icon = ImageTk.PhotoImage(icon)
    icons[f] = icon

# create buttons
restart_button = tk.Button(ui, text='RESET', width=10, command=reset)
undo_button = tk.Button(ui, text='UNDO', width=10, command=undo)
replace_butoon = tk.Button(ui, text='REPLACE', width=10, command=replace)
# add buttons to UI
restart_button.grid(row=4, column=0, padx=10, pady=10)
undo_button.grid(row=5, column=0, padx=10, pady=10)
replace_butoon.grid(row=6, column=0, padx=10, pady=10)
# put everything on the screen
display_board()
canvas.bind('<Button-1>', move_piece)  # bind left mouse click event to function switch_cell
ui.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.BOTH)
canvas.pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH)
ui2.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH, anchor=tk.W)

# load default board
load_board('boards/board.brd')

root.mainloop()
