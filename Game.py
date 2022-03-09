import pickle

def print_board(board):
    garbage = [[]]
    for i in range(9):
        if board[0][i] == 0:
            garbage[0].append(' ')
        elif board[0][i] == -1:
            garbage[0].append('O')
        else:
            garbage[0].append('X')
    board = garbage
    print("Computer (O) || User (X)\n Tic-Tac-Toe board now:")
    print("-------------")
    print("| " + board[0][0] + " | " + board[0][1] + " | " + board[0][2] + " |")
    print("-------------")
    print("| " + board[0][3] + " | " + board[0][4] + " | " + board[0][5] + " |")
    print("-------------")
    print("| " + board[0][6] + " | " + board[0][7] + " | " + board[0][8] + " |")
    print("-------------")


def check_winner(board):
    if board[0][0] == board[0][1] == board[0][2] != 0:
        return True
    if board[0][0] == board[0][3] == board[0][6] != 0:
        return True
    if board[0][0] == board[0][4] == board[0][8] != 0:
        return True
    if board[0][4] == board[0][1] == board[0][7] != 0:
        return True
    if board[0][8] == board[0][5] == board[0][2] != 0:
        return True
    if board[0][6] == board[0][4] == board[0][2] != 0:
        return True
    if board[0][3] == board[0][4] == board[0][5] != 0:
        return True
    if board[0][8] == board[0][7] == board[0][6] != 0:
        return True
    return False


def gameplay(play_model, board, turn):
    isWinner = False
    for i in range(9):
        if turn:
            move = play_model.predict(board)
            print_board(board)
            board[0][int(move[0])] = -1

            if check_winner(board):
                isWinner = True
                print_board(board)
                print("You Lose :(")
                break
        else:
            print_board(board)
            move = int(input("Your turn! Select index to play (0-8):"))

            while (board[0][move] != 0):

                move = int(input("Invalid!!\nEnter valid index (0-8):"))
            board[0][move] = 1

            if check_winner(board):
                isWinner = True
                print_board(board)
                print("You Win :)")
                break
        turn = not turn

    if not isWinner:
        print("Draw :|")


def main():
    model = pickle.load(open('trained_model.mdl', 'rb'))
    tictactoe_board = [[0, 0, 0, 0, 0, 0, 0, 0, 0]]
    inp = int(input("===Who should start ===\n"
                    "1. Computer\n"
                    "2. You\n"
                    "Enter:"))
    gameplay(model, tictactoe_board, inp == 1)


if __name__ == "__main__":
    main()
