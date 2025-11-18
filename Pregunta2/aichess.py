"""

Practica 3 d'IA: Pregunta 2
@authors: Pol Lobo & Damia Carreras

"""
import copy

import chess
import board
import numpy as np
from typing import List
from tqdm import trange

RawStateType = List[List[List[int]]]

from itertools import permutations


class Aichess():

    def __init__(self, TA, exercici, myinit=True, levelOfDrunkess=0):
        self.exercici = exercici
        self.levelOfDrunkness = levelOfDrunkess
        
        #--------------- Atributs per l'escacs ---------------#
        if myinit:
            self.chess = chess.Chess(TA, True)  # Tauler (inclou real i simulador)
        else:
            self.chess = chess.Chess([], False) # Tauler buit (inclou real i simulador)

        self.initial_position = self.getCurrentPositionBoard()  # Guardem la posicio inicial
        #-----------------------------------------------------#

        #-------------- Atributs per Q-learning --------------#
        self.qTable = None                  # q-table
        self.intermediateQTable1 = None     # q-table intermitjana 1
        self.intermediateQTable2 = None     # q-table intermitjana 2
        
        self.n_training_episodes = 250
        self.learning_rate = 0.5            # Taxa d'aprenentage (alpha)

        self.max_steps = 100                # Nombre max de pasos per episodi
        self.gamma = 0.95                   # Factor de descompte (gamma)

        self.max_epsilon = 0.95             # Probabilitat d'exploracio
        self.min_epsilon = 0.1             
        self.decay_rate = 0.01              # Taxa de decaiment de l'epsilon
        #-----------------------------------------------------#

    #----------------- METODES ESCACS -----------------#
    # Boards
    def newBoardSim(self, listPositions):
        # We create a  new boardSim
        TA = np.zeros((8, 8))
        for state in listPositions:
            TA[state[0]][state[1]] = state[2]

        self.chess.newBoardSim(TA)

    def resetBoard(self):
        # We reset the board
        TA = np.zeros((8, 8))
        for state in self.initial_position:
            TA[state[0]][state[1]] = state[2]

        self.chess.newBoard(TA)

    # State useful
    def copyPosition(self, position):
        copyPosition = []
        for piece in position:
            copyPosition.append(piece.copy())
        return copyPosition

    def getListNextPositionsW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def getListNextPositionsB(self, myState):
        self.chess.boardSim.getListNextStatesB(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    # State of pieces
    def getNextPositions(self, state):
        # Given a state, we check the next possible states
        # From these, we return a list with position, i.e., [row, column]
        if state == None:
            return None
        if state[2] > 6:
            nextStates = self.getListNextPositionsW([state])
        else:
            nextStates = self.getListNextPositionsW([state])
        nextPositions = []
        for i in nextStates:
            nextPositions.append(i[0][0:2])
        return nextPositions
    
    def getPiecePosition(self, pos, piece):
        """
        Given a list of the postion (pos) and the id of a piece, 
        return the position [row, column]
        """
        piecePosition = None
        for i in pos:
            if i[2] == piece:
                piecePosition = i
                break
        return piecePosition
    
    def getWhitePosition(self, currentPosition):
        whitePosition = []
        wkPosition = self.getPiecePosition(currentPosition, 6)
        whitePosition.append(wkPosition)
        wkPosition = self.getPiecePosition(currentPosition, 2)
        if wkPosition != None:
            whitePosition.append(wkPosition)
        return whitePosition

    def getBlackPosition(self, currentPosition):
        blackPosition = []
        bkPosition = self.getPiecePosition(currentPosition, 12)
        blackPosition.append(bkPosition)
        bkPosition = self.getPiecePosition(currentPosition, 8)
        if bkPosition != None:
            blackPosition.append(bkPosition)
        return blackPosition

    def getCurrentPositionBoard(self):
        listStates = []
        for i in self.chess.board.currentStateW:
            listStates.append(i)
        for j in self.chess.board.currentStateB:
            listStates.append(j)
        return listStates

    def getCurrentPositionSimulator(self):
        listStates = []
        for i in self.chess.boardSim.currentStateW:
            listStates.append(i)
        for j in self.chess.boardSim.currentStateB:
            listStates.append(j)

        return listStates
    
    # Cheecks and checkmates
    def isWatchedBk(self, currentState):

        self.newBoardSim(currentState)

        bkPosition = self.getPiecePosition(currentState, 12)[0:2]
        wkState = self.getPiecePosition(currentState, 6)
        wrState = self.getPiecePosition(currentState, 2)

        # If the white king has been captured, this is not a valid configuration
        if wkState is None:
            return False

        # Check all possible moves of the white king to see if it can capture the black king
        for wkPosition in self.getNextPositions(wkState):
            if bkPosition == wkPosition:
                # Black king would be in check
                return True

        if wrState is not None:
            # Check all possible moves of the white rook to see if it can capture the black king
            for wrPosition in self.getNextPositions(wrState):
                if bkPosition == wrPosition:
                    return True

        return False

    def allBkMovementsWatched(self, currentState):
        # In this method, we check if the black king is threatened by the white pieces

        self.newBoardSim(currentState)
        # Get the current state of the black king
        bkState = self.getPiecePosition(currentState, 12)
        allWatched = False

        # If the black king is on the edge of the board, all its moves might be under threat
        if bkState[0] == 0 or bkState[0] == 7 or bkState[1] == 0 or bkState[1] == 7:
            wrState = self.getPiecePosition(currentState, 2)
            whiteState = self.getWhitePosition(currentState)
            allWatched = True
            # Get the future states of the black pieces
            nextBStates = self.getListNextPositionsB(self.getBlackPosition(currentState))

            for state in nextBStates:
                newWhiteState = whiteState.copy()
                # Check if the white rook has been captured; if so, remove it from the state
                if wrState is not None and wrState[0:2] == state[0][0:2]:
                    newWhiteState.remove(wrState)
                state = state + newWhiteState
                # Move the black pieces to the new state
                self.newBoardSim(state)

                # Check if in this position the black king is not threatened; 
                # if so, not all its moves are under threat
                if not self.isWatchedBk(state):
                    allWatched = False
                    break

        # Restore the original board state
        self.newBoardSim(currentState)
        return allWatched

    def isBlackInCheckMate(self, currentState):
        if self.isWatchedBk(currentState) and self.allBkMovementsWatched(currentState):
            return True

        return False

    def isWatchedWk(self, currentState):
        self.newBoardSim(currentState)

        wkPosition = self.getPiecePosition(currentState, 6)[0:2]
        bkState = self.getPiecePosition(currentState, 12)
        brState = self.getPiecePosition(currentState, 8)

        # If the black king has been captured, this is not a valid configuration
        if bkState is None:
            return False

        # Check all possible moves for the black king and see if it can capture the white king
        for bkPosition in self.getNextPositions(bkState):
            if wkPosition == bkPosition:
                # White king would be in check
                return True

        if brState is not None:
            # Check all possible moves for the black rook and see if it can capture the white king
            for brPosition in self.getNextPositions(brState):
                if wkPosition == brPosition:
                    return True

        return False

    def allWkMovementsWatched(self, currentState):

        self.newBoardSim(currentState)
        # In this method, we check if the white king is threatened by black pieces
        # Get the current state of the white king
        wkState = self.getPieceState(currentState, 6)
        allWatched = False

        # If the white king is on the edge of the board, it may be more vulnerable
        if wkState[0] == 0 or wkState[0] == 7 or wkState[1] == 0 or wkState[1] == 7:
            # Get the state of the black pieces
            brState = self.getPieceState(currentState, 8)
            blackState = self.getBlackState(currentState)
            allWatched = True

            # Get the possible future states for the white pieces
            nextWStates = self.getListNextStatesW(self.getWhiteState(currentState))
            for state in nextWStates:
                newBlackState = blackState.copy()
                # Check if the black rook has been captured. If so, remove it from the state
                if brState is not None and brState[0:2] == state[0][0:2]:
                    newBlackState.remove(brState)
                state = state + newBlackState
                # Move the white pieces to their new state
                self.newBoardSim(state)
                # Check if the white king is not threatened in this position,
                # which implies that not all of its possible moves are under threat
                if not self.isWatchedWk(state):
                    allWatched = False
                    break

        # Restore the original board state
        self.newBoardSim(currentState)
        return allWatched

    def isWhiteInCheckMate(self, currentState):
        if self.isWatchedWk(currentState) and self.allWkMovementsWatched(currentState):
            return True
        return False
    #--------------------------------------------------#
    
    #--------------- METODES Q-LEARNING ---------------#

    def initiate_q_table(self):
        """
        Accions: (Columnes)
            (0) Moure Torre Blanca a fila 0
            (1) Moure Torre Blanca a fila 1
            (2) Moure Torre Blanca a fila 2
            (3) Moure Torre Blanca a fila 3
            (4) Moure Torre Blanca a fila 4
            (5) Moure Torre Blanca a fila 5
            (6) Moure Torre Blanca a fila 6
            (7) Moure Torre Blanca a fila 7
            (8) Moure Torre Blanca a columna 0
            (9) Moure Torre Blanca a columna 1
            (10) Moure Torre Blanca a columna 2
            (11) Moure Torre Blanca a columna 3
            (12) Moure Torre Blanca a columna 4
            (13) Moure Torre Blanca a columna 5
            (14) Moure Torre Blanca a columna 6
            (15) Moure Torre Blanca a columna 7
            (16) Moure Rei Blanc esquerra
            (17) Moure Rei Blanc diagonal superior esquerra
            (18) Moure Rei Blanc amunt
            (19) Moure Rei Blanc diagonal superior dreta
            (20) Moure Rei Blanc dreta
            (21) Moure Rei Blanc diagonal inferior dreta
            (22) Moure Rei Blanc avall
            (23) Moure Rei Blanc diagonal inferior esquerra
        
        Estats: (Files)
            (0) No hi ha check a les negres
            (1) Hi ha check a les negres
            (2) Hi ha chekmate a les negres
        """
        # Definim el numero d'accions i estats
        n_accions = 23
        n_estats = 3

        self.qTable = np.zeros((n_estats, n_accions))

        print("Q-table initialized: \n", self.qTable)

    def posToState(self, pos):
        if self.isBlackInCheckMate(pos):
            return 2
        elif self.isWatchedBk(pos):
            return 1
        else:
            return 0

    def epsilon_greedy_policy(self, state, epsilon):
        
        _, n_actions = self.qTable.shape

        random_value = np.random.uniform(0, 1)

        if random_value < epsilon:
            action = np.random.randint(n_actions)   # Explorem
        else:
            action = np.argmax(self.qTable[state])  # Explotem

        return action

    def greedy_policy(self, state):
        return np.argmax(self.qTable[state])

    def validateMove(self, originalPos, newPos):
        newRow, newColumn, piece = newPos
        
        # Posicio dins del taulell
        isValid = not (newRow < 0 or newRow > 7 or newColumn < 0 or newColumn > 7)
        
        # No es pot menjar el rei negre
        if isValid:
            bKingPos = self.getPiecePosition(self.getCurrentPositionSimulator(), 12)
            isValid = newPos[0:2] != bKingPos[0:2]

        # No es pot menjar una peca amiga
        if isValid:
            for state in self.getWhitePosition(self.getCurrentPositionSimulator()):
                if newPos[0:2] == state[0:2] and isValid:
                    isValid = False
                    break     
        
        # El rei no pot apropar-se a mes de 1 casella al rei negre
        if piece == 6 and isValid:
            fila_diff = abs(newPos[0] - bKingPos[0])
            columna_diff = abs(newPos[1] - bKingPos[1])
            if fila_diff <= 1 and columna_diff <= 1:
                isValid = False
        
        # No te sentit moure a la mateixa posicio
        if originalPos == newPos and isValid:
            isValid = False

        # El rei blanc no es pot moure a un escac
        if piece == 6 and isValid:
            tempPosition = self.copyPosition(self.getCurrentPositionSimulator())
            tempPosition.remove(originalPos)
            tempPosition.append(newPos)
            if self.isWatchedWk(tempPosition):
                isValid = False

        return isValid

    def moveSimWithAction(self, action):
        isMoved = False

        # Randomness per l'embriaguesa
        if np.random.uniform(0,1) < self.levelOfDrunkness:
            action = np.random.randint(0,self.qTable.shape[1])

        # --- Accions Torre Blanca (pieceID = 2) ---        
        if 0 <= action <= 15:
            wRookPos = self.getPiecePosition(self.getCurrentPositionSimulator(), 2)

            newRow = action if action <=7 else wRookPos[0]
            newColumn = action % 8 if action >=8 else wRookPos[1]
            
            newRookPos = [newRow, newColumn, 2]

            if self.validateMove(wRookPos, newRookPos):
                self.chess.moveSim(wRookPos, newRookPos)
                isMoved = True

        # --- Accions Rei Blanc (pieceID = 6) ---
        elif 16 <= action <= 23:
            wKingPos = self.getPiecePosition(self.getCurrentPositionSimulator(), 6)
            row, column, _ = wKingPos

            # Moviments del rei
            movs_rei = {
                16: (0, -1),   # esquerra
                17: (-1, -1),  # diagonal superior esquerra
                18: (-1, 0),   # amunt
                19: (-1, 1),   # diagonal superior dreta
                20: (0, 1),    # dreta
                21: (1, 1),    # diagonal inferior dreta
                22: (1, 0),    # avall
                23: (1, -1),   # diagonal inferior esquerra
            }

            addRow, addColumn = movs_rei[action]
            newKingPos = [row + addRow, column + addColumn, 6]

            if self.validateMove(wKingPos, newKingPos):
                self.chess.moveSim(wKingPos, newKingPos)
                isMoved = True

            return isMoved
    
        else:
            print("Acció no reconeguda:", action)

    def moureTaulellAmbAccio(self, action):
        isMoved = False

        # Randomness per l'embriaguesa
        if np.random.uniform(0,1) < self.levelOfDrunkness:
            action = np.random.randint(0,self.qTable.shape[1])

        # --- Accions Torre Blanca (pieceID = 2) ---        
        if 0 <= action <= 15:
            wRookPos = self.getPiecePosition(self.getCurrentPositionSimulator(), 2)

            newRow = action if action <=7 else wRookPos[0]
            newColumn = action % 8 if action >=8 else wRookPos[1]
            
            newRookPos = [newRow, newColumn, 2]

            if self.validateMove(wRookPos, newRookPos):
                self.chess.move(wRookPos, newRookPos)       # Movem
                self.chess.board.print_board()              # Imprimim

                isMoved = True

        # --- Accions Rei Blanc (pieceID = 6) ---
        elif 16 <= action <= 23:
            wKingPos = self.getPiecePosition(self.getCurrentPositionSimulator(), 6)
            row, column, _ = wKingPos

            # Moviments del rei
            movs_rei = {
                16: (0, -1),   # esquerra
                17: (-1, -1),  # diagonal superior esquerra
                18: (-1, 0),   # amunt
                19: (-1, 1),   # diagonal superior dreta
                20: (0, 1),    # dreta
                21: (1, 1),    # diagonal inferior dreta
                22: (1, 0),    # avall
                23: (1, -1),   # diagonal inferior esquerra
            }

            addRow, addColumn = movs_rei[action]
            newKingPos = [row + addRow, column + addColumn, 6]

            if self.validateMove(wKingPos, newKingPos):
                self.chess.move(wKingPos, newKingPos)       # Movem
                self.chess.board.print_board()              # Imprimim

                isMoved = True

            return isMoved
        else:
            print("Acció no reconeguda:", action)

    def reward_partA(self, currentState, moved):
        """
        Simple reward for part A.
        The goal is checkMate for the whites
        """
        if not moved:
            return -50
        elif currentState == 0:       # No check
            return -1
        elif currentState == 1:     # Check
            return 1
        elif currentState == 2:     # Checkmate
            return 100
    
    def reward_partB(self, currentState, moved):
        # This method calculates the heuristic value for the current state.
        # The value is initially computed from White's perspective.
        # If the 'color' parameter indicates Black, the final value is multiplied by -1.

        value = 0

        if not moved:
            value -= 50

        bkState = self.getPiecePosition(currentState, 12)  # Black King
        wkState = self.getPiecePosition(currentState, 6)   # White King
        wrState = self.getPiecePosition(currentState, 2)   # White Rook
        brState = self.getPiecePosition(currentState, 8)   # Black Rook

        filaBk, columnaBk = bkState[0], bkState[1]
        filaWk, columnaWk = wkState[0], wkState[1]

        if wrState is not None:
            filaWr, columnaWr = wrState[0], wrState[1]
        if brState is not None:
            filaBr, columnaBr = brState[0], brState[1]

        # If the black rook has been captured
        if brState is None:
            value += 50
            fila = abs(filaBk - filaWk)
            columna = abs(columnaWk - columnaBk)
            distReis = min(fila, columna) + abs(fila - columna)

            if distReis >= 3 and wrState is not None:
                filaR = abs(filaBk - filaWr)
                columnaR = abs(columnaWr - columnaBk)
                value += (min(filaR, columnaR) + abs(filaR - columnaR)) / 10

            # For White: the closer our king is to the opponent’s king, the better.
            # Subtract 7 from the king-to-king distance since 7 is the maximum distance possible on the board.
            value += (7 - distReis)

            # If the black king is against a wall, prioritize pushing him into a corner (ideal for checkmate).
            if bkState[0] in (0, 7) or bkState[1] in (0, 7):
                value += (abs(filaBk - 3.5) + abs(columnaBk - 3.5)) * 10
            # Otherwise, encourage moving the black king closer to the wall.
            else:
                value += (max(abs(filaBk - 3.5), abs(columnaBk - 3.5))) * 10

        # If the white rook has been captured.
        # The logic is similar to the previous section but with reversed (negative) values.
        if wrState is None:
            value -= 50
            fila = abs(filaBk - filaWk)
            columna = abs(columnaWk - columnaBk)
            distReis = min(fila, columna) + abs(fila - columna)

            if distReis >= 3 and brState is not None:
                filaR = abs(filaWk - filaBr)
                columnaR = abs(columnaBr - columnaWk)
                value -= (min(filaR, columnaR) + abs(filaR - columnaR)) / 10

            # For White: being closer to the opposing king is better.
            # Subtract 7 from the distance since that’s the maximum possible distance.
            value += (-7 + distReis)

            # If the white king is against a wall, penalize that position.
            if wkState[0] in (0, 7) or wkState[1] in (0, 7):
                value -= (abs(filaWk - 3.5) + abs(columnaWk - 3.5)) * 10
            # Otherwise, encourage the king to stay away from the wall.
            else:
                value -= (max(abs(filaWk - 3.5), abs(columnaWk - 3.5))) * 10

        # If the black king is in check, reward this state.
        if self.isWatchedBk(currentState):
            value += 20

        # If the white king is in check, penalize this state.
        if self.isWatchedWk(currentState):
            value -= 20

        if (self.isBlackInCheckMate(currentState)):
            value += 1000
        if (self.isWhiteInCheckMate(currentState)):
            value -= 1000    

        return value

    def doAnEpisode(self, policy, epsilon = None):
        """
        Realitza un episodi complet seguint la política donada
        Policy: 
            - True: epsilon_greeedy_policy  --> Training
            - False: greedy_policy          --> Evaluation
        """
        
        if policy:      # Training
            self.newBoardSim(self.initial_position)     # Reiniciar simulador

        else:           # Evaluating
            self.resetBoard()                           # Reiniciar taulell

        state = self.posToState(self.initial_position)  # Estat inicial
        step = 0

        # Repeat step
        for step in range(self.max_steps):
            if policy:                                  # Epsilong-greedy policy
                action = self.epsilon_greedy_policy(state, epsilon)

                moved = self.moveSimWithAction(action)          # Movem simulador
                position = self.copyPosition(self.getCurrentPositionSimulator())

            else:                                       # Greedy policy
                action = self.greedy_policy(state)
                
                moved = self.moureTaulellAmbAccio(action)       # Movem taulell
                position = self.copyPosition(self.getCurrentPositionBoard())

            newState = self.posToState(position)

            if self.exercici == "a":
                reward = self.reward_partA(newState, moved)
            elif self.exercici == "b":
                reward = self.reward_partB(position, moved)

            # Equacio de Bellman
            self.qTable[state][action] = self.qTable[state][action] + self.learning_rate * (reward + self.gamma * np.max(self.qTable[newState]) - self.qTable[state][action])

            # if not policy:
            # print("Initial Position: ", self.initial_position)
            # print("Acció: ", action)
            # print("Estat: ", state)
            # print("newState: ", newState)
            # print("Reward: ", reward)
            # print("Position: ", position)
            # print("Board: ")
            # self.chess.boardSim.print_board()
                
            # input("CONTINUA...")

            if self.isBlackInCheckMate(position):       # Si arribem a checkmate, acabem
                if (not policy):
                    print(f"Goal reached in {step+1} steps!")
                break
            else:                                       # Si no, actualitzem l'estat
                state = newState

    def train(self):
        """
        Realitza un episodi complet seguint la política donada
        Policy: 
            - True: epsilon_greeedy_policy  --> Training
            - False: greedy_policy          --> Evaluation
        """
        
        for episode in trange(self.n_training_episodes):
            epsilon = self.max_epsilon * np.exp(-self.decay_rate * episode)

            self.doAnEpisode(True, epsilon)

            # Imprimim la taula a meitat d'entrenament
            if episode == self.n_training_episodes // 2:
                self.intermediate_qtable = self.qTable.copy()
                
    def q_learning(self):
        # Iniciem q-table
        self.initiate_q_table()
        
        print("Training in process...")
        self.train()

        print(f" In episode {self.n_training_episodes // 2}, the Q-table is:\n", self.intermediate_qtable)

        print("\nFinal q-table: \n", self.qTable)
        
        print("\nEvaluating the agent...")

        self.doAnEpisode(False)

    #--------------------------------------------------#

if __name__ == "__main__":
    # Load initial positions of the pieces
    TA = np.zeros((8, 8))  

    # Configuració inicial de la practica config
    configPractica = 1

    if configPractica == 1:
        TA[7][0] = 2    
        TA[7][5] = 6 
        TA[0][5] = 12 
    elif configPractica == 2:
        TA[7][0] = 2    
        TA[7][4] = 6 
        TA[0][7] = 8
        TA[0][4] = 12 

    # Inicialitzem el board i l'imprimim
    print("Starting AI chess... ")

    # Exercici 2.a)
    aichess = Aichess(TA, "a", True)

    # Exercici 2.b)
    # aichess = Aichess(TA, "b", True)

    # Exercici 2.c)
    # aichess = Aichess(TA, "a", True, 0.01)

    print("\nPrinting board")
    aichess.chess.board.print_board()

    aichess.q_learning()