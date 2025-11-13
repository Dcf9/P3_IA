"""

Practica 3 d'IA: Pregunta 2
@authors: Pol Lobo & Damia Carreras

"""
import copy

import chess
import board
import numpy as np
from typing import List

RawStateType = List[List[List[int]]]

from itertools import permutations


class Aichess():

    def __init__(self, TA, myinit = True):
        #--------------- Atributs per l'escacs ---------------#
        if myinit:
            self.initial_chess = chess.Chess(TA, True)  # Guardem la posicio inicial
            self.chess = chess.Chess(TA, True)
        else:
            self.initial_chess = chess.Chess([], False) # Guardem la posicio inicial
            self.chess = chess.Chess([], False)
        #-----------------------------------------------------#

        #-------------- Atributs per Q-learning --------------#
        self.qTable = None                  # q-table
        self.stateToPos = {}                
        self.posToState = {}
        
        self.n_training_episodes = 1000
        self.learning_rate = 0.5            # Taxa d'aprenentage (alpha)

        self.max_steps = 100                # Nombre max de pasos per episodi
        self.gamma = 0.95                   # Factor de descompte (gamma)

        self.max_epsilon = 0.95             # Probabilitat d'exploracio
        self.min_epsilon = 0.1             
        self.decay_rate = 0.01              # Taxa de decaiment de l'epsilon
        #-----------------------------------------------------#

    #----------------- METODES ESCACS -----------------#
    # Board Simulator
    def newBoardSim(self, listStates):
        # We create a  new boardSim
        TA = np.zeros((8, 8))
        for state in listStates:
            TA[state[0]][state[1]] = state[2]

        self.chess.newBoardSim(TA)

    # State useful
    def copyState(self, state):
        copyState = []
        for piece in state:
            copyState.append(piece.copy())
        return copyState
    
    def isSameState(self, a, b):

        isSameState1 = True
        # a and b are lists
        for k in range(len(a)):

            if a[k] not in b:
                isSameState1 = False

        isSameState2 = True
        # a and b are lists
        for k in range(len(b)):

            if b[k] not in a:
                isSameState2 = False

        isSameState = isSameState1 and isSameState2
        return isSameState
    
    def isVisited(self, mystate):

        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedStates)):

                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True

            return isVisited
        else:
            return False

    # State of pieces
    def getPieceState(self, state, piece):
        pieceState = None
        for i in state:
            if i[2] == piece:
                pieceState = i
                break
        return pieceState
    
    def getWhiteState(self, currentState):
        whiteState = []
        wkState = self.getPieceState(currentState, 6)
        whiteState.append(wkState)
        wrState = self.getPieceState(currentState, 2)
        if wrState != None:
            whiteState.append(wrState)
        return whiteState

    def getBlackState(self, currentState):
        blackState = []
        bkState = self.getPieceState(currentState, 12)
        blackState.append(bkState)
        brState = self.getPieceState(currentState, 8)
        if brState != None:
            blackState.append(brState)
        return blackState

    def getCurrentState(self):
        listStates = []
        for i in self.chess.board.currentStateW:
            listStates.append(i)
        for j in self.chess.board.currentStateB:
            listStates.append(j)
        return listStates

    def changeStateSim(self, start, to):
        # Determine which piece has moved from the start state to the next state
        if start[0] == to[0]:
            movedPieceStart = 1
            movedPieceTo = 1
        elif start[0] == to[1]:
            movedPieceStart = 1
            movedPieceTo = 0
        elif start[1] == to[0]:
            movedPieceStart = 0
            movedPieceTo = 1
        else:
            movedPieceStart = 0
            movedPieceTo = 0

        # Move the piece that changed
        self.chess.moveSim(start[movedPieceStart], to[movedPieceTo])

    # Next states
    def getNextPositions(self, state):
        # Given a state, we check the next possible states
        # From these, we return a list with position, i.e., [row, column]
        if state == None:
            return None
        if state[2] > 6:
            nextStates = self.getListNextStatesB([state])
        else:
            nextStates = self.getListNextStatesW([state])
        nextPositions = []
        for i in nextStates:
            nextPositions.append(i[0][0:2])
        return nextPositions

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def getListNextStatesB(self, myState):
        self.chess.boardSim.getListNextStatesB(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates
    
    # Moves
    def getMovement(self, state, nextState):
        # Given a state and a successor state, return the postiion of the piece that has been moved in both states
        pieceState = None
        pieceNextState = None
        for piece in state:
            if piece not in nextState:
                movedPiece = piece[2]
                pieceNext = self.getPieceState(nextState, movedPiece)
                if pieceNext != None:
                    pieceState = piece
                    pieceNextState = pieceNext
                    break

        return [pieceState, pieceNextState]

    def movePieces(self, start, depthStart, to, depthTo):
        
        # To move from one state to the next we will need to find
        # the state in common, and then move until the node 'to'
        moveList = []
        # We want that the depths are equal to find a common ancestor
        nodeTo = to
        nodeStart = start
        # if the depth of the node To is larger than that of start, 
        # we pick the ancesters of the node until being at the same
        # depth
        while(depthTo > depthStart):
            moveList.insert(0,to)
            nodeTo = self.dictPath[str(nodeTo)][0]
            depthTo-=1
        # Analogous to the previous case, but we trace back the ancestors
        #until the node 'start'
        while(depthStart > depthTo):
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # We move the piece the the parerent state of nodeStart
            self.changeStateSim(nodeStart, ancestreStart)
            nodeStart = ancestreStart
            depthStart -= 1

        moveList.insert(0,nodeTo)
        # We seek for common node
        while nodeStart != nodeTo:
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # Move the piece the the parerent state of nodeStart
            self.changeStateSim(nodeStart,ancestreStart)
            # pick the parent of nodeTo
            nodeTo = self.dictPath[str(nodeTo)][0]
            # store in the list
            moveList.insert(0,nodeTo)
            nodeStart = ancestreStart
        # Move the pieces from the node in common
        # until the node 'to'
        for i in range(len(moveList)):
            if i < len(moveList) - 1:
                self.changeStateSim(moveList[i],moveList[i+1])
    
    # Cheecks and checkmates
    def isWatchedBk(self, currentState):

        self.newBoardSim(currentState)

        bkPosition = self.getPieceState(currentState, 12)[0:2]
        wkState = self.getPieceState(currentState, 6)
        wrState = self.getPieceState(currentState, 2)

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
        bkState = self.getPieceState(currentState, 12)
        allWatched = False

        # If the black king is on the edge of the board, all its moves might be under threat
        if bkState[0] == 0 or bkState[0] == 7 or bkState[1] == 0 or bkState[1] == 7:
            wrState = self.getPieceState(currentState, 2)
            whiteState = self.getWhiteState(currentState)
            allWatched = True
            # Get the future states of the black pieces
            nextBStates = self.getListNextStatesB(self.getBlackState(currentState))

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

        wkPosition = self.getPieceState(currentState, 6)[0:2]
        bkState = self.getPieceState(currentState, 12)
        brState = self.getPieceState(currentState, 8)

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

    # Path
    def reconstructPath(self, state, depth):
        # Once the solution is found, reconstruct the path taken to reach it
        for i in range(depth):
            self.pathToTarget.insert(0, state)
            # For each node, retrieve its parent from dictPath
            state = self.dictPath[str(state)][0]

        # Insert the root node at the beginning
        self.pathToTarget.insert(0, state)

    #--------------------------------------------------#
    
    #--------------- METODES Q-LEARNING ---------------#

    def initiate_q_table(self):
        pass

    def epsilon_greedy_policy(self, state, epsilon):
        pass

    def greedy_policy(self, state):
        pass

    def reward(self, currentState, color):
        # This method calculates the heuristic value for the current state.
        # The value is initially computed from White's perspective.
        # If the 'color' parameter indicates Black, the final value is multiplied by -1.

        value = 0

        bkState = self.getPieceState(currentState, 12)  # Black King
        wkState = self.getPieceState(currentState, 6)   # White King
        wrState = self.getPieceState(currentState, 2)   # White Rook
        brState = self.getPieceState(currentState, 8)   # Black Rook

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

        # If the current player is Black, invert the heuristic value.
        if not color:
            value *= -1

        # print("Current state for heuristic calculation: ", currentState, value)

        return value

    def doAnEpisode(self, policy, epsilon = None):
        """
        Realitza un episodi complet seguint la política donada
        Policy: 
            - True: epsilon_greeedy_policy  --> Training
            - False: greedy_policy          --> Evaluation
        """
        # Inicialitzem l'episodi
        if policy:      # Training
            
            # Reiniciar el simulador i agafar l'estat inicial
            state = None

        else:           # Evaluating

            # Reiniciar el tauler i agafar l'estat inicial
            state = None

        step = 0

        # Repeat step
        
        for step in range(self.max_steps):
            if policy:          # epsilong-greedy policy
                action = self.epsilon_greedy_policy(state, epsilon)
                # Movem el simulador
                # TODO

                newState = None

            else:               # greedy policy
                action = self.greedy_policy(state)
                # Movem el taulell
                # TODO

                newState = None

            reward = self.reward(newState)

            # Equacio de Bellman
            self.qTable[state][action] = self.qTable[state][action] + self.learning_rate * (reward + self.gamma * np.max(self.qTable[newState]) - self.qTable[state][action])

            # if not policy:
            #     print("Acció: ", action)
            #     print("Estat: ", state)
            #     print("newState: ", newState)
                
            #     input("CONTINUA...")

            # TODO
            if self.goalReached(newPosition):   # Si arribem a goal, acabem
                if (not policy):
                    print(f"Goal reached in {step+1} steps!")
                break
            else:                               # Si no, actualitzem l'estat
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
                print(f" In episode {episode}, the Q-table is:\n", self.qTable)

    def q_learning(self):
        # Iniciem q-table
        self.initiate_q_table()
        
        print("Training in process...")
        self.train()

        print("\nFinal q-table: ", self.qTable)

        print("\nEvaluating the agent...")

    #--------------------------------------------------#

if __name__ == "__main__":
    # Load initial positions of the pieces
    TA = np.zeros((8, 8))  

    TA[7][0] = 2    
    TA[7][5] = 6   
    TA[0][7] = 8   
    TA[0][5] = 12  

    # Inicialitzem el board i l'imprimim
    print("Starting AI chess... ")
    aichess = Aichess(TA, True)
    print("\nPrinting board")
    aichess.chess.boardSim.print_board()

    aichess.q_learning()