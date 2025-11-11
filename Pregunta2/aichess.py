#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import copy
import math

import chess
import board
import numpy as np
import sys
import queue
from typing import List

RawStateType = List[List[List[int]]]

from itertools import permutations


class Aichess():
    """
    A class to represent the game of chess.

    ...

    Attributes:
    -----------
    chess : Chess
        represents the chess game
        
    listNextStates : list
        List of next possible states for the current player.

    listVisitedStates : list
        List of all visited states during A* and other search algorithms.

    listVisitedSituations : list
        List of visited game situations (state + color) for minimax/alpha-beta pruning.

    pathToTarget : list
        Sequence of states from the initial state to the target (used by A*).

    depthMax : int
        Maximum search depth for minimax/alpha-beta searches.

    dictPath : dict
        Dictionary used to reconstruct the path in A* search.

    Methods:
    --------
    copyState(state) -> list
        Returns a deep copy of the given state.

    isVisitedSituation(color, mystate) -> bool
        Checks whether a given state with a specific color has already been visited.

    getListNextStatesW(myState) -> list
        Returns a list of possible next states for the white pieces.

    getListNextStatesB(myState) -> list
        Returns a list of possible next states for the black pieces.

    isSameState(a, b) -> bool
        Checks whether two states represent the same board configuration.

    isVisited(mystate) -> bool
        Checks if a given state has been visited in search algorithms.

    getCurrentState() -> list
        Returns the combined state of both white and black pieces.

    getNextPositions(state) -> list
        Returns a list of possible next positions for a given state.

    heuristica(currentState, color) -> int
        Calculates a heuristic value for the current state from the perspective of the given color.

    movePieces(start, depthStart, to, depthTo) -> None
        Moves all pieces along the path between two states.

    changeState(start, to) -> None
        Moves a single piece from start state to to state.

    reconstructPath(state, depth) -> None
        Reconstructs the path from initial state to the target state for A*.

    isWatchedWk(currentState) / isWatchedBk(currentState) -> bool
        Checks if the white or black king is under threat.

    allWkMovementsWatched(currentState) / allBkMovementsWatched(currentState) -> bool
        Checks if all moves of the white or black king are under threat.

    isWhiteInCheckMate(currentState) / isBlackInCheckMate(currentState) -> bool
        Determines if the white or black king is in checkmate.

    minimaxGame(depthWhite: int, depthBlack: int) -> To be implemented by you
        Simulates a full game using the Minimax algorithm for both white and black.

    alphaBetaPoda(depthWhite: int, depthBlack: int) -> To be implemented by you
        Simulates a game where both players use Minimax with Alpha-Beta Pruning.

    expectimax(depthWhite: int, depthBlack: int) -> To be implemented by you
        Simulates a full game where both players use the Expectimax algorithm.

    mean(values: list[float]) -> float
        Returns the arithmetic mean (average) of a list of numerical values.

    standardDeviation(values: list[float], mean_value: float) -> float
        Computes the standard deviation of a list of numerical values based on the given mean.

    calculateValue(values: list[float]) -> float
        Computes the expected value from a set of scores using soft-probabilities 
        derived from normalized values (exponential weighting). Can be useful for Expectimax.

    """

    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
            self.chess2 = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)
            self.chess2 = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.listVisitedSituations = []
        self.pathToTarget = []
        self.depthMax = 8;
        # Dictionary to reconstruct the visited path
        self.dictPath = {}
        # Prepare a dictionary to control the visited state and at which
        # depth they were found for DepthFirstSearchOptimized
        self.dictVisitedStates = {}

    def copyState(self, state):
        
        copyState = []
        for piece in state:
            copyState.append(piece.copy())
        return copyState
        
    def isVisitedSituation(self, color, mystate):
        
        if (len(self.listVisitedSituations) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedSituations)):
                    if self.isSameState(list(perm_state[j]), self.listVisitedSituations.__getitem__(k)[1]) and color == self.listVisitedSituations.__getitem__(k)[0]:
                        isVisited = True

            return isVisited
        else:
            return False

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def getListNextStatesB(self, myState):
        self.chess.boardSim.getListNextStatesB(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

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

    def newBoardSim(self, listStates):
        # We create a  new boardSim
        TA = np.zeros((8, 8))
        for state in listStates:
            TA[state[0]][state[1]] = state[2]

        self.chess.newBoardSim(TA)

    def newBoardSim2(self, listStates):
        # We create a  new boardSim
        TA = np.zeros((8, 8))
        for state in listStates:
            TA[state[0]][state[1]] = state[2]

        self.chess2.newBoardSim(TA)

    def getPieceState(self, state, piece):
        pieceState = None
        for i in state:
            if i[2] == piece:
                pieceState = i
                break
        return pieceState

    def getCurrentState(self):
        listStates = []
        for i in self.chess.board.currentStateW:
            listStates.append(i)
        for j in self.chess.board.currentStateB:
            listStates.append(j)
        return listStates

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
            self.changeState(nodeStart, ancestreStart)
            nodeStart = ancestreStart
            depthStart -= 1

        moveList.insert(0,nodeTo)
        # We seek for common node
        while nodeStart != nodeTo:
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # Move the piece the the parerent state of nodeStart
            self.changeState(nodeStart,ancestreStart)
            # pick the parent of nodeTo
            nodeTo = self.dictPath[str(nodeTo)][0]
            # store in the list
            moveList.insert(0,nodeTo)
            nodeStart = ancestreStart
        # Move the pieces from the node in common
        # until the node 'to'
        for i in range(len(moveList)):
            if i < len(moveList) - 1:
                self.changeState(moveList[i],moveList[i+1])

    def reconstructPath(self, state, depth):
        # Once the solution is found, reconstruct the path taken to reach it
        for i in range(depth):
            self.pathToTarget.insert(0, state)
            # For each node, retrieve its parent from dictPath
            state = self.dictPath[str(state)][0]

        # Insert the root node at the beginning
        self.pathToTarget.insert(0, state)

    def changeState(self, start, to):
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
    
    def heuristica(self, currentState, color):
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
    
    def mean(self, values):
        # Calculate the arithmetic mean (average) of a list of numeric values.
        total = 0
        n = len(values)
        
        for i in range(n):
            total += values[i]

        return total / n

    def standard_deviation(self, values, mean_value):
        # Calculate the standard deviation of a list of values.
            total = 0
            n = len(values)

            for i in range(n):
                total += pow(values[i] - mean_value, 2)

            return pow(total / n, 1 / 2)

    def calculateValue(self, values):
        # Calculate a weighted expected value based on normalized probabilities. - useful for Expectimax.
        
        # Compute mean and standard deviation
        mean_value = self.mean(values)
        std_dev = self.standard_deviation(values, mean_value)

        # If all values are equal, the deviation is 0, equal probability
        if std_dev == 0:
            return values[0]

        expected_value = 0
        total_weight = 0
        n = len(values)

        for i in range(n):
            # Normalize value using z-score
            normalized_value = (values[i] - mean_value) / std_dev

            # Convert to a positive weight using e^(-x)
            positive_weight = pow(1 / math.e, normalized_value)

            # Weighted sum
            expected_value += positive_weight * values[i]
            total_weight += positive_weight

        # Final expected value (weighted average)
        return expected_value / total_weight

#------------ AUXILIAR FUNCTIONS --------

# Checks whether the current state is valid (no moving king under threat and no stalemate)
    def isValid(self, nextState, color):

        # Color indicates the player that just moved: True for white, False for black
        # nextState: state we are checking  

        wkWatched = self.isWatchedWk(nextState)
        bkWatched = self.isWatchedBk(nextState)

        movingKingThreatened = False
        stalemate = True

        if (color and wkWatched) or (not color and bkWatched):                                      # If the moving king is under threat, invalid state
            movingKingThreatened = True


        if (color and bkWatched) or (not color and wkWatched):                                      # If opponent king is under threat, no stalemate
            stalemate = False
        
        elif (color and not bkWatched) or (not color and not wkWatched):                           # If the opponent king is not under threat, we check for stalemate

            possibleMovements = []

            if (color):                                                                             # If white pieces moved, we check black pieces movements
                possibleMovements = self.getListNextStatesB(self.getBlackState(nextState))
                partialState = self.getWhiteState(nextState)
            else:                                                                                   # If black pieces moved, we check white pieces movements
                possibleMovements = self.getListNextStatesW(self.getWhiteState(nextState))
                partialState = self.getBlackState(nextState)

            for state in possibleMovements:

                moveColor = not color

                state = self.completeState(state, partialState)

                # We now check if the moving king is under threat in this new state
                if (moveColor == True and not self.isWatchedWk(state)):
                    stalemate = False
                    break
                elif (moveColor == False and not self.isWatchedBk(state)):
                    stalemate = False
                    break
    
        return (not movingKingThreatened and not stalemate)

    # This function allows us to complete a partial state (only one color) with the pieces of the other color
    def completeState(self, partialState, currentState):

        # We need tot deepcopy to avoid modifying the original currentState
        auxState = currentState.copy()
    
        for piece in partialState:
            for otherPiece in auxState:
                if (piece[0] == otherPiece[0] and piece[1] == otherPiece[1] and piece[2] != otherPiece[2]):         # Check for possible piece captures
                    auxState.remove(otherPiece)

        return partialState + auxState

    # This functions checks whether a state has been visited before
    def isVisited(self, state, exploredStates):
        return (state in exploredStates)

    # Transforms a state into a normalized tuple representation for consistent comparison, reducing computational cost
    def normalize_state(self, state):
        return tuple(sorted((tuple(piece) for piece in state), key=lambda x: x[2]))

#------------ END AUXILIAR FUNCTIONS --------



#------------ MINIMAX -----------------
                    
    def minimax(self, currentState, depth, color, exploredStates, depthMax):

        # Terminal node
        if (self.isWhiteInCheckMate(currentState) or self.isBlackInCheckMate(currentState) or depth == 0):                                                   

                if (depthMax % 2 == 0):
                    return (self.heuristica(currentState, True), None)                                                  # Heuristic from white perspective
                
                else:
                    return (-self.heuristica(currentState, False), None)                                                # Heuristic from black perspective

        nextState = None

        # White pieces turn
        if color:

            value = -math.inf
            possibleMoves = self.getListNextStatesW(self.getWhiteState(currentState))

            if (len(possibleMoves) > 0):

                for state in possibleMoves:

                    state = self.completeState(state, self.getBlackState(currentState))                                 # Complete the state with black pieces
                    stateTuple = self.normalize_state(state.copy())                                                     # Get state tuple for comparison

                    if (self.isValid(state, True) and not self.isVisited(stateTuple, exploredStates)):                   # Check if the movement is valid and not visited before
                        
                        exploredStates.add(stateTuple)
                        result = self.minimax(state, depth - 1, False, exploredStates.copy(), depthMax)[0]

                        if (value < result):                                                                            # Get best move for white pieces
                            value = result
                            nextState = state.copy()

        # Black pieces turn
        else:
            value = math.inf
            possibleMoves = self.getListNextStatesB(self.getBlackState(currentState))

            if (len(possibleMoves) > 0):

                for state in possibleMoves:

                    state = self.completeState(state, self.getWhiteState(currentState))                                   # Complete the state with white pieces
                    stateTuple = self.normalize_state(state.copy())

                    if (self.isValid(state, False) and not self.isVisited(stateTuple, exploredStates)):                   # Check if the movement is valid and not visited before

                        exploredStates.add(stateTuple)
                        result = self.minimax(state, depth - 1, True, exploredStates.copy(), depthMax)[0]

                        if (value > result):                                                                               # Get best move for black pieces
                            value = result
                            nextState = state.copy()

        return (value, nextState)

#------------ END MINIMAX -----------------

#------------ AlphaBetaPrunning -----------------

    def alphaBetaPruning(self, currentState, depth, color, exploredStates, depthMax, alpha = -math.inf, beta = math.inf):

        if (self.isWhiteInCheckMate(currentState) or depth == 0):                                                   # Check if terminal state

                if (depthMax % 2 == 0):
                    return (self.heuristica(currentState, True), None)                                                  # Heuristic from white perspective
                
                else:
                    return (-self.heuristica(currentState, False), None)                                                # Heuristic from black perspective

        nextState = None

        # White pieces turn
        if color:

            value = -math.inf

            possibleMoves = self.getListNextStatesW(self.getWhiteState(currentState))

            if (len(possibleMoves) > 0):

                for state in possibleMoves:

                    state = self.completeState(state, self.getBlackState(currentState))                                 # Complete the state with black pieces

                    stateTuple = self.normalize_state(state.copy())                                                     # Get state tuple for comparison

                    if (self.isValid(state, True) and not self.isVisited(stateTuple, exploredStates)):                  # Check if the movement is valid and not visited before
                        
                        exploredStates.add(stateTuple)

                        result = self.alphaBetaPruning(state, depth - 1, False, exploredStates.copy(), depthMax, alpha, beta)[0]

                        if (value < result):                                                                            # Get best move for white pieces
                            value = result
                            nextState = state.copy()
                        
                        if (value >= beta):                                                                              # Beta cut-off
                            break
                        alpha = max(alpha, value)                                                                        # Update alpha

                if (nextState is None):                                                                                  # If no valid moves or new moves found
                    return (math.inf, None)                                                                              # Kill the branch by returnin math.inf

            else:
                return (math.inf, None)                                                                                  # No possible moves


        # Black pieces turn
        else:
            
            value = math.inf

            possibleMoves = self.getListNextStatesB(self.getBlackState(currentState))

            if (len(possibleMoves) > 0):

                for state in possibleMoves:

                    state = self.completeState(state, self.getWhiteState(currentState))                                   # Complete the state with white pieces
                    stateTuple = self.normalize_state(state.copy())

                    if (self.isValid(state, False) and not self.isVisited(stateTuple, exploredStates)):                   # Check if the movement is valid and not visited before

                        exploredStates.add(stateTuple)

                        result = self.alphaBetaPruning(state, depth - 1, True, exploredStates.copy(), depthMax, alpha, beta)[0]

                        if (value > result):                                                                               # Get best move for black pieces
                            value = result
                            nextState = state.copy()

                        if (value <= alpha):                                                                              # Alpha cut-off
                            break
                        beta = min(beta, value)                                                                           # Update beta

                if (nextState is None):                                                                                     # Kill the branch if no valid moves or new moves found
                    return (-math.inf, None)

            else:
                return (-math.inf, None)

        return (value, nextState)

#------------ END AlphaBetaPrunning -----------------

#------------ EXPECTIMAX -----------------

    def expectimax(self, currentState, depth, color, exploredStates, depthMax):   
        # Terminal node: checkmate or depth limit reached
        if self.isWhiteInCheckMate(currentState) or self.isBlackInCheckMate(currentState) or depth == 0:
            # Evaluate from the perspective of the player who started the search
            if color:
                return (self.heuristica(currentState, True), None)  # White's perspective
            else:
                return (-self.heuristica(currentState, False), None)  # Black's perspective

        nextState = None

        # MAX node: white's turn
        if color:
            value = -math.inf
            possibleMoves = self.getListNextStatesW(self.getWhiteState(currentState))

            for state in possibleMoves:
                state = self.completeState(state, self.getBlackState(currentState))
                stateTuple = self.normalize_state(state)

                # Important: if depth == depthMax, we allow revisiting states
                # This is because a state might be explored in one branch,
                # but skipping it in another could leave us with no legal moves
                if self.isValid(state, True) and (not self.isVisited(stateTuple, exploredStates) or depth == depthMax):
                    newExplored = exploredStates.copy()
                    newExplored.add(stateTuple)

                    result = self.expectimax(state, depth - 1, False, newExplored, depthMax)[0]

                    if result > value:
                        value = result
                        nextState = state.copy()

            # If no valid move was found, evaluate the current state instead
            if nextState is None:
                value = self.heuristica(currentState, True)

        # CHANCE node: black's turn
        else:
            valuesList = []
            possibleMoves = self.getListNextStatesB(self.getBlackState(currentState))

            for state in possibleMoves:
                state = self.completeState(state, self.getWhiteState(currentState))
                stateTuple = self.normalize_state(state)

                if self.isValid(state, False) and (not self.isVisited(stateTuple, exploredStates) or depth == depthMax):
                    newExplored = exploredStates.copy()
                    newExplored.add(stateTuple)

                    result = self.expectimax(state, depth - 1, True, newExplored, depthMax)[0]
                    valuesList.append(result)

            # If no valid moves were found, evaluate the current state
            if len(valuesList) == 0:
                value = self.heuristica(currentState, True)
            else:
                value = self.calculateValue(valuesList)

        return (value, nextState)

#------------ END EXPECTIMAX -----------------

#------------ CUSTOM GAME -----------------
    # This function allows us to customize the game by choosing the depth and algorithm (minimax or alpha-beta prune) for each player
    # algorithmWhite, algorithmBlack are strings indicating whether to use minimax, alpha-beta pruning or expectimax for each player
    def customGame(self, depthWhite, depthBlack, algorithmWhite, algorithmBlack):

        currentState = self.getCurrentState()

        validState = self.isValid(currentState, False)                                                                          # Check if initial state is valid

        checkmate = self.isWhiteInCheckMate(currentState) or self.isBlackInCheckMate(currentState)                              # Check if initial state is checkmate

        draw = len(currentState) == 2 and (currentState[0][2] == 6 and currentState[1][2] == 12)                                # Check if initial state is draw (only kings left in this case)  

        moveCnt = 0

        if not validState:
            print("The initial state is invalid")
            exit(1)

        elif checkmate:
            print("Checkmate already at the beginning of the game!!")
            exit(0)

        elif draw:
            print("Draw already at the beginning of the game!!")
            exit(0)

        else:

            visitedStates = []                                                                                                              # List of visited states
            visitedStates.append(self.normalize_state(currentState.copy()))                                                                 # Save states as normalized tuples for comparison
            
            checkmateW = False
            checkmateB = False

            depthBlack = 2 if (depthWhite % 2 == 0 and depthBlack == 1 and algorithmBlack == 1 and algorithmWhite == 2) else depthBlack

            while (not checkmateW and not checkmateB and not draw):                                                                         # If no checkmate or draw, continue the game

                # White pieces move first
                if (algorithmWhite == 0):                                                                                                        
                    value, nextState = self.minimax(currentState, depthWhite, True, set(visitedStates.copy()), depthWhite)                  # Minimax form white pieces
                elif (algorithmWhite == 1):
                    value, nextState = self.alphaBetaPruning(currentState, depthWhite, True, set(visitedStates.copy()), depthWhite)         # Alpha-beta pruning for white pieces                                                                                   
                elif (algorithmWhite == 2):
                    value, nextState = self.expectimax(currentState, depthWhite, True, set(visitedStates.copy()), depthWhite)
                if (value == None and nextState == None):                                                                                   # Check if white pieces have available moves
                    print("White king has no available moves. It's a Stalemate!!")
                    break

                visitedStates.append(self.normalize_state(nextState.copy()))                                                                # Save new state
                movement = self.getMovement(currentState, nextState)                                                                        # Get the movement made
                self.chess.move(movement[0], movement[1])                                                                                   # Update real board                                          
                currentState = self.getCurrentState()                                                                                       # Update current state

                moveCnt += 1

                print("Board after white pieces move:")                                                                                     # Print out the board to see the new state
                self.chess.board.print_board()                                                                                              
                
                checkmateW = self.isBlackInCheckMate(currentState)                                                                          # Check if black pieces are in checkmate

                draw = (len(currentState) == 2)                                                                                             # Check if there's a draw

                # Black pieces turn
                if (not checkmateW and not draw):

                    if (algorithmBlack == 0):
                        value, nextState = self.minimax(currentState, depthBlack, False, set(visitedStates.copy()), depthBlack)             # Minimax for black pieces
                    elif (algorithmBlack == 1):
                        value, nextState = self.alphaBetaPruning(currentState, depthBlack, False, set(visitedStates.copy()), depthBlack)    # Alpha-Beta Pruning for black pieces
                    elif (algorithmBlack == 2):
                        value, nextState = self.expectimax(currentState, depthBlack, False, set(visitedStates.copy()), depthBlack)

                    if (value == None and nextState == None):                                                                               # Check if black pieces have available moves
                        print("Black king has no available moves. It's a stalemate!!")

                    visitedStates.append(self.normalize_state(nextState.copy()))                                                            # Save new state
                    movement = self.getMovement(currentState, nextState)                                                                    # Get the movement made
                    self.chess.move(movement[0], movement[1])                                                                               # Update real board                                          
                    currentState = self.getCurrentState()                                                                                   # Update current state

                    print("Board after black pieces move:")                                                                                 # Print out the board to see the new state
                    self.chess.board.print_board()

                    # Check if white pieces are in checkmate
                    checkmateB = self.isWhiteInCheckMate(currentState)                                                                      # Check if white pieces are in checkmate

                    draw = len(currentState) == 2                                                                                           # Check if there's a draw

        if (checkmateW):
            print("White pieces won in ", moveCnt, " moves!!")     

        elif (checkmateB):
            print("Black pieces won in ", moveCnt, " moves!!")   

        elif (draw):
            print("The game ended in a draw!!")

#----------- END CUSTOM GAME -----------------

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     sys.exit(usage())

    # Initialize an empty 8x8 chess board
    TA = np.zeros((8, 8))

    # Load initial positions of the pieces
    TA = np.zeros((8, 8))  

    TA[7][0] = 2    
    TA[7][5] = 6   
    TA[0][7] = 8   
    TA[0][5] = 12  

    # Initialise board and print
    print("starting AI chess... ")
    aichess = Aichess(TA, True)
    print("printing board")
    aichess.chess.boardSim.print_board()
    
    ## Comment and descomment for the exercises
    ## Dictionary for chossing the algorithm easily 
    algorithms = {"minimax" : 0, "alphabeta" : 1, "expectimax" : 2}

    # Run exercise 1 and 2
    # print("-----------Running EXERCISE 1 and 2:-----------")
    # aichess.customGame(4,4,algorithms["minimax"],algorithms["minimax"])

    # Run exercise 3
    # print("-----------Running EXERCISE 3:-----------")
    # aichess.customGame(4,4,algorithms["minimax"],algorithms["alphabeta"])

    # Run exercise 4
    # print("-----------Running EXERCISE 4:-----------")
    # aichess.customGame(3,3,algorithms["alphabeta"],algorithms["alphabeta"])

    # Run exercise 5
    print("-----------Running EXERCISE 5:-----------")
    aichess.customGame(1,5,algorithms["expectimax"], algorithms["alphabeta"])