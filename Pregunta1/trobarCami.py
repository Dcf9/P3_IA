"""

Practica 3 d'IA: Pregunta 1
@authors: Pol Lobo & Damià Carreras

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from tqdm import trange
import copy


class TrobarCami():
    """

    Atributs
    -------------------------------------------
    start : tuple
        Posició d'inici
    goal : tuple
        Posició objectiu
    exercici : str
        Exercici a o b
    levelOfDrunkness : float
        Nivell d'embriaguesa (randomness)
    -------------------------------------------
        
    Mètodes
    -------------------------------------------
    initial_position() -> None
        Inicialitza l'estat segons l'exercici escollit (a o b)
    
    initiate_map() -> None
        Inicialitza la graella i el mariner

    initiate_q_table() -> None
        Inicialitza la taula Q i els diccionaris d'estat-posició i posició-estat    

    print_map() -> None
        Dibuixa la graella i un cercle indicant on es el mariner

    print_q_table() -> None
        Imprimeix per terminal la Q-table

    epsilon_greeedy_policy(state, epsilon) -> int
        Política d'actuació amb epsilon-greedy
    
    greedy_policy(state) -> int
        Política d'actuació greedy

    move_sailor(new_pos) -> None
        Movem el mariner a la nova posició i actualitzem el dibuix

    move_simulator(new_pos) -> None
        Movem el mariner en el simulador d'entrenament

    reward(state) -> float
        Retorna la recompensa associada a una posició donada
    
    goalReached(position) -> bool
        Comprova si s'ha arribat a la posició final

    doAnEpisode(policy, epsilon=None) -> None
        Realitza un episodi complet seguint la política donada
    
    train() -> None
        Entrena l'agent amb Q-learning

    q_learning() -> None
        Implementa l'algorisme Q-learning per trobar el camí òptim fins a la posició objectiu
    -------------------------------------------

    """

    def __init__(self, start, goal, exercici, levelOfDrunkness=0):

        #---------- Atributs pel mapa i el mariner ----------#
        self.sailorPosition = start                                         # Posició inicial
        self.levelOfDrunknss = levelOfDrunkness                             # Nivell d'embriaguesa (randomness)
        self.n_hiting_wall = 0                                              # Comptador de cops a la paret
        self.sailorPositionSimulator = start                                # Posició inicial per entrenar    
        self.start = start              
        self.goal = goal                                                    # Posició objectiu
        self.exercici = exercici                                            # Exercici a o b
        self.accions = {0: "Amunt", 1: "Avall", 2: "Esquerra", 3: "Dreta"}  # Accions possibles

        self.map = None
        self.initial_position()                                             # Graella inicial amb les recompenses en funció de l'exercici

        self.initiate_map()                                                 # Dibuixem la graella
        #-----------------------------------------------------#

        #-------------- Atributs per Q-learning --------------#
        self.qTable = None                                                  # Taula Q
        self.intermediateQTable = None                                      # Taula Q intermèdia
        self.posToState = {}                                                # Diccionari d'estats per la taula Q
        self.stateToPos = {}                                                # Diccionari de posicions per la taula Q

        self.n_training_episodes = 1000                                     # Nombre d'episodis d'entrenament
        self.learning_rate = 0.5                                            # Taxa d'aprenentatge (alpha)
        self.learning_rate_initial = 0.3                                    # Taxa d'aprenentatge inicial (alpha)

        self.n_eval_parametres = 100                                        # Nombre d'episodis per avaluar

        self.max_steps = 100                                                # Nombre màxim d'iteracions per episodi
        self.gamma = 0.95                                                   # Factor de descompte (gamma)             

        self.max_epsilon = 1                                                # Probabilitat d'exploració
        self.min_epsilon = 0.1                                              # Probabilitat mínima d'exploració
        self.decay_rate = 0.01                                              # Taxa de decaiguda d'epsilon
        #-----------------------------------------------------#

    def initial_position(self):
        """
        Inicialitza la posició segons l'exercici escollit (a o b).
        """
        if self.exercici == 'a':
            self.map = np.array([[-1, -1, -1, 100],
                                [-1, None, -1, -1],
                                [-1, -1, -1, -1]])
        elif self.exercici == 'b':
            self.map = np.array([[-3, -2, -1, 100],
                                [-4, None, -2, -1],
                                [-5, -4, -3, -2]])

    def initiate_map(self):
        """
        Inicialitza la graella i situa el mariner a la posició inicial.
        """
        plt.ion()  # Mode interactiu activat

        # Crear figura i eixos només una vegada
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.print_map()

        # Crear cercle (mariner) i afegir-lo
        self.cercle = Circle(
            (self.sailorPosition[1], self.sailorPosition[0]),
            0.3, fill=False, edgecolor='#8000FF', linewidth=4
        )
        self.ax.add_patch(self.cercle)

    def initiate_q_table(self):
        """
        El nostre mariner pot moure's en 4 direccions: (Columnes)
            - (0) Amunt
            - (1) Avall
            - (2) Esquerra
            - (3) Dreta

        El nostre mariner té 4x3 - 1 = 11 estats possibles: (Files)
            Cada posicio sera un estat diferent
        """

        n_files, n_cols = self.map.shape
        comptador_estats = 0

        # Inicialitzem el diccionari
        for i in range(n_files):
            for j in range(n_cols):
                if self.map[i, j] is not None:
                    self.posToState[(i, j)] = comptador_estats
                    self.stateToPos[comptador_estats] = (i, j)
                    comptador_estats += 1

        # Inicialitzem la taula Q amb zeros
        self.qTable = np.zeros((comptador_estats, 4))

        print("\nQ-table inicialitzada: \n", self.qTable)

    def print_map(self):
        """
        Dibuixa la graella i amb un cercle indicant la posició del mariner.
        """
        n_files, n_col = self.map.shape
        color_array = np.zeros((n_files, n_col, 3))
        white_rgb = np.array([1.0, 1.0, 1.0])
        gray_rgb = np.array([0.2, 0.2, 0.2])
        green_rgb = np.array([0.0, 1.0, 0.0])
        red_rgb = np.array([1.0, 0.0, 0.0])

        for i in range(n_files):
            for j in range(n_col):
                if self.map[i, j] is None:
                    color_array[i, j] = gray_rgb
                elif (i, j) == self.start:
                    color_array[i, j] = green_rgb
                elif (i, j) == self.goal:
                    color_array[i, j] = red_rgb
                else:
                    color_array[i, j] = white_rgb

        self.ax.imshow(color_array, interpolation='none')

        # Dibuixar línies de la graella
        for i in range(n_files):
            for j in range(n_col):
                rect = Rectangle((j-0.5, i-0.5), 1, 1,
                                 fill=False, edgecolor='black', linewidth=2)
                self.ax.add_patch(rect)

        # Afegir valors dins les caselles
        for i in range(n_files):
            for j in range(n_col):
                if self.map[i, j] is not None:
                    self.ax.text(j, i, str(int(self.map[i, j])),
                                 ha='center', va='center',
                                 color='black', fontsize=24)

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim(-0.5, n_col-0.5)
        self.ax.set_ylim(n_files-0.5, -0.5)

    def print_q_table(self):
        print("Q-table: \n", self.qTable)
    
    def epsilon_greeedy_policy(self, state, epsilon):
        """
        Política d'actuació
        Fixarem un epsilon, que definirà les probabilitats:
            - epsilon: Probabilitat d'exploració (triar una acció aleatòria)
            - (1 - epsilon): Probabilitat d'explotació (triar la millor acció segons la taula Q)
        Això s'escollira a partir d'un nombre aleatori de [0,1]
        """
        _, n_actions = self.qTable.shape

        random_number = np.random.uniform(0,1)
        if random_number < epsilon:                     # Exploració
            action = np.random.randint(n_actions)      
        else:                                           # Explotació
            action = np.argmax(self.qTable[state])      

        return action

    def greedy_policy(self, state):
        """
        Politica d'actualització
        Sempre tria la millor acció segons la taula Q
        """
        return np.argmax(self.qTable[state])

    def move_sailor(self, action):
        """
        Movem el mariner a la nova posició i actualitzem el dibuix.
        """
        # Randomness per l'embriaguesa
        if np.random.uniform(0,1) < self.levelOfDrunknss:
            action = np.random.randint(0,4)

        n_files, n_col = self.map.shape
        row, col = self.sailorPosition

        # Fem el moviment
        if action == 0:     # Amunt
            new_row, new_col = row - 1, col
        elif action == 1:   # Avall
            new_row, new_col = row + 1, col
        elif action == 2:   # Esquerra
            new_row, new_col = row, col - 1
        elif action == 3:   # Dreta
            new_row, new_col = row, col + 1
        else:
            raise ValueError("Acció invàlida")

        # Comprovem si el moviment es valid
        if not ((new_row < 0 or new_row >= n_files) or (new_col < 0 or new_col >= n_col) or (self.map[new_row, new_col] is None)):
            # Fem  el moviment
            self.sailorPosition = (new_row, new_col)    # format (fila, columna)
            self.cercle.center = (new_col, new_row)     # matplotlib treballa amb el format (columna, fila) 
            self.fig.canvas.draw_idle()                 # Actualitzem el dibuix
            plt.pause(0.3)                              # Pause per veure el moviment
            return True
        
        # Moviment no valid, ens quedem a la mateixa posicio: es xoca
        self.n_hiting_wall += 1
        return False

    def move_simulator(self, action):
        """
        Comprova si es possible fer el moviment en el simulador
        """
        # Randomness per l'embriaguesa
        if np.random.uniform(0,1) < self.levelOfDrunknss:
            action = np.random.randint(0,4)

        n_files, n_col = self.map.shape
        row, col = self.sailorPositionSimulator

        # Fem el moviment
        if action == 0:     # Amunt
            new_row, new_col = row - 1, col
        elif action == 1:   # Avall
            new_row, new_col = row + 1, col
        elif action == 2:   # Esquerra
            new_row, new_col = row, col - 1
        elif action == 3:   # Dreta
            new_row, new_col = row, col + 1
        else:
            raise ValueError("Acció no vàlida")

        # Comprovem si el moviment es vàlid
        if not ((new_row < 0 or new_row >= n_files) or (new_col < 0 or new_col >= n_col) or (self.map[new_row, new_col] is None)):
            # Simulem el moviment
            self.sailorPositionSimulator = (new_row, new_col)
            return True
        
        # Moviment no vàlid i ens quedem a la mateixa posició
        return False

    def reward(self, state):
        """
        Retorna la recompensa associada a una posició donada
        """
        return self.map[state]

    def goalReached(self, position):
        """
        Comprova si s'ha arribat a la meta
        """
        return position == self.goal

    def doAnEpisode(self, policy, epsilon = None):
        """
        Realitza un episodi complet seguint la política donada
        Policy: 
            - True: epsilon_greeedy_policy  --> Training
            - False: greedy_policy          --> Deployment
        """
        
        sequence = []

        # Inicialitzem l'episodi
        if policy:
            self.sailorPositionSimulator = self.start
            state = self.posToState[self.sailorPositionSimulator]

        else:
            self.sailorPosition = self.start
            state = self.posToState[self.sailorPosition]

        step = 0

        # Repeat step
        for step in range(self.max_steps):

            if policy:

                action = self.epsilon_greeedy_policy(state, epsilon)
                
                self.move_simulator(action)

                # En el move_simulator ja actualitzem la posició del mariner
                newPosition = self.sailorPositionSimulator

            else:

                action = self.greedy_policy(state)

                self.move_sailor(action)

                # En el move_sailor ja actualitzem la posició del mariner
                newPosition = self.sailorPosition

            newState = self.posToState[newPosition]

            reward = self.reward(newPosition)

            sequence.append(self.accions[action])

            # Equacio de Bellman
            self.qTable[state][action] = self.qTable[state][action] + self.learning_rate * (reward + self.gamma * np.max(self.qTable[newState]) - self.qTable[state][action])

            if self.goalReached(newPosition):   # Si arribem a goal, acabem
                if (not policy):
                    print(f"Objectiu assolit en {step+1} passes!")
                    print("Seqüència d'accions: ", sequence)
                break
            else:                               # Si no, actualitzem l'estat
                state = newState

    def train(self):
        for episode in trange(self.n_training_episodes):
            # Al principi ens interessa explorar més que explotar
            # Després ens interessa més explotar que explorar
            epsilon = self.max_epsilon * np.exp(-self.decay_rate * episode)

            # Decreixem la taxa d'aprenentatge al llarg de l'entrenament
            #TODO: Actualitzar taxa d'aprenentatge al llarg de l'entrenament
            self.learning_rate = self.learning_rate_initial / (1 + episode)

            self.doAnEpisode(True, epsilon)

            # Imprimim la taula a meitat d'entrenament
            if episode == self.n_training_episodes // 2:
                self.intermediateQTable = copy.deepcopy(self.qTable)

    def q_learning(self):
        
        self.initiate_q_table()

        print("\nEntrenament en procés...")
        self.train()

        print("\nQ-Table a meitat d'entrenament: \n", self.intermediateQTable)

        # Taula Q final
        print("\nQ-Table final: \n", self.qTable)

        print("\nAvaluant l'agent...\n")

        # Mostrem el mapa sense bloquejar el codi
        plt.show()
        plt.pause(1)    # Esperem perquè s'obri

        self.doAnEpisode(False)

        print(f"El mariner ha xocat {self.n_hiting_wall} cops amb la paret!")

        # Mapa obert fins que es tanqui
        # plt.show(block=True)
        plt.pause(2)

if __name__ == "__main__":

    # Creem la instància de la classe

    # Amb la graella de l'exercici a) o b)

    # Posició inicial
    start = (2, 0)    

    # Posició objectiu         
    goal = (0, 3)
    

    # EXERCICI 1.a)
    # cami = TrobarCami(start, goal, "a")

    # EXERCICI 1.b)
    # cami = TrobarCami(start, (goal, "b")

    # EXERCICI 1.c)
    cami = TrobarCami(start, goal, "b", 0.01)

    cami.q_learning()