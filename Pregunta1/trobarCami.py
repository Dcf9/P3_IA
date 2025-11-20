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
    level_of_drunkness : float
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

    def __init__(self, start, goal, exercici, learning_rate_initial, learning_decay, gamma, max_epsilon, decay_rate, delta, test=False, level_of_drunkness=0):

        #---------- Atributs pel mapa i el mariner ----------#
        self.sailor_position = start                                        # Posició inicial
        self.level_of_drunkness = level_of_drunkness                        # Nivell d'embriaguesa (randomness)
        self.n_hiting_wall = 0                                              # Comptador de cops a la paret
        self.sailor_position_simulator = start                              # Posició inicial per entrenar    
        self.start = start              
        self.goal = goal                                                    # Posició objectiu
        self.exercici = exercici                                            # Exercici a o b
        self.accions = {0: "Amunt", 1: "Avall", 2: "Esquerra", 3: "Dreta"}  # Accions possibles
        self.test = test                                                    # Mode execució

        self.map = None
        self.initial_position()                                             # Graella inicial amb les recompenses en funció de l'exercici

        if (not self.test):
            self.initiate_map()                                             # Dibuixem la graella
        #-----------------------------------------------------#

        #-------------- Atributs per Q-learning --------------#
        self.q_table = None                                                 # Taula Q
        self.intermediate_q_table1 = None                                   # Taula Q intermitja 1
        self.intermediate_q_table2 = None                                   # Taula Q intermitja 2

        self.pos_to_state = {}                                              # Diccionari d'estats per la taula Q
        self.state_to_pos = {}                                              # Diccionari de posicions per la taula Q

        self.n_training_episodes = 1000                                     # Nombre d'episodis d'entrenament
        self.learning_rate = 0                                              # Taxa d'aprenentatge (alpha)
        self.learning_rate_initial = learning_rate_initial                                    # Taxa d'aprenentatge inicial
        self.learning_rate_min = 0.1                                        # Taxa d'aprenentatge mínima
        self.learning_decay = learning_decay                                # Decaiguda de la taxa d'aprenentatge

        self.n_eval_parametres = 100                                        # Nombre d'episodis per avaluar

        self.max_steps = 50                                                 # Nombre màxim d'iteracions per episodi
        self.gamma = gamma                                                  # Factor de descompte (gamma)             

        self.epsilon = 1                                                    # Probabilitat d'exploració inicial
        self.max_epsilon = max_epsilon                                      # Probabilitat d'exploració
        self.min_epsilon = 0.1                                              # Probabilitat mínima d'exploració
        self.decay_rate = decay_rate                                        # Taxa de decaiguda d'epsilon

        self.delta = delta                                                  # Criteri de convergència
        self.episodes = 0                                                   # Comptador d'episodis            
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

        if (not self.test):
            self.print_map()

        # Crear cercle (mariner) i afegir-lo
        self.cercle = Circle(
            (self.sailor_position[1], self.sailor_position[0]),
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
                    self.pos_to_state[(i, j)] = comptador_estats
                    self.state_to_pos[comptador_estats] = (i, j)
                    comptador_estats += 1

        # Inicialitzem la taula Q amb zeros
        
        self.q_table = np.zeros((comptador_estats, 4))

        if (not self.test):
            print("\nQ-table inicialitzada: \n", self.q_table)

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
        """
        Mostra la Q-Table per terminal.
        """
        print("Q-table: \n", self.q_table)
    
    def epsilon_greeedy_policy(self, state, epsilon):
        """
        Política d'actuació
        Fixarem un epsilon, que definirà les probabilitats:
            - epsilon: Probabilitat d'exploració (triar una acció aleatòria)
            - (1 - epsilon): Probabilitat d'explotació (triar la millor acció segons la taula Q)
        Això s'escollira a partir d'un nombre aleatori de [0,1]
        """
        _, n_actions = self.q_table.shape

        random_number = np.random.uniform(0,1)
        if random_number < epsilon:                     # Exploració
            action = np.random.randint(n_actions)      
        else:                                           # Explotació
            action = np.argmax(self.q_table[state])      

        return action

    def greedy_policy(self, state):
        """
        Politica d'actualització
        Sempre tria la millor acció segons la taula Q
        """
        return np.argmax(self.q_table[state])

    def move_sailor(self, action):
        """
        Movem el mariner a la nova posició i actualitzem el dibuix.
        """
        # Randomness per l'embriaguesa
        if np.random.uniform(0,1) < self.level_of_drunkness:
            action = np.random.randint(0,4)

        n_files, n_col = self.map.shape
        row, col = self.sailor_position

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
            self.sailor_position = (new_row, new_col)    # format (fila, columna)
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
        if np.random.uniform(0,1) < self.level_of_drunkness:
            action = np.random.randint(0,4)

        n_files, n_col = self.map.shape
        row, col = self.sailor_position_simulator

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
            self.sailor_position_simulator = (new_row, new_col)
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

    def doAnEpisode(self, policy):
        """
        Realitza un episodi complet seguint la política donada
        Policy: 
            - True: epsilon_greeedy_policy  --> Training
            - False: greedy_policy          --> Deployment
        """
        
        sequence = []

        # Inicialitzem l'episodi
        if policy:
            self.sailor_position_simulator = self.start
            state = self.pos_to_state[self.sailor_position_simulator]

        else:
            self.sailor_position = self.start
            state = self.pos_to_state[self.sailor_position]

        step = 0

        # Repeat step
        for step in range(self.max_steps):

            if policy:

                action = self.epsilon_greeedy_policy(state, self.epsilon)
                
                self.move_simulator(action)

                # En el move_simulator ja actualitzem la posició del mariner
                new_position = self.sailor_position_simulator

            else:

                action = self.greedy_policy(state)

                self.move_sailor(action)

                # En el move_sailor ja actualitzem la posició del mariner
                new_position = self.sailor_position

            new_state = self.pos_to_state[new_position]

            reward = self.reward(new_position)

            sequence.append(self.accions[action])

            # Equacio de Bellman
            self.q_table[state][action] = self.q_table[state][action] + self.learning_rate * (reward + self.gamma * np.max(self.q_table[new_state]) - self.q_table[state][action])

            if self.goalReached(new_position):   # Si arribem a goal, acabem
                if (not policy):
                    print(f"Objectiu assolit en {step+1} passes!")
                    print("Seqüència d'accions: ", sequence)
                break
            else:                               # Si no, actualitzem l'estat
                state = new_state

    def train(self):
        """
        Entrena l'agent fins a obtenir la convergència a la taula Q.
        """

        # Contador de convergència que ens permet saber quan s'ha arribat a la convergència 
        convergence_cnt = 0

        

        # Si la taula Q no canvia en 5 episodis seguits, considerem que ha convergit
        # while (convergence_cnt < 5 and self.episodes < self.n_training_episodes):
        while (convergence_cnt < 5 and self.episodes < self.n_training_episodes):

            # Al principi ens interessa explorar més que explotar
            # Després ens interessa més explotar que explorar
            self.epsilon = max(self.min_epsilon, self.max_epsilon * np.exp(-self.decay_rate * self.episodes))

            # Decrementem la taxa d'aprenentatge al llarg de l'entrenament
            self.learning_rate = max(self.learning_rate_min, self.learning_rate_initial *np.exp(-self.learning_decay * self.episodes))

            # Actualitzem la taula Q antiga
            old_q_table = self.q_table.copy()

            # Realitzem un episodi d'entrenament
            self.doAnEpisode(True)

            # Comprovem la convergència
            if (self.converge(old_q_table, self.q_table)):
                convergence_cnt += 1
            else:
                convergence_cnt = 0

            # Actualitzem el comptador d'episodis    
            self.episodes += 1


            # Imprimim la taula a meitat d'entrenament
            if self.episodes == 50:
                self.intermediate_q_table1 = self.q_table.copy()
            elif self.episodes == 100:
                self.intermediate_q_table2 = self.q_table.copy()


        print("\nEntrenament finalitzat.")

        print("Nombre d'episodis d'entrenament:", self.episodes)

    def converge(self, old_q_table, new_q_table):

        # Comprova si la taula Q ha canviat menys que el delta definit
        return np.max(np.abs(new_q_table - old_q_table)) < self.delta

    def q_learning(self):
        
        # Inicialitzem la taula Q
        self.initiate_q_table()

        # Entrenem l'agent
        print("\nEntrenament en procés...")
        self.train()

        print("\nQ-Table a 1/3 d'entrenament: \n", self.intermediate_q_table1)

        print("\nQ-Table a 2/3 d'entrenament: \n", self.intermediate_q_table2)

        # Taula Q final
        print("\nQ-Table final: \n", self.q_table)

        print("\nAvaluant l'agent...\n")

        # Mostrem el mapa sense bloquejar el codi
        plt.show()
        plt.pause(1)    # Esperem perquè s'obri

        # Avaluem l'agent amb la política greedy
        self.doAnEpisode(False)

        print(f"El mariner ha xocat {self.n_hiting_wall} cops amb la paret!")

        # Mapa obert fins que es tanqui
        plt.show(block=True)
        plt.pause(2)


    # Funcions d'avaluació de rendiment i benchmark
    def benchmark(self, repetitions = 100):
        """
        Funció per benchmark i comparació dels resultats de diferents paràmetres.
        """

        total_episodes = 0
        oscilacions = 0

        for i in range(repetitions):

            test = True

            # Creem la instància de la classe
            bench = TrobarCami(self.start, self.goal, self.exercici, self.learning_rate_initial, self.learning_decay, self.gamma, self.max_epsilon, self.decay_rate, self.delta, test, self.level_of_drunkness)
            
            # Iniciem la taula Q
            bench.initiate_q_table()

            # Entrenem l'agent
            oscilacions += bench.testing()

            # Acumulem el nombre d'episodis
            total_episodes += bench.episodes

        # Calculem la mitjana d'episodis
        avg_episodes = total_episodes / repetitions

        # Calculem la mitjana d'oscil·lacions
        oscilacions_avg = oscilacions / repetitions

        print("Average episodes to reach goal over", repetitions, "runs:", avg_episodes)
        print("Average oscillations over", repetitions, "runs:", oscilacions_avg)
        print("Q-Table after benchmark: \n", bench.q_table, "\n")

    def testing(self): 
        """
        Funció d'entrenament de l'agent que ens retorna les oscil·lacions mitjanes i contabilitza els episodis necessaris per a la convergència.
        """

        convergence_cnt = 0

        oscilacions = 0


        # Si la taula Q no canvia en 5 episodis seguits, considerem que ha convergit
        while (convergence_cnt < 5 and self.episodes < self.n_training_episodes):

            # Al principi ens interessa explorar més que explotar
            # Després ens interessa més explotar que explorar
            self.epsilon = max(self.min_epsilon, self.max_epsilon * np.exp(-self.decay_rate * self.episodes))

            # Decrementem la taxa d'aprenentatge al llarg de l'entrenament
            self.learning_rate = max(self.learning_rate_min, self.learning_rate_initial *np.exp(-self.learning_decay * self.episodes))

            old_q_table = self.q_table.copy()

            # Realitzem un episodi d'entrenament
            self.doAnEpisode(True)

            # Calculem les oscil·lacions
            oscilacions += np.max(np.abs(self.q_table - old_q_table))

            # Comprovem la convergència
            if (self.converge(old_q_table, self.q_table)):
                convergence_cnt += 1
            else:
                convergence_cnt = 0
            self.episodes += 1

        # Calculem la mitjana d'oscil·lacions
        oscilacions_avg = oscilacions / self.episodes

        return oscilacions_avg



if __name__ == "__main__":

    # Creem la instància de la classe
    # Amb la graella de l'exercici a) o b)

    # Posició inicial
    start = (2, 0)    

    # Posició objectiu         
    goal = (0, 3)

    

    # EXERCICI 1.a)
    cami = TrobarCami(start, goal, "a", 0.45, 0.0001, 0.7, 0.25, 0.001, 1e-3)
    cami.q_learning()


    # EXERCICI 1.b)
    # cami = TrobarCami(start, goal, "b", 0.7, 0.0001, 0.4, 0.2, 0.002, 1e-3)
    # cami.q_learning()


    # EXERCICI 1.c)
    # cami = TrobarCami(start, goal, "b", 0.6, 0.0001, 0.4, 0.2, 0.002, 1e-3, 0.01)
    # cami.q_learning()



    # FUNCIÓ BENCHMARK PER COMPARAR DIFERENTS PARÀMETRES

    # repetitions = 5000

    # Exericici a
    # test = TrobarCami(start, goal, "a", 0.45, 0.0001, 0.7, 0.25, 0.001, 1e-3, False)
    # test.benchmark(repetitions)

    # Exercici b
    # test = TrobarCami(start, goal, "b", 0.7, 0.0001, 0.4, 0.2, 0.002, 1e-3, True)
    # test.benchmark(repetitions)

    # Exercici c
    # test = TrobarCami(start, goal, "b", 0.6, 0.0001, 0.4, 0.2, 0.002, 1e-3, 0.01, True)
    # test.benchmark(repetitions)





    
