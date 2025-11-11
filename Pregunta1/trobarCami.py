"""

Practica 3 d'IA: Pregunta 1
@authors: Pol Lobo & Damia Carreras

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


class TrobarCami():
    """
    ATRIBUTES
    ---------
    sailorPosition : tuple
        Posició actual del mariner a la graella
    start : tuple
        Posició d'inici
    goal : tuple
        Posició objectiu
    map : np.array
        Graella que representa l'entorn
    exercici : str
        Exercici a o b

    METHODS
    -------
    initial_state() -> None
        Inicialitza l'estat segons l'exercici escollit (a o b)
    
    initiate_map() -> None
        Inicialitza la graella i el mariner

    print_map() -> None
        Dibuixa la graella i un cercle indicant on es el mariner

    move_sailor(new_pos) -> None
        Movem el mariner a la nova posició i actualitzem el dibuix

    q_learning() -> None
        Implementa l'algorisme de Q-learning per trobar el cami optim fins la meta

    """

    def __init__(self, start, goal, exercici):
        self.sailorPosition = start     # Posició inicial
        self.start = start              
        self.goal = goal                # Posició objectiu
        self.exercici = exercici        # Exercici a o b

        self.map = None
        self.initial_state()            # Graella inicial

        self.initiate_map()             # Dibuixem la graella

    def initial_state(self):
        """
        Inicialitza l'estat segons l'exercici escollit (a o b).
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
        Dibuixa la graella i el mariner a la posició inicial.
        """
        # Crear figura i eixos només una vegada
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.print_map()

        # Crear cercle (mariner) i afegir-lo
        self.cercle = Circle(
            (self.sailorPosition[1], self.sailorPosition[0]),
            0.3, fill=False, edgecolor='#8000FF', linewidth=4
        )
        self.ax.add_patch(self.cercle)

    def print_map(self):
        """
        Dibuixa la graella i un cercle indicant on es el mariner.
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

    def move_sailor(self, new_pos):
        """
        Movem el mariner a la nova posició i actualitzem el dibuix.
        """
        self.sailorPosition = new_pos
        self.cercle.center = (new_pos[1], new_pos[0])
        self.fig.canvas.draw_idle()  # Actualitzem

    def q_learning(self):
        pass


if __name__ == "__main__":
    # Creem la instància de la classe
    # Amb l'estat inicial de l'exercici a) o b)
    # Start i Goal són les posicions d'inici i final
    cami = TrobarCami((2,0), (0, 3), "a")
    # cami = TrobarCami((2,0), (0, 3), "b")

    plt.show()
