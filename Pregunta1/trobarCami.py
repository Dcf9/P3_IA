"""

Practica 3 d'IA: Pregunta 1
@authors: Pol Lobo & Damia Carreras

"""

import numpy as np
import matplotlib.pyplot as plt

class trobarCami():
    """
    A class to represent de pathfinding problem.

    METHODS
    -------

    print_graella(state) -> None
        Rep l'estat actual i imprimeix la graella on trobarem el cami

    """

    def __init__(self, graph):
        pass

    def inicial_state(self):
        pass

    def print_graella(self, state):
        # Crear la figura
        plt.figure(figsize=(3, 4))
        plt.imshow(state, cmap='viridis', interpolation='none')

        # Añadir los valores dentro de cada casilla
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                plt.text(j, i, str(state[i, j]), ha='center', va='center', color='white', fontsize=12)

        # Quitar ejes para que se vea más limpio
        plt.xticks([])
        plt.yticks([])
        plt.show()

    if __name__ == "__main__":
        
        # Inicialitzem el tauler 3x4
        np.zeros((3,4))

        # Estat inicial
        print_graella(np.zeros((3,4)))
