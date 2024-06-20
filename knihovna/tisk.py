

import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List
import numpy as np
import os


def tisk_grafu(x: NDArray, nazev: str,
               popisky: List[str], slozka: str) -> None:
    """
        Vytiskne spojnicový graf.

        Args:
            x (NDArray): Pole hodnot.
            nazev (str): Název grafu.
            popisky (List[str]): Popisky os.
            slozka (str): Cesta ke slozce, kde chceme uložit graf.
    """
    plt.plot(x)
    plt.xlabel(popisky[0])
    plt.ylabel(popisky[1])
    plt.title(nazev)
    plt.tight_layout()
    file_name = nazev + '.png'
    file_path = os.path.join(slozka, file_name)
    plt.savefig(file_path)
    plt.show()


def tisk_grafu_sloupec(x: NDArray, nazev: str,
                       popisky: List[str], slozka: str) -> None:
    """
        Vytiskne sloupcový graf.

        Args:
            x (NDArray): Pole hodnot.
            nazev (str): Název grafu.
            popisky (List[str]): Popisky os.
            slozka (str): Cesta ke slozce, kde chceme uložit graf.

    """
    indexy = np.arange(len(x))
    plt.bar(indexy, x)
    plt.xlabel(popisky[0])
    plt.ylabel(popisky[1])
    plt.title(nazev)
    plt.tight_layout()
    file_name = nazev + '.png'
    file_path = os.path.join(slozka, file_name)
    plt.savefig(file_path)
    plt.show()
