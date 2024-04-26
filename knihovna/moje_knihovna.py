

import numpy as np
from numpy.typing import NDArray


def nacist_data(nazev_souboru: str) -> NDArray[np.float64]:
    pocet_radku = sum(1 for _ in open(nazev_souboru))
    ceny = np.zeros(pocet_radku)

    with open(nazev_souboru, "r") as soubor:
        for index, radek in enumerate(soubor):
            datum, hodnota = radek.split("\t")
            ceny[index] = float(hodnota) 
    return ceny