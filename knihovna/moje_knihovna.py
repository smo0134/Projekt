

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


def investicni_strategie(
    ceny: NDArray[np.float64],
    obdobi: int,
    posun: int
) -> NDArray[np.float64]:
    opakovani = 0
    pocet = len(ceny)    
    while pocet > obdobi:
        pocet = pocet - posun
        opakovani = opakovani + 1

    denni_vynosy = np.zeros((obdobi-1, opakovani))
    mocnina = np.zeros((obdobi-1, opakovani))
    suma = np.zeros(opakovani)
    suma_mocnina = np.zeros(opakovani)

    for k in range(opakovani):
        for i in range(obdobi-1):
            denni_vynosy[i, k] = (ceny[(i+1)+k*20]-ceny[k*20+i]) / ceny[k*20+i]
            mocnina[i, k] = denni_vynosy[i, k]**2
            suma[k] += denni_vynosy[i, k]
            suma_mocnina[k] += mocnina[i, k]
    
    return denni_vynosy, mocnina, suma, suma_mocnina