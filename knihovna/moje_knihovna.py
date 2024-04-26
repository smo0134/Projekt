

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def nacist_data(nazev_souboru: str) -> NDArray[np.float64]:
    pocet_radku = sum(1 for _ in open(nazev_souboru))
    ceny = np.zeros(pocet_radku)

    with open(nazev_souboru, "r") as soubor:
        for index, radek in enumerate(soubor):
            datum, hodnota = radek.split("\t")
            ceny[index] = float(hodnota)
    return ceny


def opakovani(pocet, obdobi, posun):
    opakovani = 0
    while pocet > obdobi:
        pocet = pocet - posun
        opakovani = opakovani + 1
    return opakovani


def denni_vynosy(ceny, opakovani, obdobi, posun):
    denni_vynosy = np.zeros((obdobi - 1, opakovani))
    for k in range(opakovani):
        for i in range(obdobi - 1):
            current_index = k * posun + i
            next_index = (i + 1) + k * posun
            denominator = ceny[current_index]
            denni_vynosy[i, k] = (ceny[next_index] - denominator) / denominator
    return denni_vynosy


def mocnina_denni_vynosy(ceny, opakovani, obdobi, posun):
    mocnina = np.zeros((obdobi - 1, opakovani))
    for k in range(opakovani):
        for i in range(obdobi - 1):
            current_index = k * posun + i
            next_index = (i + 1) + k * posun
            denominator = ceny[current_index]
            mocnina[i, k] = ((ceny[next_index]-denominator)/denominator)**2
    return mocnina


def suma_denni_vynosy(ceny, opakovani, obdobi, posun):
    suma = np.zeros(opakovani)
    for k in range(opakovani):
        for i in range(obdobi - 1):
            current_index = k * posun + i
            next_index = (i + 1) + k * posun
            denominator = ceny[current_index]
            suma[k] += ((ceny[next_index] - denominator) / denominator)
    return suma


def suma_mocnina_denni_vynosy(ceny, opakovani, obdobi, posun):
    suma_mocnina = np.zeros(opakovani)
    for k in range(opakovani):
        for i in range(obdobi - 1):
            current_index = k * posun + i
            next_index = (i + 1) + k * posun
            denominator = ceny[current_index]
            suma_mocnina[k] += (
                ((ceny[next_index] - denominator) / denominator) ** 2
            )
    return suma_mocnina


def stredni_vynosy(ceny, opakovani, obdobi, posun):
    stredni_vynosy = np.zeros(opakovani)
    for k in range(opakovani):
        for i in range(obdobi - 1):
            current_index = k * posun + i
            next_index = (i + 1) + k * posun
            denominator = ceny[current_index]
            stredni_vynosy[k] += (
                (ceny[next_index] - denominator) / denominator
            )
        stredni_vynosy[k] = (1 / obdobi) * stredni_vynosy[k]
    return stredni_vynosy


def rizika(ceny, opakovani, obdobi, posun):
    rizika = np.zeros(opakovani)
    for k in range(opakovani):
        for i in range(obdobi - 1):
            current_index = k * posun + i
            next_index = (i + 1) + k * posun
            denominator = ceny[current_index]
            rizika[k] += ((ceny[next_index] - denominator) / denominator) ** 2
        rizika[k] = np.sqrt(
            (1 / (obdobi - 1)) * rizika[k] -
            (1 / (obdobi * (obdobi - 1))) * rizika[k] ** 2
        )
    return rizika


def investicni_strategie(
    ceny: NDArray[np.float64], obdobi: int, posun: int
) -> NDArray[np.float64]:
    pocet = len(ceny)
    opak = opakovani(pocet, obdobi, posun)
    denni = denni_vynosy(ceny, opak, obdobi, posun)
    mocnina = mocnina_denni_vynosy(ceny, opak, obdobi, posun)
    suma = suma_denni_vynosy(ceny, opak, obdobi, posun)
    suma_mocnina = suma_mocnina_denni_vynosy(ceny, opak, obdobi, posun)
    stredni = stredni_vynosy(ceny, opak, obdobi, posun)
    riz = rizika(ceny, opak, obdobi, posun)
    return denni, mocnina, suma, suma_mocnina, stredni, riz


def vyvoj(ceny:NDArray[np.float64]):
    plt.plot(ceny)
    plt.xlabel("Čas")
    plt.ylabel("Cena")
    plt.title("Vývoj cen")
    plt.show()
