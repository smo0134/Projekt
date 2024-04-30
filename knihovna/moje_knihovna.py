

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List


def nacist_data(nazev_souboru: str) -> NDArray[np.float64]:
    pocet_radku = sum(1 for _ in open(nazev_souboru))
    ceny = np.zeros(pocet_radku)

    with open(nazev_souboru, "r") as soubor:
        for index, radek in enumerate(soubor):
            datum, hodnota = radek.split("\t")
            ceny[index] = float(hodnota)
    return ceny


def datum(nazev):
    datum = []
    with open(nazev, "r") as soubor:
        for index, radek in enumerate(soubor):
            cislo, _ = radek.split("\t")
            datum.append(cislo)
    return datum


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


def mesicni_rizika(rizika):
    mesicni_rizika = np.zeros(len(rizika))
    for i in range(len(rizika)):
        mesicni_rizika[i] = rizika[i] * np.sqrt(20)
    return mesicni_rizika


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


def tisk(ceny: NDArray[np.float64], nazev: str) -> None:
    plt.plot(ceny)
    plt.xlabel("Čas")
    plt.ylabel("Cena")
    plt.title(nazev + " vyvoj ceny")
    plt.tight_layout()
    plt.show()


def kovariance(
    ceny1: NDArray[np.float64],
    ceny2: NDArray[np.float64],
    obdobi: int,
    posun: int
) -> NDArray[np.float64]:
    d1, m1, s1, sm1, st1, r1 = investicni_strategie(ceny1, obdobi, posun)
    d2, m2, s2, sm2, st2, r2 = investicni_strategie(ceny2, obdobi, posun)
    mr1 = mesicni_rizika(r1)
    mr2 = mesicni_rizika(r2)
    opak = opakovani(len(ceny1), obdobi, posun)
    kov = np.zeros(opak)
    for k in range(opak):
        kov[k] = (d1[:, k] @ d2[:, k])
        kov[k] = (1 / (obdobi - 1)) * kov[k] - \
                 (1 / (obdobi * (obdobi - 1))) * mr1[k] * mr2[k]
    return kov


def korelace(
    ceny1: NDArray[np.float64],
    ceny2: NDArray[np.float64],
    obdobi: int,
    posun: int
) -> NDArray[np.float64]:
    kov = kovariance(ceny1, ceny2, obdobi, posun)
    d1, m1, s1, sm1, st1, r1 = investicni_strategie(ceny1, obdobi, posun)
    d2, m2, s2, sm2, st2, r2 = investicni_strategie(ceny2, obdobi, posun)
    mr1 = mesicni_rizika(r1)
    mr2 = mesicni_rizika(r2)
    opak = opakovani(len(ceny1), obdobi, posun)
    kor = np.zeros(opak)
    for k in range(opak):
        kor[k] = kov[k] / (mr1[k] * mr2[k])
    return kor


def tvorba_matice(
    ceny: List[NDArray[np.float64]], obdobi: int, posun: int
) -> List[NDArray[np.float64]]:
    pocet = len(ceny)
    a = int(np.sqrt(pocet))
    opak = len(ceny[0])
    vsechny_matice = []
    for k in range(opak):
        matice = np.zeros((a, a))
        for i in range(a):
            for j in range(a):
                matice[i, j] = ceny[a*i+j][k]
        vsechny_matice.append(matice)
    return vsechny_matice


def kovariancni_matice(
        matice: List[NDArray[np.float64]], obdobi: int, posun: int
) -> List[NDArray[np.float64]]:
    pocet = len(matice[0])
    velikost = len(matice)
    vsechny_matice = []
    for k in range(velikost):
        matice_kovariance = np.zeros((pocet+1, pocet+1))
        for i in range(pocet+1):
            for j in range(pocet+1):
                if i != pocet and j != pocet:
                    matice_kovariance[i, j] = 2*matice[k][i, j]
                else:
                    matice_kovariance[i, j] = 1
                if i == pocet and j == pocet:
                    matice_kovariance[i, j] = 0
        vsechny_matice.append(matice_kovariance)
    return vsechny_matice


def inverze(matice: List[NDArray[np.float64]]) -> List[NDArray[np.float64]]:
    vsechny_matice = []
    for i in range(len(matice)):
        vsechny_matice.append(np.linalg.inv(matice[i]))
    return vsechny_matice
