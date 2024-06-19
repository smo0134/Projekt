

import numpy as np
from numpy.typing import NDArray


# nacteni dat z textoveho souboru
def nacist_data(nazev_souboru: str) -> NDArray[np.float64]:
    """
        Načte data z textového souboru.

        Args:
            nazev_souboru (str): Název textového souboru obsahujícího data.

        Returns:
            NDArray[np.float64]: Pole obsahující ceny.
    """
    pocet_radku = sum(1 for _ in open(nazev_souboru))
    ceny = np.zeros(pocet_radku)

    with open(nazev_souboru, "r") as soubor:
        for index, radek in enumerate(soubor):
            datum, hodnota = radek.split("\t")
            ceny[index] = float(hodnota)
    return ceny


# zjistovani poctu opakovani, pro vypocty
# aktivni/pasivni investicni strategie
def opakovani(pocet, obdobi, posun):
    """
        Zjistí počet opakování pro výpočty.

        Args:
            pocet (int): Celkový počet datových bodů.
            obdobi (int): Délka období.
            posun (int): Posun mezi obdobími.

        Returns:
            int: Počet opakování.
    """
    opakovani = 0
    while pocet >= obdobi:
        pocet = pocet - posun
        opakovani = opakovani + 1
    return opakovani


# vypocet dennich vynosu
def denni_vynosy(ceny, opakovani, obdobi, posun):
    """
        Vypočítá denní výnosy.

        Args:
            ceny (NDArray[np.float64]): Pole obsahující ceny.
            opakovani (int): Počet opakování.
            obdobi (int): Délka období.
            posun (int): Posun mezi obdobími.

        Returns:
            NDArray[np.float64]: Pole obsahující denní výnosy.
    """
    denni_vynosy = np.zeros((obdobi - 1, opakovani))
    for k in range(opakovani):
        for i in range(obdobi - 1):
            current_index = k * posun + i
            next_index = (i + 1) + k * posun
            denominator = ceny[current_index]
            denni_vynosy[i, k] = (ceny[next_index] - denominator) / denominator
    return denni_vynosy


# vypocet mocniny dennich vynosu
def mocnina_denni_vynosy(ceny, opakovani, obdobi, posun):
    """
        Vypočítá mocniny denních výnosů.

        Args:
            ceny (NDArray[np.float64]): Pole obsahující ceny.
            opakovani (int): Počet opakování.
            obdobi (int): Délka období.
            posun (int): Posun mezi obdobími.

        Returns:
            NDArray[np.float64]: Pole obsahující mocniny denních výnosů.
    """
    mocnina = np.zeros((obdobi - 1, opakovani))
    for k in range(opakovani):
        for i in range(obdobi - 1):
            current_index = k * posun + i
            next_index = (i + 1) + k * posun
            denominator = ceny[current_index]
            mocnina[i, k] = ((ceny[next_index]-denominator)/denominator)**2
    return mocnina


def suma_denni_vynosy(ceny, opakovani, obdobi, posun):
    """
        Vypočítá sumu denních výnosů.

        Args:
            ceny (NDArray[np.float64]): Pole obsahující ceny.
            opakovani (int): Počet opakování.
            obdobi (int): Délka období.
            posun (int): Posun mezi obdobími.

        Returns:
            NDArray[np.float64]: Pole obsahující sumu denních výnosů.
    """
    suma = np.zeros(opakovani)
    for k in range(opakovani):
        for i in range(obdobi - 1):
            current_index = k * posun + i
            next_index = (i + 1) + k * posun
            denominator = ceny[current_index]
            suma[k] += ((ceny[next_index] - denominator) / denominator)
    return suma


def suma_mocnina_denni_vynosy(ceny, opakovani, obdobi, posun):
    """
        Vypočítá sumu mocnin denních výnosů.

        Args:
            ceny (NDArray[np.float64]): Pole obsahující ceny.
            opakovani (int): Počet opakování.
            obdobi (int): Délka období.
            posun (int): Posun mezi obdobími.

        Returns:
            NDArray[np.float64]: Pole obsahující sumu mocnin denních výnosů.
    """
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
    """
        Vypočítá střední výnosy.

        Args:
            ceny (NDArray[np.float64]): Pole obsahující ceny.
            opakovani (int): Počet opakování.
            obdobi (int): Délka období.
            posun (int): Posun mezi obdobími.

        Returns:
            NDArray[np.float64]: Pole obsahující střední výnosy.
    """
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
    """
        Vypočítá rizika.

        Args:
            ceny (NDArray[np.float64]): Pole obsahující ceny.
            opakovani (int): Počet opakování.
            obdobi (int): Délka období.
            posun (int): Posun mezi obdobími.

        Returns:
            NDArray[np.float64]: Pole obsahující rizika.
    """
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
    """
        Vypočítá měsíční rizika.

        Args:
            rizika (NDArray[np.float64]): Pole obsahující rizika.

        Returns:
            NDArray[np.float64]: Pole obsahující měsíční rizika.
    """
    mesicni_rizika = np.zeros(len(rizika))
    for i in range(len(rizika)):
        mesicni_rizika[i] = rizika[i] * np.sqrt(20)
    return mesicni_rizika


def mesicni_vynosy(stredni_vynosy):
    """
        Vypočítá měsíční výnosy.

        Args:
            stredni_vynosy (NDArray[np.float64]):
            Pole obsahující střední výnosy.

        Returns:
            NDArray[np.float64]: Pole obsahující měsíční výnosy.
    """
    mesicni_vynosy = np.zeros(len(stredni_vynosy))
    for i in range(len(stredni_vynosy)):
        mesicni_vynosy[i] = stredni_vynosy[i] * 20
    return mesicni_vynosy


def investicni_strategie(
    ceny: NDArray[np.float64], obdobi: int, posun: int
) -> NDArray[np.float64]:
    """
        Vypočítá investiční strategii.

        Args:
            ceny (NDArray[np.float64]): Pole obsahující ceny.
            obdobi (int): Délka období.
            posun (int): Posun mezi obdobími.

        Returns:
            List[NDArray[np.float64]]: Seznam obsahující denní výnosy,
            mocniny dennich výnosů,
            měsíční výnosy, měsíční rizika, střední výnosy a rizika.
    """
    pocet = len(ceny)
    opak = opakovani(pocet, obdobi, posun)
    denni = denni_vynosy(ceny, opak, obdobi, posun)
    mocnina = mocnina_denni_vynosy(ceny, opak, obdobi, posun)
    stredni = stredni_vynosy(ceny, opak, obdobi, posun)
    riz = rizika(ceny, opak, obdobi, posun)
    mv = mesicni_vynosy(stredni)
    mr = mesicni_rizika(riz)
    return denni, mocnina, mv, mr, stredni, riz
