

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List


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


# zjistovani data
def datum(nazev):
    """
        Získá seznam dat z textového souboru.

        Args:
            nazev (str): Název textového souboru obsahujícího data.

        Returns:
            List[str]: Seznam dat.
    """
    datum = []
    with open(nazev, "r") as soubor:
        for index, radek in enumerate(soubor):
            cislo, _ = radek.split("\t")
            datum.append(cislo)
    return datum


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


def tisk(ceny: NDArray[np.float64], nazev: str) -> None:
    """
        Vytvoří graf vývoje ceny.

        Args:
            ceny (NDArray[np.float64]): Pole obsahující ceny.
            nazev (str): Název grafu.
    """
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
    """
        Vypočítá kovarianci mezi dvěma sadami cen.

        Args:
            ceny1 (NDArray[np.float64]): První pole obsahující ceny.
            ceny2 (NDArray[np.float64]): Druhé pole obsahující ceny.
            obdobi (int): Délka období.
            posun (int): Posun mezi obdobími.

        Returns:
            NDArray[np.float64]: Pole obsahující kovarianci.
    """
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
    """
        Vypočítá korelaci mezi dvěma sadami cen.

        Args:
            ceny1 (NDArray[np.float64]): První pole obsahující ceny.
            ceny2 (NDArray[np.float64]): Druhé pole obsahující ceny.
            obdobi (int): Délka období.
            posun (int): Posun mezi obdobími.

        Returns:
            NDArray[np.float64]: Pole obsahující korelaci.
    """
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


def c(
    inverze: List[NDArray[np.float64]],
    obdobi: int,
    posun: int
) -> List[NDArray[np.float64]]:
    c = []
    for i in range(len(inverze)):
        c.append(inverze[i][len(inverze[i])-1, :len(inverze[i])-1])
    return c


def optimalizace_vynos(
    mesicnivynosy: List[NDArray[np.float64]],
    c: List[NDArray[np.float64]]
) -> NDArray[np.float64]:
    """
        Vypočítá optimalizované výnosy.

        Args:
            mesicnivynosy (List[NDArray[np.float64]]): Seznam měsíčních výnosů.
            c (List[NDArray[np.float64]]): Seznam hodnot c.

        Returns:
            NDArray[np.float64]: Pole obsahující optimalizované výnosy.
    """
    optvynos = np.zeros(len(c))
    for i in range(len(c)):
        for j in range(len(c[i])):
            optvynos[i] = optvynos[i] + c[i][j] * mesicnivynosy[j][i]
    return optvynos


def optimalizace_riziko(
    matice: List[NDArray[np.float64]],
    c: List[NDArray[np.float64]]
) -> NDArray[np.float64]:
    """
        Vypočítá optimalizovaná rizika.

        Args:
            mesicnirizika (List[NDArray[np.float64]]): Seznam měsíčních rizik.
            c (List[NDArray[np.float64]]): Seznam hodnot c.

        Returns:
            NDArray[np.float64]: Pole obsahující optimalizovaná rizika.
    """
    optriziko = np.zeros(len(c))
    for i in range(len(c)):
        for j in range(len(c[i])):
            for k in range(len(c[i])):
                optriziko[i] += c[i][j] * c[i][k] * matice[j][j, k]
    return optriziko


def d(
    inverze: List[NDArray[np.float64]],
    ms: List[NDArray[np.float64]],
) -> NDArray[np.float64]:
    d = np.zeros(len(inverze))
    for i in range(len(inverze)):
        for j in range(len(inverze[i])-1):
            for k in range(len(inverze[i])-1):
                d[i] += inverze[i][j, k] * ms[j][i] * ms[k][i]
    return d


def di(
    inverze: List[NDArray[np.float64]],
    ms: List[NDArray[np.float64]],
) -> List[NDArray[np.float64]]:
    di = []
    for i in range(len(inverze)):
        pole = np.zeros(len(ms))
        for j in range(len(ms)):
            for k in range(len(ms)):
                pole[j] += inverze[i][j, k] * ms[k][i]
        di.append(pole)
    return di


def par_vynos(
    optvyn: NDArray[np.float64],
    d: NDArray[np.float64],
    vynos: float
) -> NDArray[np.float64]:
    parvyn = np.zeros(len(optvyn))
    for i in range(len(optvyn)):
        parvyn[i] = (optvyn[i] - vynos) / d[i]
    return parvyn


def vahy(
    c: List[NDArray[np.float64]],
    di: List[NDArray[np.float64]],
    parvyn: NDArray[np.float64]
) -> List[NDArray[np.float64]]:
    vahy = []
    for i in range(len(c)):
        pole = np.zeros(len(c[i]))
        for j in range(len(c[i])):
            pole[j] = c[i][j] + di[i][j] * parvyn[i]
        vahy.append(pole)
    return vahy


def riziko_dan(
        váhy: List[NDArray[np.float64]],
        matice: List[NDArray[np.float64]]
) -> NDArray[np.float64]:
    rizikodan = np.zeros(len(váhy))
    for i in range(len(váhy)):
        sum1 = 0
        sum2 = 0
        for j in range(len(váhy[i])):
            sum1 += (váhy[i][j]**2) * (matice[i][j, j]**2)
        for k in range(len(váhy[i])-1):
            for m in range(k+1, len(váhy[i])):
                sum2 += váhy[i][k] * váhy[i][m] * matice[i][k, m]
        rizikodan[i] = np.sqrt(sum1 - 2*sum2)
    return rizikodan


def par_riz(
    optriz: NDArray[np.float64],
    d: NDArray[np.float64],
    riziko: float
) -> NDArray[np.float64]:
    parriz = np.zeros(len(optriz))
    for i in range(len(optriz)):
        parriz[i] = (2*(optriz[i] - riziko) / d[i])**2
    return parriz


def vynos_dan(
    váhy: List[NDArray[np.float64]],
    vynosy: List[NDArray[np.float64]]
) -> NDArray[np.float64]:
    vynosdan = np.zeros(len(váhy))
    for i in range(len(váhy)):
        for j in range(len(váhy[i])):
            vynosdan[i] += váhy[i][j] * vynosy[j][i]
    return vynosdan