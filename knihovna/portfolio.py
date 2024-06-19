

import akcie
import numpy as np
from numpy.typing import NDArray
from typing import List


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
    d1, m1, s1, sm1, st1, r1 = akcie.investicni_strategie(ceny1, obdobi, posun)
    d2, m2, s2, sm2, st2, r2 = akcie.investicni_strategie(ceny2, obdobi, posun)
    mr1 = akcie.mesicni_rizika(r1)
    mr2 = akcie.mesicni_rizika(r2)
    opak = akcie.opakovani(len(ceny1), obdobi, posun)
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
    d1, m1, s1, sm1, st1, r1 = akcie.investicni_strategie(ceny1, obdobi, posun)
    d2, m2, s2, sm2, st2, r2 = akcie.investicni_strategie(ceny2, obdobi, posun)
    mr1 = akcie.mesicni_rizika(r1)
    mr2 = akcie.mesicni_rizika(r2)
    opak = akcie.opakovani(len(ceny1), obdobi, posun)
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
