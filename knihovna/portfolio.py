

from . import akcie
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
        Vypočítá kovarianci mezi dvěmi sadami cen.

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
        Vypočítá korelaci mezi dvěmi sadami cen.

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
    kovariance: List[NDArray[np.float64]], obdobi: int, posun: int
) -> List[NDArray[np.float64]]:
    """
        Vytvoří matici z hodnot kovarianci.

        Args:
            kovariance: List[NDArray[np.float64]]: Seznam hodnot kovarianci.
            obdobi (int): Délka období.
            posun (int): Posun mezi obdobími.

        Returns:
           Lists[NDArray[np.float64]]: Seznam matic obsahující kovariance.
    """
    pocet = len(kovariance)
    a = int(np.sqrt(pocet))
    opak = len(kovariance[0])
    vsechny_matice = []
    for k in range(opak):
        matice = np.zeros((a, a))
        for i in range(a):
            for j in range(a):
                matice[i, j] = kovariance[a * i + j][k]
        vsechny_matice.append(matice)
    return vsechny_matice


def kovariancni_matice(
        matice: List[NDArray[np.float64]], obdobi: int, posun: int
) -> List[NDArray[np.float64]]:
    """
        Vytvoří kovarianční matice obsahující prvky - prvek_ij = 2*kov_ij

        Args:
            matice: List[NDArray[np.float64]]: Seznam matic
            obsahující kovariance.
            obdobi (int): Délka období.
            posun (int): Posun mezi obdobími.

        Returns:
           Lists[NDArray[np.float64]]: Seznam kovariančních matic.
    """
    pocet = len(matice[0])
    velikost = len(matice)
    vsechny_matice = []
    for k in range(velikost):
        matice_kovariance = np.zeros((pocet + 1, pocet + 1))
        for i in range(pocet + 1):
            for j in range(pocet + 1):
                if i != pocet and j != pocet:
                    matice_kovariance[i, j] = 2 * matice[k][i, j]
                else:
                    matice_kovariance[i, j] = 1
                if i == pocet and j == pocet:
                    matice_kovariance[i, j] = 0
        vsechny_matice.append(matice_kovariance)
    return vsechny_matice


def inverze(matice: List[NDArray[np.float64]]) -> List[NDArray[np.float64]]:
    """
        Vytvoří inverze kovariančních matic.

        Args:
            matice: List[NDArray[np.float64]]: Seznam kovariančních matic.

        Returns:
           Lists[NDArray[np.float64]]: Seznam inverzních matic.
    """
    vsechny_matice = []
    for i in range(len(matice)):
        vsechny_matice.append(np.linalg.inv(matice[i]))
    return vsechny_matice


def c(
    inverze: List[NDArray[np.float64]],
    obdobi: int,
    posun: int
) -> List[NDArray[np.float64]]:
    """
        Vytvoří hodnoty c, pro ktere platí
        c = inverzní matice[len(matice),:len(matice)-1].
        Tedy poslední řádek inverzní matice kromě posledního prvku.
        Pro optimalizované riziko a jemu příslušný výnos,
        považujeme c za váhy, tedy w_i = c_i.

        Args:
            inverze: List[NDArray[np.float64]]: Seznam inverzních matic.
            obdobi (int): Délka období.
            posun (int): Posun mezi obdobími.

        Returns:
           Lists[NDArray[np.float64]]: Seznam hodnot c.
    """
    c = []
    for i in range(len(inverze)):
        c.append(inverze[i][len(inverze[i]) - 1, :len(inverze[i]) - 1])
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
    """
        Vypočítá hodnoty d. Pomocné hodnoty pro
        výpočty portfolií s daným rizikem a daným výnosem.

        Args:
            inverze: List[NDArray[np.float64]]: Seznam inverzních matic.
            ms: List[NDArray[np.float64]]: Seznam měsíčních výnosů.

        Returns:
            NDArray[np.float64]: Pole obsahující hodnoty d.
    """
    d = np.zeros(len(inverze))
    for i in range(len(inverze)):
        for j in range(len(inverze[i]) - 1):
            for k in range(len(inverze[i]) - 1):
                d[i] += inverze[i][j, k] * ms[j][i] * ms[k][i]
    return d


def di(
    inverze: List[NDArray[np.float64]],
    ms: List[NDArray[np.float64]],
) -> List[NDArray[np.float64]]:
    """
        Vypočítá hodnoty di. Pomocné hodnoty pro výpočty
        portfolií s daným rizikem a daným výnosem.

        Args:
            inverze: List[NDArray[np.float64]]: Seznam inverzních matic.
            ms: List[NDArray[np.float64]]: Seznam měsíčních výnosů.

        Returns:
            List[NDArray[np.float64]]: Seznam hodnot di.
    """
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
    """
        Vypočítá hodnoty pararametru pro daný výnos.
        Parametr je potřebný pro výpočet vah pro portfolio s daným výnosem.

        Args:
            optvyn (NDArray[np.float64]): Pole obsahující
            optimalizované výnosy.
            d (NDArray[np.float64]): Pole obsahující hodnoty d.
            vynos (float): Požadovaný výnos.

        Returns:
            NDArray[np.float64]: Pole obsahující hodnoty parvyn.
    """
    parvyn = np.zeros(len(optvyn))
    for i in range(len(optvyn)):
        parvyn[i] = (optvyn[i] - vynos) / d[i]
    return parvyn


def vahy(
    c: List[NDArray[np.float64]],
    di: List[NDArray[np.float64]],
    parvyn: NDArray[np.float64]
) -> List[NDArray[np.float64]]:
    """
        Vypočítá váhy pro portfolio s daným rizikem a daným výnosem.

        Args:
            c (List[NDArray[np.float64]]): Seznam hodnot c.
            di (List[NDArray[np.float64]]): Seznam hodnot di.
            parvyn (NDArray[np.float64]): Pole obsahující hodnoty parvyn.

        Returns:
            List[NDArray[np.float64]]: Seznam vah.
    """
    vahy = []
    for i in range(len(c)):
        pole = np.zeros(len(c[i]))
        for j in range(len(c[i])):
            pole[j] = c[i][j] + di[i][j] * parvyn[i]
        vahy.append(pole)
    return vahy


def riziko_dan(
        vahy: List[NDArray[np.float64]],
        matice: List[NDArray[np.float64]]
) -> NDArray[np.float64]:
    """
        Vypočítá riziko pro daný výnos.

        Args:
            vahy (List[NDArray[np.float64]]): Seznam vah.
            matice (List[NDArray[np.float64]]): Seznam kovariančních matic.

        Returns:
            NDArray[np.float64]: Pole obsahující riziko.
    """
    rizikodan = np.zeros(len(vahy))
    for i in range(len(vahy)):
        sum1 = 0
        sum2 = 0
        for j in range(len(vahy[i])):
            sum1 += (vahy[i][j]**2) * (matice[i][j, j]**2)
        for k in range(len(vahy[i]) - 1):
            for m in range(k + 1, len(vahy[i])):
                sum2 += vahy[i][k] * vahy[i][m] * matice[i][k, m]
        rizikodan[i] = np.sqrt(sum1 - 2 * sum2)
    return rizikodan


def par_riz(
    optriz: NDArray[np.float64],
    d: NDArray[np.float64],
    riziko: float
) -> NDArray[np.float64]:
    """
        Vypočítá hodnoty pararametru pro dané riziko.
        Parametr je potřebný pro výpočet vah pro portfolio s daným rizikem.

        Args:
            optriz (NDArray[np.float64]): Pole obsahující
            optimalizovaná rizika.
            d (NDArray[np.float64]): Pole obsahující hodnoty d.
            riziko (float): Požadované riziko.

        Returns:
            NDArray[np.float64]: Pole obsahující hodnoty parriz.
    """
    parriz = np.zeros(len(optriz))
    for i in range(len(optriz)):
        parriz[i] = (2 * (optriz[i] - riziko) / d[i])**2
    return parriz


def vynos_dan(
    vahy: List[NDArray[np.float64]],
    vynosy: List[NDArray[np.float64]]
) -> NDArray[np.float64]:
    """
        Vypočítá výnos pro dané riziko.

        Args:
            váhy (List[NDArray[np.float64]]): Seznam vah.
            vynosy (List[NDArray[np.float64]]): Seznam výnosů.

        Returns:
            NDArray[np.float64]: Pole obsahující výnos.
    """
    vynosdan = np.zeros(len(vahy))
    for i in range(len(vahy)):
        for j in range(len(vahy[i])):
            vynosdan[i] += vahy[i][j] * vynosy[j][i]
    return vynosdan
