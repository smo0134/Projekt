

import moje_knihovna
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt


def tisk_vyvoj(
    vystup: str,
    nazev: str,
    ceny: NDArray[np.float64]
):
    plt.plot(ceny)
    plt.xlabel("Čas")
    plt.ylabel("Cena")
    plt.title(nazev + " vyvoj ceny")
    plt.tight_layout()
    plt.show()



    