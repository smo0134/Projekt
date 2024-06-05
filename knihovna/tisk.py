

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mpld3
import moje_knihovna
from numpy.typing import NDArray
from typing import List

# Prozatim nefunkcni nastrel

def ulozit_html(
    ceny: List[NDArray[np.float64]],
    obdobi: int,
    posun: int,
    nazev: str,
    html_soubor: str
) -> None:
    # Základní údaje
    pocet = len(ceny[0])
    opak = moje_knihovna.opakovani(pocet, obdobi, posun)
    
    # Výpočty pro každou cenovou řadu
    strategicke_vysledky = [moje_knihovna.investicni_strategie(c, obdobi, posun) for c in ceny]
    denni_vynosy_vse = [sv[0] for sv in strategicke_vysledky]
    mocnina_vynosy_vse = [sv[1] for sv in strategicke_vysledky]
    mesicni_vynosy_vse = [sv[2] for sv in strategicke_vysledky]
    mesicni_rizika_vse = [sv[3] for sv in strategicke_vysledky]
    stredni_vynosy_vse = [sv[4] for sv in strategicke_vysledky]
    rizika_vse = [sv[5] for sv in strategicke_vysledky]
    
    # Sestavení tabulek
    mesicni_vynosy_df = pd.DataFrame(mesicni_vynosy_vse).T
    mesicni_rizika_df = pd.DataFrame(mesicni_rizika_vse).T

    # Grafy
    grafy = []
    for i, cena in enumerate(ceny):
        fig, ax = plt.subplots()
        ax.plot(cena)
        ax.set_xlabel("Čas")
        ax.set_ylabel("Cena")
        ax.set_title(f"Cena {i + 1}")
        grafy.append(mpld3.fig_to_html(fig))
        plt.close(fig)

    # Optimalizace
    matice = moje_knihovna.tvorba_matice(ceny, obdobi, posun)
    kovariancni_matice_vse = moje_knihovna.kovariancni_matice(matice, obdobi, posun)
    inverze_vse = moje_knihovna.inverze(kovariancni_matice_vse)
    c_vse = moje_knihovna.c(inverze_vse, obdobi, posun)
    opt_vynos_vse = moje_knihovna.optimalizace_vynos(mesicni_vynosy_vse, c_vse)
    opt_riziko_vse = moje_knihovna.optimalizace_riziko(matice, c_vse)

    # Sestavení HTML stránky
    html_content = f"""
    <html>
    <head>
        <title>{nazev}</title>
    </head>
    <body>
        <h1>{nazev}</h1>
        <h2>Období: {obdobi}, Posun: {posun}</h2>
        
        <h2>Grafy</h2>
        {"".join(grafy)}
        
        <h2>Měsíční výnosy</h2>
        {mesicni_vynosy_df.to_html(index=False)}
        
        <h2>Měsíční rizika</h2>
        {mesicni_rizika_df.to_html(index=False)}
        
        <h2>Optimalizované výsledky</h2>
        <p>Optimalizovaný výnos: {np.array2string(opt_vynos_vse, precision=2)}</p>
        <p>Optimalizované riziko: {np.array2string(opt_riziko_vse, precision=2)}</p>
    </body>
    </html>
    """

    with open(html_soubor, 'w', encoding='utf-8') as file:
        file.write(html_content)
        
 