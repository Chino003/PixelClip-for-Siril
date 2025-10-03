# PixelClip_siril.py
# PixelClip per Siril 1.4 — replaces pixels < threshold with mean of neighbors >= threshold
# Salva questo file in una cartella a piacere e aggiungi la cartella a Siril (Preferences -> Scripts)

import sirilpy as s
import numpy as np

# opzionale: dialog semplice con tkinter per chiedere la soglia (se presente nel venv)
def ask_threshold(default=0.0001):
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        val = simpledialog.askfloat("PixelClip", "PixelClip threshold (es. 0.0001):", initialvalue=default)
        root.destroy()
        return val if val is not None else default
    except Exception:
        # fallback: nessun dialog disponibile -> valore di default
        return default

def _pixelclip_channel(data, threshold):
    """
    data: 2D numpy array (H, W) float
    threshold: float
    restituisce un array modificato dove i pixel < threshold sono
    sostituiti con la media dei vicini >= threshold (finestra 3x3).
    """
    # pad con 'edge' per gestire bordi come nell'esempio originale
    padded = np.pad(data, ((1,1),(1,1)), mode='edge')
    neighs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighs.append(padded[1+dy : 1+dy+data.shape[0],
                                 1+dx : 1+dx+data.shape[1]])
    neighs = np.stack(neighs, axis=0)   # shape (9, H, W)

    # consideriamo solo i vicini >= threshold
    mask = neighs >= threshold
    sums = np.where(mask, neighs, 0.0).sum(axis=0)   # somma dei vicini validi
    counts = mask.sum(axis=0)                        # numero di vicini validi

    # media (se counts==0 -> mettiamo la soglia come fallback)
    with np.errstate(divide='ignore', invalid='ignore'):
        means = np.where(counts > 0, sums / counts, threshold)

    out = data.copy()
    replace = data < threshold
    out[replace] = means[replace]
    return out

def pixelclip_on_array(arr, threshold):
    """
    arr: numpy array (H,W) o (H,W,C)
    restituisce array modificato
    """
    if arr.ndim == 2:
        return _pixelclip_channel(arr, threshold)
    elif arr.ndim == 3:
        out = arr.copy()
        for c in range(arr.shape[2]):
            out[:, :, c] = _pixelclip_channel(arr[:, :, c], threshold)
        return out
    else:
        raise ValueError("Unexpected image dimensions: {}".format(arr.shape))

def main():
    siril = s.SirilInterface()
    try:
        siril.connect()
    except Exception as e:
        print("Errore di connessione a Siril:", e)
        return

    threshold = ask_threshold(0.0001)
    siril.log(f"PixelClip: threshold = {threshold}")

    try:
        with siril.image_lock():
            img = siril.get_image()
            if img is None:
                siril.log("Nessuna immagine caricata in Siril. Apri un'immagine e riprova.")
                return

            data = img.data  # numpy array: shape (H,W) o (H,W,C)
            siril.log(f"Immagine: {img.width} x {img.height}, channels={img.channels}")

            newdata = pixelclip_on_array(data, threshold)

            # aggiorna l'immagine in Siril (sostituisce i pixel correnti)
            siril.set_image_pixeldata(newdata)

            siril.log("PixelClip: elaborazione completata. Salva manualmente l'immagine se vuoi.")
    except Exception as e:
        siril.log(f"PixelClip: errore durante l'elaborazione: {e}")

if __name__ == "__main__":
    main()
