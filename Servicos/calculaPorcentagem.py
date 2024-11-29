import cv2
import numpy as np

from Modelo.Tipos.Previsao import Estatistica

def calcular_cobertura(mascara_path):
    # Carregar a máscara em escala de cinza
    mascara = cv2.imread(mascara_path, cv2.IMREAD_GRAYSCALE)

    if mascara is None:
        raise ValueError(f"A máscara {mascara_path} não foi carregada corretamente. Verifique o arquivo.")

    # Valores definidos para as classes
    fundo_valor = 0
    sombra_valor = 127
    nuvem_valor = 254

    # Total de pixels na máscara
    total_pixels = mascara.size

    # Contar os pixels de cada classe
    nuvem_pixels = np.sum(mascara == nuvem_valor)
    sombra_pixels = np.sum(mascara == sombra_valor)
    fundo_pixels = np.sum(mascara == fundo_valor)

    # Calcular as porcentagens
    nuvem_percentual = (nuvem_pixels / total_pixels) * 100
    sombra_percentual = (sombra_pixels / total_pixels) * 100
    fundo_percentual = (fundo_pixels / total_pixels) * 100

    estatisticas = Estatistica(nuvem_percentual, sombra_percentual, fundo_percentual)

    return estatisticas