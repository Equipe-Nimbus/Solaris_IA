import cv2
import numpy as np
import sys
from Servicos.redimencionaImagem import resize_image_to_match
sys.path.append(r"c:/Users/BrunoDenardo/Desktop/Fatec/Solaris/IA/Solaris_IA")
from Tipos.Previsao import Estatistica


def calcular_cobertura(mascara_path):
    # Carregar a máscara com transparência
    mascara = cv2.imread(mascara_path, cv2.IMREAD_UNCHANGED)
    if mascara is None:
        raise ValueError("A máscara não foi carregada corretamente.")
    
    # Separar o canal alfa (transparência) se presente
    if mascara.shape[-1] == 4:  # Se BGRA
        canal_alpha = mascara[:, :, 3]
        mascara = mascara[:, :, 0]  # Usar o primeiro canal (escala de cinza)
        mascara[canal_alpha == 0] = 0  # Define pixels transparentes como fundo
    else:
        raise ValueError("A máscara não possui canal alfa para transparência.")

    # Obter altura e largura
    altura, largura = mascara.shape
    print(f"Dimensões da máscara: Altura={altura}, Largura={largura}")

    # Redimensionar a imagem principal
    resize_image_to_match("Modelo\\chunks\\resized_image.tif", "Modelo\\chunks\\resized_image.png", (largura, altura))

    # Carregar a imagem principal em escala de cinza
    imagem = cv2.imread("Modelo\\chunks\\resized_image.png", cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        raise ValueError("A imagem principal não foi carregada corretamente.")
    
    print(f"Dimensões da imagem principal: {imagem.shape}")

    # Pixels diferentes de 0 na imagem principal
    pixels_nao_nulos = np.sum(imagem > 0)
    print(f"Total de pixels não nulos na imagem principal: {pixels_nao_nulos}")

    # Pixels de nuvem (255) e sombra (128) na máscara
    nuvem_pixels = np.sum(mascara == 255)
    sombra_pixels = np.sum(mascara == 128)
    fundo_pixels = pixels_nao_nulos - (nuvem_pixels + sombra_pixels)
    print(f"Pixels de nuvem: {nuvem_pixels}")
    print(f"Pixels de sombra: {sombra_pixels}")
    print(f"Pixels de fundo: {fundo_pixels}")

    # Calcular porcentagens com base nos pixels não nulos da imagem principal
    if pixels_nao_nulos > 0:
        nuvem_percentual = (nuvem_pixels / pixels_nao_nulos) * 100
        sombra_percentual = (sombra_pixels / pixels_nao_nulos) * 100
        fundo_percentual = (fundo_pixels / pixels_nao_nulos) * 100
    else:
        raise ValueError("O número de pixels não nulos na imagem principal é zero.")
    
    print(f"Percentual de nuvem: {nuvem_percentual}")
    print(f"Percentual de sombra: {sombra_percentual}")
    print(f"Percentual de fundo: {fundo_percentual}")

    estatisticas = Estatistica(nuvem_percentual, sombra_percentual, fundo_percentual)
    return estatisticas


