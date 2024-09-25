import cv2
import os
import numpy as np
from PIL import Image

# Função para dividir as imagens PNG em blocos de tamanho especificado e salvar como PNG
def divide_png(input_png_anotacao, input_png_rgb, output_dir, tile_size, original):
    # Ler as imagens usando OpenCV
    img_anotacao = cv2.imread(input_png_anotacao, cv2.IMREAD_GRAYSCALE)  # Ler como escala de cinza (para anotação)
    img_rgb = cv2.imread(input_png_rgb)  # Ler como imagem colorida (RGB)
    
    if img_anotacao is None or img_rgb is None:
        raise FileNotFoundError(f"Arquivo PNG não encontrado")

    # Obtém o tamanho da imagem original
    height, width = img_anotacao.shape  # Assumimos que as duas imagens têm o mesmo tamanho

    # Calcula o número de blocos
    x_tiles = (width // tile_size) + (1 if width % tile_size != 0 else 0)
    y_tiles = (height // tile_size) + (1 if height % tile_size != 0 else 0)

    # Itera sobre os blocos e cria subimagens
    for i in range(x_tiles):
        for j in range(y_tiles):
            # Define os offsets (posição de início) e o tamanho dos blocos
            x_offset = i * tile_size
            y_offset = j * tile_size
            x_block_size = min(tile_size, width - x_offset)
            y_block_size = min(tile_size, height - y_offset)

            # Lê o bloco de ambas as imagens (anotação e RGB)
            block_anotacao = img_anotacao[y_offset:y_offset+y_block_size, x_offset:x_offset+x_block_size]
            block_rgb = img_rgb[y_offset:y_offset+y_block_size, x_offset:x_offset+x_block_size]

            # Verifica as dimensões das arrays
            print(f"Dimensões do bloco RGB: {block_rgb.shape}")
            print(f"Dimensões do bloco Anotação: {block_anotacao.shape}")

            # Salvar o bloco RGB como PNG
            output_file_rgb = os.path.join(output_dir, f"tile_{original}_{i}_{j}.png")
            rgb_image = Image.fromarray(cv2.cvtColor(block_rgb, cv2.COLOR_BGR2RGB))  # Converter de BGR para RGB para o Pillow
            rgb_image.save(output_file_rgb)

            # Salvar o bloco de anotação como PNG
            output_file_anotacao = os.path.join(output_dir, f"tile_{original}_{i}_{j} anotacao.png")
            anotacao_image = Image.fromarray(block_anotacao)  # A anotação já está em escala de cinza
            anotacao_image.save(output_file_anotacao)

    print(f"Arquivo dividido com sucesso em blocos de {tile_size}x{tile_size}.")

# Caminho para os arquivos PNG e diretório de saída
input_png_anotacao = "nuvens_dilatada_mask_151.png"
input_png_rgb = "CBERS4A_WPM_PCA_RGB321_20240915_205_151.png"
output_dir = "tiles"
tile_size = 256

# Cria o diretório de saída se ele não existir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Divide os arquivos PNG
divide_png(input_png_anotacao, input_png_rgb, output_dir, tile_size, "RGB321_20240915_205_151")
