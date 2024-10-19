import os
import logging
import numpy as np
from osgeo import gdal
from PIL import Image

# Configurar o nível de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Função para inspecionar as bandas do arquivo TIFF
def inspect_tiff_bands(tiff_file):
    dataset = gdal.Open(tiff_file)
    if not dataset:
        logging.error("Erro ao abrir o arquivo TIFF.")
        return
    
    logging.info(f"Dimensões da imagem: {dataset.RasterXSize} x {dataset.RasterYSize}")
    logging.info(f"Número de bandas: {dataset.RasterCount}")
    
    for i in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(i)
        min_val, max_val = band.ComputeRasterMinMax(1)
        logging.info(f"Banda {i}: Min={min_val}, Max={max_val}, Tipo={gdal.GetDataTypeName(band.DataType)}")

# Função para ler um chunk específico da imagem TIFF
def read_chunk(tiff_file, x_offset, y_offset, x_chunk_size, y_chunk_size):
    dataset = gdal.Open(tiff_file)
    
    if not dataset:
        logging.error("Erro ao abrir o arquivo TIFF.")
        return None
    
    logging.info(f"Lendo chunk nas coordenadas x_offset={x_offset}, y_offset={y_offset}")
    
    chunk = dataset.ReadAsArray(x_offset, y_offset, x_chunk_size, y_chunk_size)
    if chunk is None:
        logging.error(f"Erro ao ler o chunk nas coordenadas ({x_offset}, {y_offset}).")
    else:
        logging.info(f"Valores do chunk: Min={np.min(chunk)}, Max={np.max(chunk)}")
    
    return chunk

# Função para salvar um chunk como imagem PNG
def save_chunk_image(chunk, chunk_index, output_dir="Modelo/arquivosProvisorios"):
    os.makedirs(output_dir, exist_ok=True)
    chunk_image_path = os.path.join(output_dir, f"chunk_{chunk_index}.png")
    
    # Converter o chunk para uma imagem PIL e salvar como PNG
    chunk_image = np.moveaxis(chunk, 0, -1)  # Mover os canais para o final (de [C, H, W] para [H, W, C])
    
    # Verificar se o chunk contém apenas valores baixos (muito escuros)
    logging.info(f"Salvando chunk {chunk_index}: Min={np.min(chunk_image)}, Max={np.max(chunk_image)}")
    
    chunk_image = Image.fromarray(chunk_image.astype(np.uint8))
    chunk_image.save(chunk_image_path)

    logging.info(f"Chunk salvo como imagem PNG: {chunk_image_path}")
    return chunk_image_path

# Função principal que integra todas as partes e solicita inputs
def main():
    # Input do usuário para o caminho do arquivo TIFF
    tiff_file = input("Digite o caminho do arquivo TIFF: ")
    
    # Verificar as bandas da imagem TIFF
    print("\nInspecionando as bandas da imagem TIFF...")
    inspect_tiff_bands(tiff_file)
    
    # Solicitar os parâmetros para ler um chunk específico
    x_offset = int(input("\nDigite o x_offset (posição X inicial): "))
    y_offset = int(input("Digite o y_offset (posição Y inicial): "))
    x_chunk_size = int(input("Digite o tamanho do chunk em X (largura): "))
    y_chunk_size = int(input("Digite o tamanho do chunk em Y (altura): "))
    
    # Ler o chunk solicitado
    chunk = read_chunk(tiff_file, x_offset, y_offset, x_chunk_size, y_chunk_size)
    
    # Salvar o chunk como imagem PNG, se os dados forem válidos
    if chunk is not None:
        chunk_index = 0  # Índice do chunk (ajustável)
        save_chunk_image(chunk, chunk_index)

# Chamar a função principal
if __name__ == '__main__':
    main()
