from osgeo import gdal
import logging
import torch
import os
from PIL import Image
import numpy as np

def predict_full_image(net, image_path, device, output_dir="output"):
    """
    Realiza a previsão de uma imagem completa sem dividi-la em chunks e salva o resultado como PNG.
    
    Args:
        net (torch.nn.Module): Modelo treinado para fazer as previsões.
        image_path (str): Caminho para a imagem TIFF de entrada.
        device (str): Dispositivo para execução ('cpu' ou 'cuda').
        output_dir (str): Diretório para salvar o PNG gerado.
    
    Returns:
        str: Caminho do arquivo PNG salvo.
    """
    # Carregar a imagem TIFF completa usando GDAL
    dataset = gdal.Open(image_path)
    if dataset is None:
        logging.error(f"Erro: Não foi possível abrir o arquivo TIFF: {image_path}")
        return None
    
    # Ler a imagem como um array NumPy
    image = dataset.ReadAsArray()
    if image is None:
        logging.error(f"Erro: Não foi possível ler os dados da imagem TIFF: {image_path}")
        return None
    
    # Converter a imagem para um tensor do PyTorch e enviar para o dispositivo
    img = torch.from_numpy(image).unsqueeze(0).to(device=device, dtype=torch.float32)
    
    # Realizar a previsão
    with torch.no_grad():
        output = net(img).cpu()
        output = torch.nn.functional.interpolate(output, (image.shape[1], image.shape[2]), mode='bilinear')
        mask = output.argmax(dim=1).squeeze(0).numpy()
        
        # Criar uma imagem de máscara baseada nas classes (nuvens e sombras)
        mask_image = np.zeros_like(mask, dtype=np.uint8)
        mask_image[mask == 2] = 255  # Nuvens
        mask_image[mask == 1] = 127  # Sombras
    
    # Converter a máscara para uma imagem PNG e salvar
    os.makedirs(output_dir, exist_ok=True)
    output_png_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_mask.png")
    
    mask_image_pil = Image.fromarray(mask_image)
    mask_image_pil.save(output_png_path)
    
    logging.info(f"Máscara salva em {output_png_path}")
    
    return output_png_path


import glob

def predict_all_from_folder(net, input_folder, output_folder, device):
    """
    Realiza a previsão de todos os arquivos TIFF em uma pasta e salva os resultados como PNGs.
    
    Args:
        net (torch.nn.Module): Modelo treinado para fazer as previsões.
        input_folder (str): Caminho para a pasta contendo os arquivos TIFF.
        output_folder (str): Diretório para salvar os PNGs gerados.
        device (str): Dispositivo para execução ('cpu' ou 'cuda').
    
    Returns:
        list: Lista de caminhos dos arquivos PNG salvos.
    """
    # Buscar todos os arquivos TIFF na pasta de entrada
    tiff_files = glob.glob(os.path.join(input_folder, "*.tif"))
    if not tiff_files:
        logging.warning(f"Nenhum arquivo TIFF encontrado na pasta: {input_folder}")
        return []

    # Garantir que a pasta de saída exista
    os.makedirs(output_folder, exist_ok=True)

    saved_files = []
    for tiff_file in tiff_files:
        logging.info(f"Processando {tiff_file}...")
        
        # Realizar a previsão e salvar a máscara como PNG
        output_png = predict_full_image(net, tiff_file, device, output_dir=output_folder)
        if output_png:
            saved_files.append(output_png)

    logging.info(f"Processamento concluído. {len(saved_files)} arquivos PNG gerados.")
    return saved_files