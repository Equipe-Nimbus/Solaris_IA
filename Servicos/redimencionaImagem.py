from PIL import Image
from osgeo import gdal
import os

def resize_image_to_match(tiff_path, output_png_path, target_size):

    # Abre o arquivo TIFF usando GDAL
    source_ds = gdal.Open(tiff_path)
    if source_ds is None:
        raise FileNotFoundError(f"Arquivo não encontrado: {tiff_path}")
    
    # Converte os dados do TIFF para uma imagem Pillow
    source_array = source_ds.ReadAsArray()
    if source_array.ndim == 3:  # Verifica se há mais de uma banda
        source_array = source_array[0]  # Pega a primeira banda (se necessário)
    
    image = Image.fromarray(source_array.astype('uint8'))
    
    # Salva como PNG
    image.save(output_png_path, format="PNG")
    print(f"Imagem salva como PNG em: {output_png_path}")
    
    # Abre o PNG gerado para redimensionar
    png_image = Image.open(output_png_path)
    
    # Redimensiona a imagem
    resized_image = png_image.resize(target_size, Image.Resampling.LANCZOS)
    resized_image.save(output_png_path, format="PNG")
