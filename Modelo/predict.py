import os
import tempfile
import torch
import logging
import numpy as np
import time
from osgeo import gdal
from tqdm import tqdm
from PIL import Image
from Modelo.unet import UNet
import json
from shapely.geometry import Polygon, mapping
#from Servicos.geojsonToPNG import geojson_to_png
from Modelo.predict_thumbnail import predict_and_save

# Definir o caminho da pasta de arquivos provisórios
PROVISORY_FOLDER = "Modelo/chunks"

# Função para gerar nomes de arquivos de saída
def get_output_filenames(input_files):
    def _generate_name(filename):
        return f'{os.path.splitext(filename)[0]}_OUT.png'
    return list(map(_generate_name, input_files))

# Função para redimensionar um arquivo TIFF
def resize_tiff(tiff_file, new_width, new_height, output_file):
    try:
        src_ds = gdal.Open(tiff_file)
        if src_ds is None:
            logging.error(f"Erro: Não foi possível abrir o arquivo TIFF: {tiff_file}")
            return
        
        gdal.Warp(
            destNameOrDestDS=output_file,
            srcDSOrSrcDSTab=src_ds,
            width=new_width,
            height=new_height,
            resampleAlg=gdal.GRA_Bilinear,
            dstSRS=src_ds.GetProjection(),
            options=["-overwrite"]
        )
        logging.info(f"Imagem redimensionada salva em: {output_file}")
    except Exception as e:
        logging.error(f"Erro ao redimensionar a imagem: {e}")

# Função para converter uma máscara para GeoJSON
def mask_to_geojson_multiclass(mask, x_offset, y_offset, geotransform):
    features = []
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    
    # Verificar se o `mask` contém apenas zeros
    if np.all(mask == 0):
        logging.info("Mask contains only zeros; no features to process.")
        return features  # Retorna vazio sem entrar nos loops

    # Barra de progresso para as linhas (y)
    for y in tqdm(range(height), desc="Processing Rows"):
        x = 0
        while x < width:
            if x < width and y < height and mask[y, x] != 0 and not visited[y, x]:
                mask_value = mask[y, x]
                
                x_start = x
                # Barra de progresso para as colunas (x)
                with tqdm(total=width - x, desc="Processing Columns", leave=False) as pbar:
                    while x < width and mask[y, x] == mask_value and not visited[y, x]:
                        visited[y, x] = True
                        x += 1
                        pbar.update(1)
                x_end = x

                y_end = y
                while y_end < height and np.all(mask[y_end, x_start:x_end] == mask_value):
                    visited[y_end, x_start:x_end] = True
                    y_end += 1

                pixel_coords = [(x_start, y), (x_end, y), (x_end, y_end), (x_start, y_end)]
                geo_coords = []
                for px, py in pixel_coords:
                    geo_x = geotransform[0] + (px + x_offset) * geotransform[1] + (py + y_offset) * geotransform[2]
                    geo_y = geotransform[3] + (px + x_offset) * geotransform[4] + (py + y_offset) * geotransform[5]
                    geo_coords.append((geo_x, geo_y))
                

                if(int(mask_value)==127):
                    classe = "Shadow"
                elif(int(mask_value)==255):
                    classe = "Cloud"
                else:
                    classe = "Background"

                poly = Polygon(geo_coords)
                features.append({
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {"class": classe}
                })
            else:
                x += 1  # Avança `x` se o pixel for `0` ou já foi visitado

    return features

# Função para salvar um chunk como GeoJSON
def save_chunk_geojson(mask_chunk, x_offset, y_offset, geotransform, output_dir, chunk_index):
    features = mask_to_geojson_multiclass(mask_chunk, x_offset, y_offset, geotransform)
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    os.makedirs(PROVISORY_FOLDER, exist_ok=True)
    geojson_filename = os.path.join(PROVISORY_FOLDER, f"chunk_{chunk_index}.geojson")
    with open(geojson_filename, 'w', encoding='utf-8') as f:
        json.dump(geojson_data, f)

    return geojson_filename

# Função para salvar um chunk de imagem como PNG diretamente da máscara
def save_chunk_image_from_mask(mask_chunk, x_chunk, y_chunk, chunk_index):
    os.makedirs(PROVISORY_FOLDER, exist_ok=True)
    chunk_image_path = os.path.join(PROVISORY_FOLDER, f"chunk_{chunk_index}_preview.png")
    
    # Converte a máscara para uma imagem RGB
    chunk_image = np.zeros((y_chunk, x_chunk, 3), dtype=np.uint8)
    chunk_image[mask_chunk == 255] = [255, 255, 255]  # Nuvens em branco
    chunk_image[mask_chunk == 127] = [127, 127, 127]  # Sombras em cinza

    # Salva como PNG
    Image.fromarray(chunk_image).save(chunk_image_path)

    return chunk_image_path

# Função para fazer a predição de um chunk
def predict_chunk(net, chunk, device):
    img = torch.from_numpy(chunk).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img).cpu()
        output = torch.nn.functional.interpolate(output, (chunk.shape[1], chunk.shape[2]), mode='bilinear')
        mask = output.argmax(dim=1).squeeze(0).numpy()

        mask_image = np.zeros_like(mask, dtype=np.uint8)
        mask_image[mask == 2] = 255  # Nuvens
        mask_image[mask == 1] = 127  # Sombras
        return mask_image

# Função para processar e salvar um chunk
def process_and_save_chunk(net, chunk, x_offset, y_offset, device, output_dir, chunk_index, geotransform):
    logging.info(f"Processing chunk {chunk_index}, x_offset: {x_offset}, y_offset: {y_offset}")
    
    if np.min(chunk) == 0 and np.max(chunk) == 0:
        logging.info(f"Chunk {chunk_index} contém apenas zeros.")
        return None

    mask_chunk = predict_chunk(net, chunk, device)
    geojson_filename = save_chunk_geojson(mask_chunk, x_offset, y_offset, geotransform, output_dir, chunk_index)
    png_preview_path = save_chunk_image_from_mask(mask_chunk, mask_chunk.shape[1], mask_chunk.shape[0], chunk_index)
    
    return geojson_filename, png_preview_path

# Processamento sequencial de chunks (sem paralelismo)
def process_large_tiff_and_save_chunks(tiff_file, model, chunk_size=(1024, 1024), device='cpu', resize_factor=0.5):
    dataset = gdal.Open(tiff_file)
    if dataset is None:
        logging.error(f"Erro: Não foi possível abrir o arquivo TIFF: {tiff_file}")
        return [], (0, 0)

    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize
    geotransform = dataset.GetGeoTransform()

    new_width = int(x_size * resize_factor)
    new_height = int(y_size * resize_factor)
    resized_tiff = os.path.join(PROVISORY_FOLDER, "resized_image.tif")
    resize_tiff(tiff_file, new_width, new_height, resized_tiff)

    dataset = gdal.Open(resized_tiff)
    if dataset is None:
        logging.error(f"Erro: Não foi possível abrir o arquivo TIFF redimensionado: {resized_tiff}")
        return [], (0, 0)

    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize

    net = UNet(n_channels=3, n_classes=3, bilinear=False)
    net.to(device=device)
    state_dict = torch.load(model, map_location=device)
    net.load_state_dict(state_dict['model_state_dict'])

    chunk_paths = []
    preview_paths = []
    chunk_index = 0

    for x_offset in tqdm(range(0, x_size, chunk_size[0]), desc="Processing Chunks"):
        for y_offset in range(0, y_size, chunk_size[1]):
            start_time = time.time()

            x_chunk = min(chunk_size[0], x_size - x_offset)
            y_chunk = min(chunk_size[1], y_size - y_offset)

            chunk = dataset.ReadAsArray(x_offset, y_offset, x_chunk, y_chunk)
            result = process_and_save_chunk(net, chunk, x_offset, y_offset, device, PROVISORY_FOLDER, chunk_index, geotransform)
            
            if result:
                geojson_filename, png_preview_path = result
                chunk_paths.append(geojson_filename)
                preview_paths.append(png_preview_path)
            chunk_index += 1

            logging.info(f"Chunk {chunk_index} processed in {time.time() - start_time:.2f} seconds")

    return chunk_paths, preview_paths, (new_width, new_height)

def combine_and_resize_chunks(preview_paths, output_path, tiff_file, final_size=(1024, 1024), chunk_size=(1024, 1024), resize_factor=1.0):
    # Carregar o arquivo TIFF original para obter as dimensões
    dataset = gdal.Open(tiff_file)
    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize
    new_width = int(x_size * resize_factor)
    new_height = int(y_size * resize_factor)

    # Calcular o número de chunks em x e y
    num_chunks_x = (new_width + chunk_size[0] - 1) // chunk_size[0]
    num_chunks_y = (new_height + chunk_size[1] - 1) // chunk_size[1]

    # Criar uma imagem grande para combinar os chunks
    combined_image = Image.new("RGB", (new_width, new_height))

    # Ordenar preview_paths para garantir que os chunks estejam na ordem correta
    preview_paths = sorted(preview_paths, key=lambda x: int(os.path.basename(x).split('_')[1]))

    # Loop para carregar cada chunk e posicioná-lo na imagem grande
    for i, chunk_path in enumerate(preview_paths):
        # Carregar o chunk
        chunk_image = Image.open(chunk_path)

        
        # Calcular a posição (x, y) onde o chunk deve ser colado na imagem final
        print("x_offset: " + str(i % num_chunks_x) + " Calculo: " + str(chunk_image.width))
        print("y_offset: " + str(i % num_chunks_x) + " Calculo: " + str(chunk_image.height))
        x_offset = (i % num_chunks_x) * chunk_image.width
        y_offset = (i // num_chunks_x) * chunk_image.height

        # Colar o chunk redimensionado na imagem grande
        combined_image.paste(chunk_image, (y_offset, x_offset))

    # Redimensionar a imagem combinada para o tamanho final de 1024x1024
    resized_image = combined_image.resize(final_size, Image.LANCZOS)

    # Salva a imagem final consolidada
    resized_image.save(output_path)
    print(f"Imagem PNG final de 1024x1024 salva em: {output_path}")

    return f"http://localhost:8080/view/{os.path.basename(output_path)}"

# Função principal para criar a máscara, PNG preview e GeoJSON
def run_predict(model, input, output, no_save, mask_threshold, refactor_size, bilinear, classes, avaliacao):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if isinstance(input, str):
        input = [input]

    if os.path.isdir(input[0]):
        filenames = os.listdir(input[0])
        in_files = [os.path.join(input[0], filename) for filename in filenames if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
    else:
        in_files = input

    out_files = get_output_filenames(input)
    net = UNet(n_channels=3, n_classes=classes, bilinear=bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f'Loading model on device {device}')
    net.to(device=device)
    state_dict = torch.load(model, map_location=device)
    net.load_state_dict(state_dict['model_state_dict'])

    geojson_list = []
    pngsList = []
    download_links = []

    if(avaliacao):
        pngsList, download_links = predict_and_save(input[0], model, output, device)
    else:
        for i, filename in enumerate(in_files):
            logging.info(f'Predicting image {filename} ...')
            chunk_paths, preview_paths, new_size = process_large_tiff_and_save_chunks(
                tiff_file=filename,
                model=model,
                chunk_size=(1024, 1024),
                device=device,
                resize_factor=refactor_size
            )

            # Salvar o GeoJSON final consolidado
            print("Output original: ",filename)
            print("Output exterior: ", output)
            geojson_output = os.path.join(output, f"{filename.replace('.tif', '.geojson')}".split("\\")[-1])
            with open(geojson_output, 'w', encoding='utf-8') as f:
                geojson_data = {"type": "FeatureCollection", "features": []}
                for chunk_path in chunk_paths:
                    with open(chunk_path, 'r') as chunk_file:
                        chunk_data = json.load(chunk_file)
                        geojson_data["features"].extend(chunk_data["features"])
                json.dump(geojson_data, f)
            logging.info(f'GeoJSON saved to {geojson_output}')

            geojson_list.append(geojson_output)
            download_links.append(f"http://localhost:8080/download/{os.path.basename(geojson_output)}")

            png_output = os.path.join(output, f"{filename.replace('.tif', '.png')}".split('/')[-1].split("\\")[-1])
            preview_links = combine_and_resize_chunks(preview_paths, png_output, filename, (1024,1024), (1024,1024), refactor_size)
            pngsList.append(preview_links)

    return {"pngs": pngsList, "download_links": download_links}
