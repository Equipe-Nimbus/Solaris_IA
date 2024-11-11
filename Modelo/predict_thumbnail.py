import os
import torch
import numpy as np
from PIL import Image
import json
from shapely.geometry import Polygon, mapping
import geopandas as gpd
import matplotlib.pyplot as plt
from Modelo.unet import UNet

# Função para converter a máscara para GeoJSON
def mask_to_geojson(mask, geotransform):
    features = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if mask[y, x] != 0:
                mask_value = mask[y, x]
                pixel_coords = [(x, y), (x+1, y), (x+1, y+1), (x, y+1)]
                geo_coords = [(geotransform[0] + px * geotransform[1], geotransform[3] + py * geotransform[5]) for px, py in pixel_coords]
                
                poly = Polygon(geo_coords)
                features.append({
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {"class": int(mask_value)}
                })
    return features

# Função para converter GeoJSON para PNG
def geojson_to_png(geojson_file, output_png):
    gdf = gpd.read_file(geojson_file)
    colors = {255: 'white', 127: 'gray'}  # Definir cores para nuvens e sombras

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')

    for idx, row in gdf.iterrows():
        color = colors.get(row['properties']['class'], 'black')
        gdf.loc[[idx]].plot(ax=ax, color=color, edgecolor='k')

    ax.axis('off')
    plt.savefig(output_png, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)
    return output_png

# Função para fazer a previsão e salvar GeoJSON e PNG
def predict_and_save(input_files, model_path, output_folder, device='cpu'):
    # Carrega o modelo
    net = UNet(n_channels=3, n_classes=3, bilinear=False)
    net.to(device=device)
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict['model_state_dict'])
    net.eval()

    os.makedirs(output_folder, exist_ok=True)
    pngsList = []
    download_links = []

    for input_file in input_files:
        # Carrega a imagem
        img = Image.open(input_file).convert('RGB')
        img = np.array(img)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32)

        # Faz a previsão
        with torch.no_grad():
            output = net(img_tensor).cpu()
            mask = output.argmax(dim=1).squeeze(0).numpy()

        # Gera o GeoJSON a partir da máscara
        geotransform = [0, 1, 0, 0, 0, -1]  # Defina a transformada conforme necessário
        features = mask_to_geojson(mask, geotransform)
        geojson_data = {"type": "FeatureCollection", "features": features}

        # Salva o GeoJSON
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        geojson_output = os.path.join(output_folder, f"{base_name}.geojson")
        with open(geojson_output, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f)
        
        # Adiciona o link de download do GeoJSON
        download_link = f"http://localhost:8080/download/{os.path.basename(geojson_output)}"
        download_links.append(download_link)

        # Gera e salva o PNG a partir do GeoJSON
        png_output = os.path.join(output_folder, f"{base_name}_preview.png")
        geojson_to_png(geojson_output, png_output)
        pngsList.append(png_output)

        print(f"GeoJSON saved to {geojson_output}")
        print(f"Preview PNG saved to {png_output}")
        print(f"Download link: {download_link}")

    return pngsList, download_links

""" # Exemplo de uso
input_files = ['input_image1.png', 'input_image2.png']  # Lista de arquivos PNG
model_path = 'caminho_do_modelo.pth'  # Caminho para o modelo treinado
output_folder = 'output_directory'
pngsList, download_links = predict_and_save(input_files, model_path, output_folder) """
