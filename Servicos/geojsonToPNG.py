import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Define o backend como Agg, que é seguro para renderização sem interface gráfica
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as mplPolygon

def geojson_to_png(geojson_file, output_png, output_size=(1024, 1024)):
    # Carrega o GeoJSON
    with open(geojson_file, 'r') as f:
        geojson_data = json.load(f)

    # Define as cores para cada classe ("Cloud" = branco, "Shadow" = cinza)
    colors = {"Cloud": (1, 1, 1, 1), "Shadow": (0.5, 0.5, 0.5, 1)}  # Branco e Cinza
    default_color = (0, 0, 0, 0)  # Transparente para o fundo

    # Coleta os limites de todas as coordenadas para ajustar a escala
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    for feature in geojson_data['features']:
        coordinates = feature['geometry']['coordinates'][0]
        for x, y in coordinates:
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x), max(max_y, y)  # Corrigido: x e y separados

    # Cria a figura e os eixos ajustados para o tamanho desejado
    fig, ax = plt.subplots(figsize=(output_size[0] / 100, output_size[1] / 100), facecolor='none')
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal')
    ax.set_facecolor((0, 0, 0, 0))  # Define o fundo da imagem como transparente

    # Itera sobre cada feature e desenha o polígono
    for feature in geojson_data['features']:
        class_value = feature['properties'].get('class', None)
        coordinates = feature['geometry']['coordinates'][0]
        
        # Define a cor com base na classe
        color = colors.get(class_value, default_color)
        
        # Adiciona o polígono ao gráfico sem borda
        polygon = mplPolygon(coordinates, closed=True, facecolor=color, edgecolor='none')
        ax.add_patch(polygon)

    # Remove os eixos para deixar a imagem limpa
    ax.axis('off')

    # Salva a figura como PNG com fundo transparente
    plt.savefig(output_png, transparent=True, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)

    print(f"Imagem salva em: {output_png} com tamanho {output_size[0]}x{output_size[1]}")
    return output_png

""" 
# Exemplo de uso
geojson_file = 'preview/CBERS4A_WPM_PCA_RGB321_20240924_197_123.geojson'
output_png = 'preview/saida_arquivo_resized_1024x1024.png'
geojson_to_png(geojson_file, output_png, output_size=(1024, 1024)) """
