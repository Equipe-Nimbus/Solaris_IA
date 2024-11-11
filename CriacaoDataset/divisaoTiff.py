from osgeo import gdal
import os

# Carregar a imagem TIFF original
nomeArquivo = "CBERS4A_WPM_PCA_RGB321_20241020_229_107"
input_file = f'CriacaoDataset/DataSet/originais/{nomeArquivo}.tif'
output_dir = 'CriacaoDataset/DataSet/imgs'
os.makedirs(output_dir, exist_ok=True)

# Abrir o arquivo com GDAL
dataset = gdal.Open(input_file)
img_width = dataset.RasterXSize
img_height = dataset.RasterYSize
tile_width = 1024  # Defina o tamanho desejado do bloco em pixels
tile_height = 1024

# Loop para cortar a imagem em blocos menores
for i in range(0, img_width, tile_width):
    for j in range(0, img_height, tile_height):
        # Definir os parâmetros de corte
        x_offset = i
        y_offset = j
        x_size = min(tile_width, img_width - i)
        y_size = min(tile_height, img_height - j)
        
        # Cortar e salvar o bloco
        output_file = os.path.join(output_dir, f'{nomeArquivo}_{i}_{j}.tif')
        gdal.Translate(output_file, dataset, srcWin=[x_offset, y_offset, x_size, y_size])

print("Divisão da imagem em blocos menores concluída!")
