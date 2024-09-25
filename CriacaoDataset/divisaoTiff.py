from osgeo import gdal
import os

# Função para dividir o TIFF em blocos de tamanho especificado
def divide_tiff(input_tiff_anotacao, input_tiff_rgb, output_dir, tile_size):
    datasetAnotacao = gdal.Open(input_tiff_anotacao)
    datasetRGB = gdal.Open(input_tiff_rgb)
    if (not datasetAnotacao) or (not datasetRGB):
        raise FileNotFoundError(f"Arquivo TIFF não encontrado")

    # Obtém o tamanho da imagem original
    width = datasetAnotacao.RasterXSize
    height = datasetAnotacao.RasterYSize

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
            block_anotacao = datasetAnotacao.ReadAsArray(x_offset, y_offset, x_block_size, y_block_size)
            block_rgb = datasetRGB.ReadAsArray(x_offset, y_offset, x_block_size, y_block_size)

            # Verifica as dimensões das arrays
            print(f"Dimensões do bloco RGB: {block_rgb.shape}")
            print(f"Dimensões do bloco Anotação: {block_anotacao.shape}")

            # Define o caminho do arquivo de saída
            output_file_anotacao = os.path.join(output_dir, f"tile_{i}_{j}_anotacao.tif")
            output_file_rgb = os.path.join(output_dir, f"tile_{i}_{j}.tif")

            # Cria um novo arquivo TIFF para o bloco
            driver = gdal.GetDriverByName('GTiff')
            output_ds_rgb = driver.Create(output_file_rgb, x_block_size, y_block_size, datasetRGB.RasterCount, gdal.GDT_Float32)
            output_ds_anotacao = driver.Create(output_file_anotacao, x_block_size, y_block_size, datasetAnotacao.RasterCount, gdal.GDT_Float32)


            max_val = [915, 900, 910]
            # Copia as bandas do bloco para o novo arquivo TIFF
            for band_index in range(1, datasetRGB.RasterCount + 1):
                # Processa a banda RGB
                output_band_rgb = output_ds_rgb.GetRasterBand(band_index)
                output_band_rgb.WriteArray(block_rgb[band_index - 1, :, :])
                
                # Definir valores de mínimo e máximo (ajustar conforme necessário)
                min_val = -100
                output_band_rgb.SetStatistics(min_val, max_val[band_index-1], (min_val + max_val[band_index-1]) / 2, (max_val[band_index-1] - min_val) / 2)

            # Processa a banda de anotação
            output_band_anotacao = output_ds_anotacao.GetRasterBand(1)
            output_band_anotacao.WriteArray(block_anotacao[ :, :])

            # Fecha os arquivos TIFF de saída
            output_ds_rgb.FlushCache()
            output_ds_anotacao.FlushCache()
            output_ds_rgb = None
            output_ds_anotacao = None

    print(f"Arquivo dividido com sucesso em blocos de {tile_size}x{tile_size}.")

# Caminho para os arquivos TIFF e diretório de saída
input_tiff_anotacao = "anotacao/anotacao.tif"
input_tiff_rgb = "RGB.tif"
output_dir = "anotacao/Resultado"
tile_size = 384

# Cria o diretório de saída se ele não existir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Divide o arquivo TIFF
divide_tiff(input_tiff_anotacao, input_tiff_rgb, output_dir, tile_size)
