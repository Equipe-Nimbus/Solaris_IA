from osgeo import gdal
import numpy as np
from scipy.ndimage import binary_dilation

# Função para calcular o NDVI, NDWI e gerar a máscara de nuvens com dilatação
def calculate_ndvi_ndwi_and_mask_with_dilation(input_red_tiff, input_nir_tiff, input_green_tiff, output_mask_tiff, ndvi_threshold, ndwi_threshold, dilation_iterations=1):
    # Abrir os arquivos TIFF
    datasetRED = gdal.Open(input_red_tiff)
    datasetNIR = gdal.Open(input_nir_tiff)
    datasetGREEN = gdal.Open(input_green_tiff)
    
    # Ler as bandas RED, NIR e GREEN
    red_band = datasetRED.GetRasterBand(1).ReadAsArray().astype('float32')
    nir_band = datasetNIR.GetRasterBand(1).ReadAsArray().astype('float32')
    green_band = datasetGREEN.GetRasterBand(1).ReadAsArray().astype('float32')

    # Evitar divisão por zero
    np.seterr(divide='ignore', invalid='ignore')

    # Cálculo do NDVI
    ndvi = (nir_band - red_band) / (nir_band + red_band)

    # Cálculo do NDWI (usando a banda GREEN e NIR)
    ndwi = (green_band - nir_band) / (green_band + nir_band)
    
    # Criar a máscara binária
    # 0: background (água ou terra), 1: nuvem
    mask = np.zeros_like(ndvi)

    # Condição para background (água ou terra)
    background_condition = (ndvi < ndvi_threshold) | (ndwi < ndwi_threshold)

    # Condição para nuvens (não background)
    cloud_condition = (ndvi >= ndvi_threshold) | (ndwi >= ndwi_threshold)
    
    # Definir background como 0 e nuvem como 1
    mask[background_condition] = 0  # Água ou solo
    mask[cloud_condition] = 1       # Nuvem

    # Aplicar a dilatação para expandir as áreas de nuvem
    mask_dilated = binary_dilation(mask, iterations=dilation_iterations)

    # Obter as informações de transformação e projeção do dataset original
    transform = datasetRED.GetGeoTransform()
    projection = datasetRED.GetProjection()

    # Criar o arquivo de saída para a máscara
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_mask_tiff, datasetRED.RasterXSize, datasetRED.RasterYSize, 1, gdal.GDT_Byte)
    out_dataset.SetGeoTransform(transform)
    out_dataset.SetProjection(projection)
    
    # Escrever a máscara no arquivo de saída
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(mask_dilated)
    out_band.SetNoDataValue(0)

    # Fechar os datasets
    out_dataset = None
    datasetRED = None
    datasetNIR = None
    datasetGREEN = None

# Definir os parâmetros
input_red_tiff = "../CBERS_4A_WPM_20240912_218_107_L2_BAND3.tif"  # Banda vermelha
input_nir_tiff = "../CBERS_4A_WPM_20240912_218_107_L2_BAND4.tif"  # Banda NIR
input_green_tiff = "../CBERS_4A_WPM_20240912_218_107_L2_BAND2.tif"  # Banda verde
output_mask_tiff = 'anotacao/anotacao.tif'  # Caminho para o arquivo TIFF de saída

ndvi_threshold = 0.2   # Limiar para separar nuvens do background
ndwi_threshold = 0   # Limiar para separar corpos d'água do background
dilation_iterations = 2  # Número de iterações de dilatação (expande a área da nuvem)

# Executar a função
calculate_ndvi_ndwi_and_mask_with_dilation(input_red_tiff, input_nir_tiff, input_green_tiff, output_mask_tiff, ndvi_threshold, ndwi_threshold, dilation_iterations)
