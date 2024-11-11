from osgeo import gdal

def is_geotiff(tiff_file):
    # Abre o arquivo TIFF
    dataset = gdal.Open(tiff_file)
    
    if dataset is None:
        print(f"Erro: Não foi possível abrir o arquivo {tiff_file}")
        return False

    # Verifica se o arquivo possui uma transformação affine
    transform = dataset.GetGeoTransform()
    if transform == (0, 1, 0, 0, 0, 1):
        print("Não é um GeoTIFF: Não contém transformação affine válida.")
        return False
    else:
        print("Transformação affine encontrada:", transform)

    # Verifica se o arquivo possui um sistema de coordenadas de referência (CRS)
    projection = dataset.GetProjection()
    if not projection:
        print("Não é um GeoTIFF: Não contém sistema de coordenadas de referência (CRS).")
        return False
    else:
        print("Sistema de coordenadas de referência (CRS) encontrado:", projection)

    return True

# Exemplo de uso
tiff_file = 'Modelo\\arquivosProvisorios\CBERS4A_WPM_PCA_RGB321_20240924_197_123.tif'
if is_geotiff(tiff_file):
    print(f"{tiff_file} é um GeoTIFF.")
else:
    print(f"{tiff_file} não é um GeoTIFF.")