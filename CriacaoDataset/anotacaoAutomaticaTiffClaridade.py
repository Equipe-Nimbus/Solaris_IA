import os
import shutil
from PIL import Image
import numpy as np
from skimage.filters import threshold_otsu
from skimage import morphology, measure
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from refinamentoManualSelecao import interactive_mask_editor



# Abrir a imagem TIFF
imageName="CBERS4A_WPM_PCA_RGB321_20241020_229_107_0_44032"
image_path = f'CriacaoDataset\DataSet\imgs\{imageName}.tif'
#image_path = f'DataSet\\feito\{imageName}.tif'
destino_pasta = f'CriacaoDataset\DataSet\\feito\{imageName}.tif'
try:
    img = Image.open(image_path)
except:
    img = Image.open(destino_pasta)
# Converter a imagem para escala de cinza

gray_img = img.convert('L')

# Converter a imagem em um array NumPy
gray_np = np.array(gray_img)

# Aplicar threshold Otsu para detectar nuvens (partes claras da imagem)
#thresh_value_nuvem = threshold_otsu(gray_np)
thresh_value_nuvem = 170
nuvem_mask = (gray_np > thresh_value_nuvem).astype(np.uint8)



# Aplicar threshold Otsu inverso para detectar sombras (partes escuras)
#thresh_value_sombra = threshold_otsu(gray_np)
thresh_value_sombra = 50
sombra_mask = (gray_np < thresh_value_sombra).astype(np.uint8)


# Aplicar dilatação para ajustar as áreas de nuvens e sombras
kernel_size = 10
kernel_sombra = 10
kernel = morphology.disk(kernel_size)
kernel_sombra = morphology.disk(kernel_sombra)

# Remover pequenos objetos das máscaras
nuvem_mask_refined = morphology.remove_small_objects(nuvem_mask.astype(bool), min_size=300)
sombra_mask_refined = morphology.remove_small_objects(sombra_mask.astype(bool), min_size=300)



nuvem_mask_dilated = morphology.dilation(nuvem_mask_refined, kernel)
sombra_mask_dilated = morphology.dilation(sombra_mask_refined, kernel_sombra)



# Criar a máscara 3multiclasse (0: fundo, 1: nuvem, 2: sombra)
multiclasse_mask = np.zeros_like(gray_np)

# Sombra tem valor 2
multiclasse_mask[sombra_mask_dilated == 1] = 1

# Nuvem tem valor 1
multiclasse_mask[nuvem_mask_dilated == 1] = 2



# Identificar o background (valor 0)
background_mask = (multiclasse_mask == 0).astype(np.uint8)


# Identificar o background (valor 0)
background_mask = (multiclasse_mask == 0).astype(np.uint8)

# Rotular os componentes conectados no background
labels = measure.label(background_mask, connectivity=1)

# Definir o tamanho mínimo e máximo para os componentes conectados de background
min_size_background = 5000  # Tamanho mínimo
max_size_background = 1000  # Tamanho máximo, que você pode ajustar conforme necessário

# Filtrar componentes conectados no background
for region in measure.regionprops(labels):
    if min_size_background > region.area:
        # Se o trecho de background ultrapassar o limiar de tamanho mínimo, e for menor que o máximo,
        # então modificar a máscara para transformar o background em nuvem (valor 2)
        if region.area < max_size_background:
            multiclasse_mask[labels == region.label] = 1
        else:
            multiclasse_mask[labels == region.label] = 2  # Valor 2 representa nuvem



# Defina o raio desejado para a seleção
selecao_raio = 10  # Ajuste o raio conforme necessário

# Exibir a imagem original e a máscara multiclasse para edição manual com múltiplas seleções e raio aumentado
interactive_mask_editor(img, multiclasse_mask, radius=selecao_raio)

multiclasse_mask_scaled = (multiclasse_mask * 127).astype(np.uint8)
# Converter para imagem PIL e salvar a máscara multiclasse
multiclasse_img = Image.fromarray(multiclasse_mask_scaled)
multiclasse_img.save(f'CriacaoDataset/DataSet/anotacao/{imageName}.png')

try:
    shutil.move(image_path, destino_pasta)
except:
    shutil.move(destino_pasta, destino_pasta)

# Plotar a imagem original e a máscara lado a lado
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Exibir a imagem original
axes[0].imshow(gray_np, cmap='gray')
axes[0].set_title('Imagem Original')
axes[0].axis('off')

# Exibir a máscara multiclasse
axes[1].imshow(multiclasse_mask, cmap='gray')
axes[1].set_title('Máscara Multiclasse (Nuvem: 1, Sombra: 2)')
axes[1].axis('off')

plt.show()