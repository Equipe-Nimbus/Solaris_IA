import cv2
import numpy as np

# Carregar a imagem TIFF
img = cv2.imread('CBERS4A_WPM_PCA_RGB321_20240915_205_151.png')

# Converter para escala de cinza para simplificar
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar thresholding (limiarização)
# Defina um valor de threshold apropriado para sua imagem
_, binary_mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)

# Criar um kernel para a dilatação
# O kernel define o tamanho da área que será dilatada
kernel = np.ones((5, 5), np.uint8)  # Você pode ajustar o tamanho para mais ou menos dilatação

# Aplicar a dilatação
dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)  # Aumentar 'iterations' para mais dilatação

# Salvar a máscara dilatada
cv2.imwrite('nuvens_dilatada_mask_151.png', dilated_mask)

# Exibir o resultado
cv2.imshow('Nuvens dilatadas', dilated_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
