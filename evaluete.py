import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision import transforms
from tqdm import tqdm
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from Modelo.predict import run_predict
from Modelo.predict_without_chunks import predict_all_from_folder
from Modelo.unet.unet_model import UNet


# Função para carregar as máscaras (ajustada para 1 canal)
def load_mask(mask_path, image_size):
    local_filename = mask_path.split('/')[-1].split('\\')[-1]
    mascara = os.path.join(mask_path.split('\\')[0], local_filename)
    mask = Image.open(mascara).convert('L')  # Convertendo para escala de cinza (1 canal)
    mask = mask.resize(image_size)
    mask = np.array(mask)
    mask = torch.from_numpy(mask).long()  # Máscara binária com valores inteiros
    return mask

# Função de avaliação para calcular as métricas
def evaluate_and_compare(image_paths, mask_paths, predict_paths, device, image_size=(747, 768)):

    # Ordenar imagens e máscaras para garantir correspondência
    image_paths = sorted(image_paths)
    mask_paths = sorted(mask_paths)
    predict_paths = sorted(predict_paths)

    # Listas para armazenar os resultados
    y_true = []
    y_pred = []


    # Iterar pelas imagens e máscaras
    for img_path, mask_path, predict_path in tqdm(zip(image_paths[:50], mask_paths[:50], predict_paths[:50]), total=50, desc="Avaliando"):

        img = Image.open(img_path)
        mask_true = load_mask(mask_path, image_size)
        pred_mask = load_mask(predict_path, image_size)
        # Pré-processar a imagem
        img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

        # Permutar as dimensões para (H, W, C) antes de exibir
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        """ # Exibir a imagem original que está sendo analisada
        plt.imshow(img_np)
        plt.title(f'Analisando: {os.path.basename(img_path)}\nMáscara: {os.path.basename(mask_path)}')
        plt.axis('off')
        plt.show()



        # Exibir a imagem, máscara real e máscara prevista
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))

        ax[0].imshow(img_np)  # Imagem original
        ax[0].set_title('Imagem Original')
        ax[0].axis('off')

        ax[1].imshow(mask_true.cpu().numpy(), cmap='gray')  # Máscara real
        ax[1].set_title('Máscara Real')
        ax[1].axis('off')

        ax[2].imshow(pred_mask, cmap='gray')  # Máscara prevista
        ax[2].set_title('Máscara Prevista')
        ax[2].axis('off')

        plt.show() """

        # Binarizar ambas as máscaras para comparação correta
        mask_true_np = (mask_true.cpu().numpy() > 0).astype(np.uint8)  # Máscara real
        mask_pred_resized = (np.resize(pred_mask, mask_true_np.shape) > 0).astype(np.uint8)  # Máscara prevista

        # Achatar as máscaras para comparação de pixel a pixel
        y_true.extend(mask_true_np.flatten().tolist())
        y_pred.extend(mask_pred_resized.flatten().tolist())

    # Calcular as métricas de avaliação
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary')
    rec = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    conf_matrix = confusion_matrix(y_true, y_pred)

    logging.info(f'Acurácia: {acc}')
    logging.info(f'Precisão: {prec}')
    logging.info(f'Recall: {rec}')
    logging.info(f'F1-Score: {f1}')
    logging.info(f'Matriz de Confusão: \n{conf_matrix}')

    return acc, prec, rec, f1, conf_matrix

# Função para plotar a matriz de confusão como uma imagem
def plot_confusion_matrix(conf_matrix):
    conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum() * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Oranges', cbar=True,
                xticklabels=['Negativo (FP)', 'Positivo (VP)'],
                yticklabels=['Negativo (FN)', 'Positivo (VN)'],
                vmin=0.01, vmax=conf_matrix_percentage.max())  # Ajuste do contraste da matriz
    plt.xlabel('Predição')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.show()


# Função principal
def main(model_path, image_dir, mask_dir):
    # Carregar o modelo
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    predict_dir = "avaliacao/previsao"
    there_is_not_pred = len([os.path.join(image_dir, f) for f in os.listdir(predict_dir) if f.endswith('.png')][:50]) == 0
    # Listar as imagens e as máscaras (carregar até 130)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if(there_is_not_pred):
        net = UNet(n_channels=3, n_classes=3, bilinear=False)
       
        net.to(device=device)
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict['model_state_dict'])
        predict_all_from_folder(net, "avaliacao/imgs", "avaliacao/previsao", device)
    

    predict_paths = [os.path.join(predict_dir, f) for f in os.listdir(predict_dir) if f.endswith('.png')][:50]
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')][:50]
    mask_paths = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')][:50]

    print("Teste pred path: " + predict_paths[0])
    # Verificar se a quantidade de imagens e máscaras bate
    assert len(image_paths) == len(mask_paths) and len(image_paths) == len(predict_paths) and len(mask_paths) == len(predict_paths), (
    f"A quantidade de imagens ({len(image_paths)}), máscaras ({len(mask_paths)}) e previsões ({len(predict_paths)}) não corresponde."
    )
    # Avaliar a rede e capturar os resultados
    acc, prec, rec, f1, conf_matrix = evaluate_and_compare(image_paths, mask_paths, predict_paths, device)

    # Exibir os resultados
    print(f'Acurácia: {acc:.4f}')
    print(f'Precisão: {prec:.4f}')
    print(f'Recall: {rec:.4f}')
    print(f'F1-Score: {f1:.4f}')

    # Plotar a matriz de confusão
    plot_confusion_matrix(conf_matrix)

# Chamada da função principal
if __name__ == '__main__':
    model_path = './Modelo/checkpoints/checkpoint_epoch31.pth'  # Substitua pelo caminho do seu modelo
    image_dir = './avaliacao/imgs'  # Substitua pelo caminho das suas imagens
    mask_dir = './avaliacao/mask'  # Substitua pelo caminho das suas máscaras

    main(model_path, image_dir, mask_dir)