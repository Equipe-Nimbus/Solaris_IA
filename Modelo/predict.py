import argparse
import logging
import os
import svgwrite
import cairosvg
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as ET
from Modelo.utils.data_loading import BasicDataset
from Modelo.unet import UNet
from Modelo.utils.utils import plot_img_and_mask
from Servicos.compressSVG import compress_svg, decompress_svg

def get_segment_crop(img, mask, cl=[0]):
    img[~np.isin(mask, cl)] = 0
    return img

def mask_to_svg(mask, image_size):
    dwg = svgwrite.Drawing(size=image_size)
    
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)

    for y in range(height):
        x = 0
        while x < width:
            if mask[y, x] and not visited[y, x]:
                # Encontrar o fim do bloco horizontal de pixels contíguos
                x_start = x
                while x < width and mask[y, x] and not visited[y, x]:
                    visited[y, x] = True
                    x += 1
                x_end = x

                # Agora encontramos um bloco horizontal de x_start a x_end na linha y
                # Tentar expandir para linhas abaixo
                y_end = y
                while y_end < height and np.all(mask[y_end, x_start:x_end]):
                    visited[y_end, x_start:x_end] = True
                    y_end += 1
                
                # Adicionar um único retângulo cobrindo todo o bloco
                dwg.add(dwg.rect(insert=(x_start, y), size=(x_end - x_start, y_end - y), fill='black'))
            else:
                x += 1

    return dwg.tostring()


def predict_img(net,
                full_img,
                device,
                image_size,
                out_threshold):
    net.eval()
    if full_img.mode != 'RGB':
        full_img = full_img.convert('RGB')
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, image_size, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    

    with torch.no_grad():
        output = net(img).cpu()
        output = torch.nn.functional.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        
        if net.n_classes > 1:
            # Segmentação multi-classe
            mask = output.argmax(dim=1)
            
            # Para cada classe, gera o SVG correspondente
            for cl, mask_class in enumerate(output[0]):
                mask_reshaped = mask.numpy().reshape((mask.shape[1], mask.shape[2]))
                
                # Aplicar a sigmoide e limiar
                mask_class = torch.sigmoid(mask_class) > out_threshold
                mask_class_np = mask_class.numpy().astype(bool)

                # Gera o SVG para a classe
                if(cl == 1):
                    svg_mask = mask_to_svg(mask_class_np, full_img.size)
                

            return svg_mask

        else:
            # Segmentação binária
            mask = torch.sigmoid(output) > out_threshold

            # Converte a máscara para numpy e gera SVG
            mask_numpy = mask[0].long().squeeze().numpy()
            svg_mask = mask_to_svg(mask_numpy, full_img.size)
            return svg_mask




def get_output_filenames(input):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return list(map(_generate_name, input))



#checkpoint, pasta onde está os inputs, salvar ou não salvar(False ou True), limear das mascaras (0.5), tamanho da imagem, bilineares (False), quantidades de classes(int)
def run_predict(model, input, output, no_save, mask_threshold, image_size, bilinear, classes):

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Se `input` for uma string, transforme-o em uma lista
    if isinstance(input, str):
        input = [input]

    # Verifique se o primeiro item da lista é um diretório
    if os.path.isdir(input[0]):
        filenames = os.listdir(input[0])
        in_files = []
        for filename in filenames:
            # Verifica se é um arquivo de imagem válido
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                in_files.append(os.path.join(input[0], filename))  # Corrige o caminho
        input = in_files
    else:
        in_files = input

    out_files = get_output_filenames(input)

    net = UNet(n_channels=3, n_classes=classes, bilinear=bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model model')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict['model_state_dict'])

    logging.info('Model loaded!')

    svgsList = []

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           image_size=image_size,
                           out_threshold=mask_threshold,
                           device=device)
        
        svgsList.append(compress_svg(mask))
        
        if not no_save:
            os.makedirs(output, exist_ok=True)
            # Gera nome de saída para a imagem
            out_filename = os.path.join(output, f"{out_files[i].replace('.png', f'_nuvem.png')}".split('/')[-1].split("\\")[-1])   # Gera nome de saída
            print(out_filename)
            cairosvg.svg2png(bytestring=mask.encode('utf-8'), write_to=out_filename)
            logging.info(f'Mask saved to {out_filename}')
            # Se quiser salvar também o SVG, faça assim:
            svg_filename = os.path.join(output,f"{out_files[i].replace('.png', f'_nuvem.svg')}".split('/')[-1].split("\\")[-1])  # Gera nome de saída para SVG
            # Parseia o SVG string para um objeto ElementTree
            root = ET.fromstring(mask)

            # Agora você pode manipular o SVG como um XML
            for element in root:
                print(element.tag, element.attrib)

            # Se quiser salvar ou gerar novamente o SVG
            tree = ET.ElementTree(root)
            tree.write(svg_filename)
            logging.info(f'SVG saved to {svg_filename}')
        

        

    return svgsList

        
