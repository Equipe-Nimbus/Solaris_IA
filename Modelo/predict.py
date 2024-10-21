import os
import tempfile
import svgwrite
import torch
import logging
import numpy as np
from osgeo import gdal
from tqdm import tqdm
from PIL import Image
from Modelo.unet import UNet
from Servicos.compressSVG import compress_svg
import xml.etree.ElementTree as ET
import base64
#from Servicos.pegarSvg import read_svg_file 
from Servicos.compressPNG import svg_to_png_base64

# Definir o caminho da pasta de arquivos provisórios
PROVISORY_FOLDER = "Modelo/chunks"

# Função para converter SVG comprimido para uma string base64
def convert_svg_to_base64(svg_bytes):
    """
    Converte SVG em bytes para uma string base64 para ser enviada como JSON.
    
    Args:
        svg_bytes (bytes): Dados do SVG comprimido.
    
    Returns:
        str: String base64 do SVG.
    """
    return base64.b64encode(svg_bytes).decode('utf-8')

def get_output_filenames(input):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'
    return list(map(_generate_name, input))

def resize_tiff(tiff_file, new_width, new_height, output_file):
    """
    Redimensiona um arquivo TIFF sem cortar partes da imagem original.
    
    Args:
        tiff_file (str): Caminho para o arquivo TIFF original.
        new_width (int): Largura desejada para o novo TIFF.
        new_height (int): Altura desejada para o novo TIFF.
        output_file (str): Caminho para salvar o TIFF redimensionado.
    """
    try:
        # Abrir o dataset de origem
        src_ds = gdal.Open(tiff_file)
        if src_ds is None:
            logging.error(f"Erro: Não foi possível abrir o arquivo TIFF: {tiff_file}")
            return
        
        # Usar gdal.Warp para redimensionar a imagem preservando a proporção
        gdal.Warp(
            destNameOrDestDS=output_file,
            srcDSOrSrcDSTab=src_ds,  # Fonte de dados é especificada corretamente agora
            width=new_width,
            height=new_height,
            resampleAlg=gdal.GRA_Bilinear,
            dstSRS=src_ds.GetProjection(),  # Manter o sistema de coordenadas de referência
            options=["-overwrite"]  # Permitir sobrescrever se o arquivo existir
        )
        logging.info(f"Imagem redimensionada salva em: {output_file}")
    except Exception as e:
        logging.error(f"Erro ao redimensionar a imagem: {e}")

# Função para gerar o SVG de uma máscara multiclasse com sobreposição leve
def mask_to_svg_multiclass(mask, image_size, overlap=1):
    dwg = svgwrite.Drawing(size=image_size)
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)

    for y in range(height):
        x = 0
        while x < width:
            if mask[y, x] != 0 and not visited[y, x]:
                mask_value = mask[y, x]
                fill_color = 'white' if mask_value == 255 else 'gray'

                x_start = x
                while x < width and mask[y, x] == mask_value and not visited[y, x]:
                    visited[y, x] = True
                    x += 1
                x_end = x

                y_end = y
                while y_end < height and np.all(mask[y_end, x_start:x_end] == mask_value):
                    visited[y_end, x_start:x_end] = True
                    y_end += 1

                # Adicionar um pequeno overlap (sobreposição) para evitar linhas
                dwg.add(dwg.rect(
                    insert=(x_start - overlap, y - overlap), 
                    size=(x_end - x_start + overlap * 2, y_end - y + overlap * 2), 
                    fill=fill_color, 
                    stroke='none'
                ))

            else:
                x += 1

    return dwg

# Função para salvar um chunk de imagem como PNG
def save_chunk_image(chunk, x_chunk, y_chunk, chunk_index):
    os.makedirs(PROVISORY_FOLDER, exist_ok=True)
    chunk_image_path = os.path.join(PROVISORY_FOLDER, f"chunk_{chunk_index}.png")
    
    # Converter o chunk para uma imagem PIL e salvar como PNG
    chunk_image = np.moveaxis(chunk, 0, -1)  # Mover os canais para o final (de [C, H, W] para [H, W, C])
    
    # Verificar se o chunk contém apenas valores baixos (muito escuros)
    logging.info(f"Valores do chunk {chunk_index}: Min={np.min(chunk_image)}, Max={np.max(chunk_image)}")
    
    chunk_image = Image.fromarray(chunk_image.astype(np.uint8))
    chunk_image.save(chunk_image_path)

    return chunk_image_path

# Função para adicionar um chunk ao SVG final
def append_chunk_to_svg(svg, chunk_path, x_offset, y_offset):
    svg_chunk = svgwrite.Drawing(filename=chunk_path)

    # Adicionar o conteúdo do chunk ao SVG final, ajustando as posições
    for element in svg_chunk.elements:
        if isinstance(element, svgwrite.shapes.Rect):
            element.attribs['x'] = str(float(element.attribs['x']) + x_offset)
            element.attribs['y'] = str(float(element.attribs['y']) + y_offset)
        svg.add(element)

# Função para fazer a predição de um chunk
def predict_chunk(net, chunk, device):
    img = torch.from_numpy(chunk).unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img).cpu()
        output = torch.nn.functional.interpolate(output, (chunk.shape[1], chunk.shape[2]), mode='bilinear')
        mask = output.argmax(dim=1).squeeze(0).numpy()

        mask_image = np.zeros_like(mask, dtype=np.uint8)
        mask_image[mask == 2] = 255  # Nuvens
        mask_image[mask == 1] = 127  # Sombras
        return mask_image

# Função para processar e salvar um chunk
def process_and_save_chunk(net, chunk, x_offset, y_offset, device, output_dir, chunk_index):
    logging.info(f"Processing chunk {chunk_index}, x_offset: {x_offset}, y_offset: {y_offset}")
    
    # Verificar se o chunk contém apenas valores zero
    if np.min(chunk) == 0 and np.max(chunk) == 0:
        logging.info(f"Chunk {chunk_index} contém apenas zeros. Criando SVG vazio.")
        svg_filename = create_empty_svg(chunk.shape[1], chunk.shape[2], output_dir, chunk_index)
        return (svg_filename, x_offset, y_offset)

    # Salvar o chunk original como imagem PNG
    chunk_image_path = save_chunk_image(chunk, chunk.shape[1], chunk.shape[2], chunk_index)
    logging.info(f"Chunk image saved at {chunk_image_path}")

    # Gerar a máscara
    mask_chunk = predict_chunk(net, chunk, device)

    if np.sum(mask_chunk) == 0:
        logging.warning(f"Chunk {chunk_index} contém apenas zeros (máscara vazia)")

    # Salvar o SVG correspondente ao chunk
    svg_filename = save_chunk_svg(mask_chunk, chunk.shape[1], chunk.shape[2], output_dir, chunk_index)
    return (svg_filename, x_offset, y_offset)

# Função para criar um SVG vazio para chunks que não contêm dados válidos
def create_empty_svg(x_chunk, y_chunk, output_dir, chunk_index):
    svg_chunk = svgwrite.Drawing(size=(x_chunk, y_chunk))
    
    os.makedirs(PROVISORY_FOLDER, exist_ok=True)
    svg_filename = os.path.join(PROVISORY_FOLDER, f"chunk_{chunk_index}_empty.svg")
    
    svg_chunk.saveas(svg_filename)
    return svg_filename

# Função para salvar cada chunk como SVG
def save_chunk_svg(mask_chunk, x_chunk, y_chunk, output_dir, chunk_index):
    svg_chunk = svgwrite.Drawing(size=(x_chunk, y_chunk))
    mask_svg = mask_to_svg_multiclass(mask_chunk, (x_chunk, y_chunk))
    
    os.makedirs(PROVISORY_FOLDER, exist_ok=True)
    svg_filename = os.path.join(PROVISORY_FOLDER, f"chunk_{chunk_index}.svg")
    
    mask_svg.saveas(svg_filename)
    return svg_filename

# Processamento sequencial de chunks (sem paralelismo)
def process_large_tiff_and_save_chunks(tiff_file, model, chunk_size=(1024, 1024), device='cpu', resize_factor=0.5):
    # Certifique-se de que o dataset está sendo aberto corretamente
    dataset = gdal.Open(tiff_file)
    if dataset is None:
        logging.error(f"Erro: Não foi possível abrir o arquivo TIFF: {tiff_file}. Verifique as permissões e o caminho do arquivo.")
        return [], (0, 0)

    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize

    # Redimensionar a imagem antes de processar
    new_width = int(x_size * resize_factor)
    new_height = int(y_size * resize_factor)

    # Criar um arquivo TIFF temporário para o redimensionamento
    resized_tiff = os.path.join(PROVISORY_FOLDER, "resized_image.tif")
    resize_tiff(tiff_file, new_width, new_height, resized_tiff)

    # Reabrir o TIFF redimensionado para continuar o processamento
    dataset = gdal.Open(resized_tiff)
    if dataset is None:
        logging.error(f"Erro: Não foi possível abrir o arquivo TIFF redimensionado: {resized_tiff}")
        return [], (0, 0)

    x_size = dataset.RasterXSize
    y_size = dataset.RasterYSize

    net = UNet(n_channels=3, n_classes=3, bilinear=False)
    net.to(device=device)
    state_dict = torch.load(model, map_location=device)
    net.load_state_dict(state_dict['model_state_dict'])

    with tempfile.TemporaryDirectory() as temp_dir:
        chunk_paths = []
        chunk_index = 0

        for x_offset in tqdm(range(0, x_size, chunk_size[0]), desc="Processing Chunks"):
            for y_offset in range(0, y_size, chunk_size[1]):
                x_chunk = min(chunk_size[0], x_size - x_offset)
                y_chunk = min(chunk_size[1], y_size - y_offset)

                chunk = dataset.ReadAsArray(x_offset, y_offset, x_chunk, y_chunk)

                # Verificar se os valores do chunk são significativos
                logging.info(f"Valores do chunk {chunk_index}: Min={np.min(chunk)}, Max={np.max(chunk)}")

                chunk_paths.append(process_and_save_chunk(net, chunk, x_offset, y_offset, device, temp_dir, chunk_index))
                chunk_index += 1

        return chunk_paths, (new_width, new_height)

# Função final para combinar os chunks e gerar um único SVG
def merge_chunks_to_svg(chunk_paths, x_size, y_size):
    svg_final = svgwrite.Drawing(size=(x_size, y_size))

    # Combinar todos os SVGs individuais
    for chunk_path, x_offset, y_offset in tqdm(chunk_paths, desc="Merging Chunks"):
        # Carregar o conteúdo do SVG chunk como XML
        try:
            tree = ET.parse(chunk_path)
            root = tree.getroot()

            # Iterar sobre todos os elementos <rect> no SVG chunk
            for element in root.findall('.//{http://www.w3.org/2000/svg}rect'):
                # Extrair atributos e ajustar as coordenadas
                x = float(element.get('x', 0)) + x_offset
                y = float(element.get('y', 0)) + y_offset
                width = float(element.get('width'))
                height = float(element.get('height'))
                fill_color = element.get('fill')

                # Adicionar o retângulo ajustado ao SVG final
                svg_final.add(svg_final.rect(
                    insert=(x, y),
                    size=(width, height),
                    fill=fill_color,
                    stroke='none'
                ))
        except ET.ParseError as e:
            logging.error(f"Erro ao parsear o SVG chunk {chunk_path}: {e}")

    return svg_final.tostring()

# Função para criar a máscara e salvar o SVG final
def run_predict(model, input, output, no_save, mask_threshold, refactor_size, bilinear, classes, avaliacao):

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if isinstance(input, str):
        input = [input]

    if os.path.isdir(input[0]):
        filenames = os.listdir(input[0])
        in_files = []
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                in_files.append(os.path.join(input[0], filename))
        input = in_files
    else:
        in_files = input

    out_files = get_output_filenames(input)

    net = UNet(n_channels=3, n_classes=classes, bilinear=bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict['model_state_dict'])

    logging.info('Model loaded!')
    pngsList = []
    download_links = []  # Para armazenar os links de download

    for i, filename in enumerate(in_files):
        if "\\" in filename:
            filename = filename.replace("\\", "/")
        logging.info(f'Predicting image {filename} ...')
        svg = filename.split('/')[0]+ "/preview/" + filename.split('/')[2].split('.')[0] + "_OUT_multiclass_mask.svg"
        print(svg)
        #if(not os.path.exists(svg)):
        chunk_paths, new_size = process_large_tiff_and_save_chunks(
            tiff_file=filename,
            model=model,
            chunk_size=(1024, 1024),
            device=device,
            resize_factor=refactor_size
        )

        svg_final = merge_chunks_to_svg(chunk_paths, x_size=new_size[0], y_size=new_size[1])
        #else:
        #    svg_final = read_svg_file(svg)
            
        """ # Ajuste para usar dimensões menores e gerar SVG comprimido
        compressed_svg = compress_svg(svg_final, width=747, height=768)
        svg_base64 = convert_svg_to_base64(compressed_svg)
        svgsList.append(svg_base64) """

        # Converta o SVG final para PNG e salve
        png_data = svg_to_png_base64(svg_final, width=747, height=768)


        if not no_save:
            os.makedirs(output, exist_ok=True)
            print(output)
            print(out_files)
            svg_filename = os.path.join(output, f"{out_files[i].replace('.png', f'.svg')}".split('/')[-1].split("\\")[-1])
            with open(svg_filename, 'w', encoding='utf-8') as f:
                f.write(svg_final)
            logging.info(f'SVG saved to {svg_filename}')
            

            # Adicionar o link de download
            download_link = f"http://localhost:8080/download/{os.path.basename(svg_filename)}"
            download_links.append(download_link)

            # Salvar o PNG convertido
            png_filename = os.path.join(output, f"{out_files[i]}".split('/')[-1].split("\\")[-1])
            with open(png_filename, 'wb') as f:
                f.write(png_data)  # Decodifique a string Base64 para bytes e salve como PNG
            logging.info(f'PNG saved to {png_filename}')
            
            # Adicionar o link de download para o PNG
            png_download_link = f"http://localhost:8080/view/{os.path.basename(png_filename)}"
            pngsList.append(png_download_link)

    return {"pngs": pngsList, "download_links": download_links}
