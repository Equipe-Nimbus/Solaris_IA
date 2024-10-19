import cairosvg
import base64
import logging
import io

def svg_to_png_base64(svg_content, width=747, height=768):
    """
    Converte um conteúdo SVG para PNG e retorna em formato Base64.
    
    Args:
        svg_content (str): String contendo o SVG.
        width (int): Largura do PNG resultante.
        height (int): Altura do PNG resultante.
    
    Returns:
        str: PNG em formato Base64, ou None em caso de erro.
    """
    try:
        # Verificar se o conteúdo do SVG é realmente válido
        if not svg_content.strip().startswith("<svg"):
            logging.error("O conteúdo fornecido não é um SVG válido. Verifique o conteúdo de entrada.")
            return None

        # Usar cairosvg para converter o SVG para PNG
        png_bytes = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), output_width=width, output_height=height)

        # Converter para Base64
        png_base64 = base64.b64encode(png_bytes).decode('utf-8')
        return png_base64

    except Exception as e:
        logging.error(f"Erro ao converter SVG para PNG Base64: {e}")
        return None
