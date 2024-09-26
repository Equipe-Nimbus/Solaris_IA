import gzip
import base64

def compress_svg(svg_string):
    compressed_data = gzip.compress(svg_string.encode('utf-8'))
    return base64.b64encode(compressed_data).decode('utf-8')


def decompress_svg(compressed_string):
    compressed_data = base64.b64decode(compressed_string.encode('utf-8'))
    return gzip.decompress(compressed_data).decode('utf-8')