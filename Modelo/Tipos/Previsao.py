class Estatistica:
    def __init__(self, nuvem, sombra, fundo):
        self.nuvem = nuvem
        self.sombra = sombra
        self.fundo = fundo

class Previsao:
    def __init__(self, download_link, png_preview, estatiscas:Estatistica):
        self.download_link = download_link
        self.png_preview = png_preview
        self.estatistica = estatiscas
