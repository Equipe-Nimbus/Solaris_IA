class Estatistica:
    def __init__(self, nuvem, sombra, fundo):
        self.nuvem = nuvem
        self.sombra = sombra
        self.fundo = fundo
    
    def to_dict(self):
        return{
            "nuvem":self.nuvem,
            "sombra":self.sombra,
            "fundo":self.fundo
        }

class Previsao:
    def __init__(self, download_link, png_preview, estatiscas:Estatistica):
        self.download_link = download_link
        self.png_preview = png_preview
        self.estatistica = estatiscas

    def to_dict(self):
        return {
            "download_link":self.download_link,
            "png_preview":self.png_preview,
            "estatistica":{
                "nuvem":self.estatistica.nuvem,
                "sombra":self.estatistica.sombra,
                "fundo":self.estatistica.fundo,
            }
        }
