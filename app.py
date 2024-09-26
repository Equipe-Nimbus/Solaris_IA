from Modelo.predict import run_predict
from Servicos.delete import delete_downloaded_files
from Servicos.download import download_file
from flask import Flask, request, jsonify

#Thumbnail 


app = Flask(__name__)

@app.route('/geraMascaraThumbnail', methods=['POST'])
def create_item():
    links = request.json["links"]
    download_file(links)
    svgs = run_predict("Modelo/checkpoints/checkpoint_epoch40.pth", "Modelo/arquivosProvisorios", "preview/", True, 0.5, (747, 768), False, 2)
    delete_downloaded_files(links)
    return jsonify({"svgs":svgs}), 200

if __name__ == '__main__':
    app.run(debug=True, port=8080)  # Especificando a porta 8080



""" links = [
    "https://data.inpe.br/bdc/data/CB4A-WPM-PCA-FUSED/v001/197/123/2024/9/CBERS4A_WPM_PCA_RGB321_20240924_197_123.png",
    "https://data.inpe.br/bdc/data/CB4A-WPM-PCA-FUSED/v001/197/139/2024/9/CBERS4A_WPM_PCA_RGB321_20240924_197_139.png",
    "https://data.inpe.br/bdc/data/CB4A-WPM-PCA-FUSED/v001/197/142/2024/9/CBERS4A_WPM_PCA_RGB321_20240924_197_142.png",
    "https://data.inpe.br/bdc/data/CB4A-WPM-PCA-FUSED/v001/228/108/2024/9/CBERS4A_WPM_PCA_RGB321_20240924_228_108.png",
    "https://data.inpe.br/bdc/data/CB4A-WPM-PCA-FUSED/v001/228/114/2024/9/CBERS4A_WPM_PCA_RGB321_20240924_228_114.png"
] """

