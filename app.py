from Modelo.predict import run_predict
from Servicos.delete import delete_downloaded_files
from Servicos.download import download_file
from flask import Flask, request, jsonify, send_from_directory

#Thumbnail 
OUTPUT_FOLDER = "preview"


app = Flask(__name__)
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route('/geraMascaraThumbnail', methods=['POST'])
def create_item():
    links = request.json["links"]
    #download_file(links)
    result = run_predict(
        model="Modelo/checkpoints/checkpoint_epoch31.pth", 
        input="Modelo/arquivosProvisorios", 
        output=OUTPUT_FOLDER, 
        no_save=False, 
        mask_threshold=0.5, 
        refactor_size=0.1, 
        bilinear=False, 
        classes=3, 
        avaliacao=False
    )
    #delete_downloaded_files(links)
    # Retornar os links de download
    return jsonify(result), 200

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    print(filename)
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

@app.route('/view/<filename>', methods=['GET'])
def view_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=8080, use_reloader=False)  # Especificando a porta 8080



""" links = [
    "https://data.inpe.br/bdc/data/CB4A-WPM-PCA-FUSED/v001/197/123/2024/9/CBERS4A_WPM_PCA_RGB321_20240924_197_123.png",
    "https://data.inpe.br/bdc/data/CB4A-WPM-PCA-FUSED/v001/197/139/2024/9/CBERS4A_WPM_PCA_RGB321_20240924_197_139.png",
    "https://data.inpe.br/bdc/data/CB4A-WPM-PCA-FUSED/v001/197/142/2024/9/CBERS4A_WPM_PCA_RGB321_20240924_197_142.png",
    "https://data.inpe.br/bdc/data/CB4A-WPM-PCA-FUSED/v001/228/108/2024/9/CBERS4A_WPM_PCA_RGB321_20240924_228_108.png",
    "https://data.inpe.br/bdc/data/CB4A-WPM-PCA-FUSED/v001/228/114/2024/9/CBERS4A_WPM_PCA_RGB321_20240924_228_114.png"
] """

