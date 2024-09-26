import os
import requests


def download_file(urls):
    download_folder = 'Modelo/arquivosProvisorios/'
    os.makedirs(download_folder, exist_ok=True)
    for url in urls:
        local_filename = os.path.join(download_folder, url.split('/')[-1])
        print(local_filename)
        with requests.get(url, stream=True) as response:
            if response.status_code == 200:
                with open(local_filename, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                print(f"Download concluído: {local_filename}")
            else:
                print(f"Erro ao baixar o arquivo: {url}")
