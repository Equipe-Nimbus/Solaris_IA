import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

def download_file(urls):
    download_folder = 'Modelo/arquivosProvisorios/'
    os.makedirs(download_folder, exist_ok=True)

    # Configurar a sessão com retentativas (sem timeout)
    session = requests.Session()
    retries = Retry(
        total=5,  # Número máximo de tentativas
        backoff_factor=1,  # Tempo de espera exponencial (1s, 2s, 4s, etc.)
        status_forcelist=[500, 502, 503, 504]  # Códigos HTTP para justificar nova tentativa
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    for url in urls:
        local_filename = os.path.join(download_folder, url.split('/')[-1])
        print(f"Iniciando download: {local_filename}")
        
        try:
            with session.get(url, stream=True) as response:
                if response.status_code == 200:
                    # Obtém o tamanho total do arquivo (se disponível)
                    total_size = int(response.headers.get('content-length', 0))
                    # Cria a barra de progresso
                    with open(local_filename, 'wb') as file, tqdm(
                        desc=local_filename,
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:  # Filtra chunks vazios
                                file.write(chunk)
                                # Atualiza a barra de progresso
                                bar.update(len(chunk))
                    print(f"Download concluído: {local_filename}")
                else:
                    print(f"Erro ao baixar o arquivo: {url} - Status Code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Erro ao tentar baixar {url}: {e}")
