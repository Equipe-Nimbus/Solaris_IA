import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import time

def download_file(urls):
    download_folder = 'Modelo/arquivosProvisorios/'
    os.makedirs(download_folder, exist_ok=True)

    # Configurar a sessão com retentativas e timeout
    session = requests.Session()
    retries = Retry(
        total=10,  # Número máximo de tentativas
        backoff_factor=2,  # Tempo de espera exponencial (1s, 2s, 4s, etc.)
        status_forcelist=[500, 502, 503, 504]  # Códigos HTTP para justificar nova tentativa
    )
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    CHUNK_SIZE = 4096
    RESTART_THRESHOLD = 1080000000  # Reiniciar após 1.01 GB (em bytes)

    for url in urls:
        local_filename = os.path.join(download_folder, url.split('/')[-1])
        print(f"Iniciando download: {local_filename}")

        # Tamanho do arquivo baixado até agora
        downloaded_size = os.path.getsize(local_filename) if os.path.exists(local_filename) else 0

        try:
            while downloaded_size < int(requests.head(url).headers.get('content-length', 0)):
                resume_header = {'Range': f'bytes={downloaded_size}-'}
                with session.get(url, headers=resume_header, stream=True, timeout=(10, 60)) as response:
                    response.raise_for_status()  # Levanta uma exceção para erros HTTP
                    total_size = int(response.headers.get('content-length', 0)) + downloaded_size

                    # Barra de progresso
                    with open(local_filename, 'ab') as file, tqdm(
                        desc=local_filename,
                        initial=downloaded_size,
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        downloaded_since_restart = 0
                        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk:
                                file.write(chunk)
                                bar.update(len(chunk))
                                downloaded_size += len(chunk)
                                downloaded_since_restart += len(chunk)

                                # Verificar se o limite de 1.01 GB foi atingido desde o último reinício
                                if downloaded_since_restart >= RESTART_THRESHOLD:
                                    print("\nReiniciando conexão para continuar o download...")
                                    time.sleep(2)  # Pausa para evitar sobrecarregar o servidor
                                    break

            print(f"Download concluído: {local_filename}")
        except requests.exceptions.RequestException as e:
            print(f"Erro ao tentar baixar {url}: {e}")
