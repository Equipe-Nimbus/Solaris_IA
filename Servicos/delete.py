import os

def delete_downloaded_files(urls):
    for url in urls:
        local_filename = 'Modelo/arquivosProvisorios/' + url.split('/')[-1]  # Nome do arquivo baseado no URL
        if os.path.exists(local_filename):
            os.remove(local_filename)
            print(f"Arquivo deletado: {local_filename}")
        else:
            print(f"Arquivo não encontrado: {local_filename}")