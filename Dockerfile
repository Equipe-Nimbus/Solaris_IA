FROM continuumio/miniconda3

# Configurar canais Conda
RUN conda config --add channels defaults \
    && conda config --add channels conda-forge \
    && conda config --add channels pytorch \
    && conda config --add channels nvidia \
    && conda config --set channel_priority flexible

# Copiar o arquivo environment.yml para o container
COPY environment.yml /tmp/environment.yml

# Criar e ativar o ambiente Conda a partir do arquivo environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -a -y

# Usar o ambiente criado como padrão
SHELL ["conda", "run", "-n", "solaris_ia", "/bin/bash", "-c"]

# Copiar o código para o diretório raiz do container
COPY . /

# Expor a porta usada pelo Flask
EXPOSE 8080

# Comando para iniciar o serviço
CMD ["conda", "run", "--no-capture-output", "-n", "solaris_ia", "python", "app.py"]
