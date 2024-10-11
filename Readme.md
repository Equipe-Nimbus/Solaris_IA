# Como usar

### Intalar Miniconda

https://www.anaconda.com/download/success

### Instale o pythorch

conda install pytorch torchvision torchaudio cpuonly -c pytorch

### Criar um ambiente anaconda e instalar dependencias

conda env create -f ambiente.yml

### Baixar um checkout do modelo

https://drive.google.com/drive/folders/1-ePNBLe-q229CPzmYFw9rMh6ZzFmx09R

Coloco o caminho do ckeckpoint na função run_predict do app.py

### Rodar a aplicação

python app.py