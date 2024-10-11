import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector, Button
from matplotlib.path import Path
from skimage import morphology
from PIL import Image

def interactive_mask_editor(original_img, multiclasse_mask, radius=5):
    """
    Função para permitir a edição interativa da máscara multiclasse com modos de retoque para sombras, nuvens e background.

    Args:
        original_img (numpy array): A imagem original colorida para exibição como referência.
        multiclasse_mask (numpy array): A máscara multiclasse que será editada.
        radius (int): O raio da seleção para o LassoSelector.
    """
    # Estado local das variáveis
    retoque_mode = 'sombra'  # Iniciar com sombra como modo de retoque
    mask_history = []  # Para armazenar o histórico de alterações
    current_selection = np.zeros_like(multiclasse_mask, dtype=bool)  # Inicializar uma máscara de seleção vazia

    # Função para mudar para o modo de retoque de sombra
    def set_mode_sombra(event):
        nonlocal retoque_mode
        retoque_mode = 'sombra'
        ax2.set_title('Máscara Multiclasse (Retoque: Sombra)')
        fig.canvas.draw_idle()

    # Função para mudar para o modo de retoque de nuvem
    def set_mode_nuvem(event):
        nonlocal retoque_mode
        retoque_mode = 'nuvem'
        ax2.set_title('Máscara Multiclasse (Retoque: Nuvem)')
        fig.canvas.draw_idle()

    # Função para mudar para o modo de retoque de background
    def set_mode_background(event):
        nonlocal retoque_mode
        retoque_mode = 'background'
        ax2.set_title('Máscara Multiclasse (Retoque: Background)')
        fig.canvas.draw_idle()

    # Função para aplicar as mudanças na máscara com base no modo atual
    def on_select(verts):
        nonlocal multiclasse_mask, current_selection

        # Salvar o estado atual da máscara antes de modificar (para poder desfazer)
        mask_history.append(multiclasse_mask.copy())

        path = Path(verts)
        # Criar uma grade com as coordenadas de todos os pixels
        x, y = np.meshgrid(np.arange(multiclasse_mask.shape[1]), np.arange(multiclasse_mask.shape[0]))
        coords = np.vstack((x.flatten(), y.flatten())).T
        # Verificar quais coordenadas estão dentro da área selecionada
        mask = path.contains_points(coords).reshape(multiclasse_mask.shape)

        # Aplicar uma dilatação na área selecionada para simular um raio maior
        if radius > 1:
            selem = morphology.disk(radius)
            mask = morphology.dilation(mask, selem)

        # Atualizar a seleção atual para o tipo de máscara correspondente ao modo
        current_selection = mask

        # Modificar apenas a nova seleção, mantendo as áreas já ajustadas
        if retoque_mode == 'sombra':
            multiclasse_mask[current_selection] = 1  # Retoque de sombra (valor 1)
        elif retoque_mode == 'nuvem':
            multiclasse_mask[current_selection] = 2  # Retoque de nuvem (valor 2)
        elif retoque_mode == 'background':
            multiclasse_mask[current_selection] = 0  # Retoque de background (valor 0)

        update_mask(img_mask)

    # Função para desfazer a última alteração
    def undo_last_change(event):
        nonlocal multiclasse_mask
        if mask_history:
            multiclasse_mask = mask_history.pop()  # Restaurar o estado anterior da máscara
            update_mask(img_mask)

    # Função para atualizar a máscara após a seleção
    def update_mask(img_mask):
        img_mask.set_data(multiclasse_mask)
        fig.canvas.draw_idle()

    # Inicializar a interface
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))

    # Exibir imagem original à esquerda (com cores originais)
    ax1.imshow(original_img)
    ax1.set_title('Imagem Original (Referência)')
    ax1.axis('off')

    # Exibir máscara à direita
    img_mask = ax2.imshow(multiclasse_mask, cmap='gray')
    ax2.set_title('Máscara Multiclasse (Retoque: Sombra)')
    ax2.axis('off')

    # Permitir a seleção de áreas na máscara
    lasso = LassoSelector(ax2, on_select)

    # Adicionar botão para mudar para o modo Sombra
    ax_button_sombra = plt.axes([0.7, 0.05, 0.1, 0.075])
    button_sombra = Button(ax_button_sombra, 'Modo Sombra')
    button_sombra.on_clicked(set_mode_sombra)

    # Adicionar botão para mudar para o modo Nuvem
    ax_button_nuvem = plt.axes([0.8, 0.05, 0.1, 0.075])
    button_nuvem = Button(ax_button_nuvem, 'Modo Nuvem')
    button_nuvem.on_clicked(set_mode_nuvem)

    # Adicionar botão para mudar para o modo Background
    ax_button_background = plt.axes([0.6, 0.05, 0.1, 0.075])
    button_background = Button(ax_button_background, 'Modo Background')
    button_background.on_clicked(set_mode_background)

    # Adicionar botão para desfazer a última mudança
    ax_button_undo = plt.axes([0.9, 0.05, 0.1, 0.075])
    button_undo = Button(ax_button_undo, 'Desfazer')
    button_undo.on_clicked(undo_last_change)

    plt.show()

# Função para salvar a máscara após a edição
def save_edited_mask(mask, save_path):
    mask_img = Image.fromarray((mask * 127).astype(np.uint8))
    mask_img.save(save_path)
