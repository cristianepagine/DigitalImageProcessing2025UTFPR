# #===============================================================================
# # Exemplo: segmentação de uma imagem em escala de cinza.
# #-------------------------------------------------------------------------------
# # Autor: Cristiane. R Pagine
# # Universidade  Federal do Paraná
# #===============================================================================

import sys
import timeit
import numpy as np
import cv2

# #===============================================================================

INPUT_IMAGE =  'arroz.bmp'

NEGATIVO = False
THRESHOLD = 0.8
ALTURA_MIN = 15
LARGURA_MIN = 15
N_PIXELS_MIN = 100

# #===============================================================================

def binariza(img, threshold):
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.

Valor de retorno: versão binarizada da img_in.'''

    # TODO: escreva o código desta função.
    # Dica/desafio: usando a função np.where, dá para fazer a binarização muito
    # rapidamente, e com apenas uma linha de código! 
    # tentei fazer com uma linha mas só dava erro!!! código abaixo, dai fiz da forma tradicional com dois laços
    #return np.where(img > threshold, 1, 0)  # Se o valor for maior que o threshold, coloca 1, senão coloca 0
    rows, cols, channels = img.shape
    for row in range(rows):
        for col in range(cols):
            if img[row, col] > threshold:
                img[row, col] = 1
            else:
                img[row, col] = 0

    return img

# #-------------------------------------------------------------------------------

def rotula(img, largura_min, altura_min, n_pixels_min):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores [0.1, 0.2, etc].'''
    # Verifica se a imagem tem 2 ou 3 dimensões
    if len(img.shape) == 3:
        rows, cols, _ = img.shape  # Para imagens coloridas (RGB)
    else:
        rows, cols = img.shape  # Para imagens em escala de cinza
        
    # Inicializa a matriz de rótulos com zeros
    rotulos = np.zeros_like(img)
    # Inicializa a variável de rótulo atual
    rotulo_atual = 0
    # Lista para armazenar os componentes encontrados
    componentes = []

    # Função flood_fill recursiva
    def flood_fill(r, c, rotulo):
        '''Função recursiva de flood fill.'''
        # Cria uma pilha para armazenar os pixels a serem visitados
        stack = [(r, c)]
        pixels = []
        while stack:
            x, y = stack.pop()
            if x < 0 or y < 0 or x >= rows or y >= cols or img[x, y] == 0 or rotulos[x, y] != 0:
                continue # Ignora pixels fora dos limites, fundo ou já rotulados
            
            rotulos[x, y] = rotulo
            pixels.append((x, y))
            # Adiciona vizinhos à pilha
            stack.append((x - 1, y))
            stack.append((x + 1, y))
            stack.append((x, y - 1))
            stack.append((x, y + 1))
        return pixels
    
    # Laço para percorrer todos os pixels da imagem
    for i in range(rows):
        for j in range(cols):
            # Novo componente encontrado
            if img[i, j] == 1 and rotulos[i, j] == 0:  
                rotulo_atual += 0.1
                # Executa o flood fill
                pixels = flood_fill(i, j, rotulo_atual)
                n_pixels = len(pixels)
                # Se o componente for grande o suficiente, considera-o
                if n_pixels >= n_pixels_min:
                    # Calculando as coordenadas 
                    T = min(p[0] for p in pixels)
                    B = max(p[0] for p in pixels)
                    L = min(p[1] for p in pixels)
                    R = max(p[1] for p in pixels)
                    # Adiciona o componente à lista
                    componentes.append({
                        'label': rotulo_atual,
                        'n_pixels': n_pixels,
                        'T': T, 'L': L, 'B': B, 'R': R
                    })

    return componentes
# #===============================================================================


def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza (img, THRESHOLD)
    cv2.imshow ('01 - binarizada', img)
    cv2.imwrite ('01 - binarizada.png', img*255)

    start_time = timeit.default_timer ()
    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados com retangulos.
    #for c in componentes:
     #   cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))
     
    # Convertemos a imagem binária para uint8 para usar com findContours
    img_bin_uint8 = (img * 255).astype(np.uint8)

    # Encontrar contornos
    contornos, _ = cv2.findContours(img_bin_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Desenhar os contornos
    cv2.drawContours(img_out, contornos, -1, (0, 0, 1), 1)
    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', img_out*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

# #===============================================================================
