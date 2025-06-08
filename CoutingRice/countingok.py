
# Primeiro tentei fazer a contagem dos grãos de arroz usando flood fill e rotulagem manual.
# A ideia era identificar os componentes conectados na imagem binarizada e contar com base na mediana do tamanho.
# Usei critérios como número mínimo de pixels, largura e altura mínima, mas o resultado não ficou bom.
# Acho que foi por causa de ruídos ou da forma como o binarizado estava sendo feito.
# Depois testei de outro jeito, usando findContors e operações morfológicas (erode/dilate) para limpar a imagem.
# Também usei o adaptiveThreshold com outros parâmetros, fui testando diferentes valores até conseguir um bom resultado.
# A abordagem que funcionou melhor foi usando funções do OpenCV para tratar e segmentar a imagem.
# Primeiro converti a imagem para escala de cinza com cvtColor, depois usei adaptiveThreshold para binarizar.
# Esse threshold adaptativo ajuda porque ele considera a iluminação local de cada região da imagem.# 
# Depois apliquei operações morfológicas: erode para remover ruídos pequenos e dilate para reforçar os objetos principais.
# Isso ajudou a separar melhor os grãos que estavam grudados.
# Para contar, usei o findContours para encontrar os contornos dos objetos.
# Em seguida, calculei a área de cada contorno e usei a mediana como base para estimar a quantidade de grãos(conforme a dica do professor na ultima aula),
# já que alguns contornos podem representar mais de um grão (quando estão grudados).
# Essa combinação deu resultados bem mais precisos. Ficou um codigo simples, curto, porem funciona
import sys
import statistics
import math
import numpy as np
import cv2


IMAGENS = ['60.bmp', '82.bmp', '114.bmp', '150.bmp', '205.bmp']

#algoritmo para processar as imagens
def process(img, nome_imagem):
    copia = img.copy()
    #transforma a imagem em escala de cinza 
    copia = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    #kernel de 3x3 utilizado para operações morfológicas
    kernel = np.ones((3, 3), np.uint8)
    
    #aplica o adaptiveThreshold para binarizar a imagem utilizando o método gaussiano
    thresh = cv2.adaptiveThreshold(copia, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101, -27) #treshold adaptativo
    
    #faz operações morfológicas para remover ruídos
    #primeiro erode para remover pequenos ruídos, depois dilate para reforçar os objetos principais
    img_erode = cv2.erode(thresh, kernel, iterations=2) 
    img_dilated = cv2.dilate(img_erode, kernel,iterations=1) 
    
    #chama a função de contagem, passando a imagem sem ruídos e a original
    contagem(img_dilated,img, nome_imagem) 
    
# A função contagem recebe a imagem binarizada e a imagem original para desenhar os contornos
# e calcular a quantidade de grãos de arroz. 
def contagem(opening, img, nome_imagem):
    # encontra os contornos na imagem binarizada
    arroz, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # desenha os contornos encontrados na imagem original
    cv2.drawContours(img, arroz, -1, (255, 0, 0), 2)
    # calcula a área de cada contorno e usa a mediana para estimar a quantidade de grãos conforme o professor deu a dica em aula.
    area_m = [cv2.contourArea(contador) for contador in arroz]
    mediana = statistics.median(area_m)
    count = 0
    for i in range(len(arroz)):
        area = cv2.contourArea(arroz[i])
        if area > mediana:
            count += round(area / mediana)
        else:
            count += 1
    #Abaixo está comentado o cv2.imshow, caso queira ver a imagem com os contornos desenhados que foram utilizados para a contagem.
    # cv2.imshow("Detectados", img)
    print(f'Imagem {nome_imagem} : Contagem deu {count}')
    
    
def main():
    for nome_imagem in IMAGENS:
        img = cv2.imread(nome_imagem, cv2.IMREAD_COLOR)
        if img is None:
            print(f'Erro abrindo a imagem {nome_imagem}.\n')
            continue
        process(img, nome_imagem)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

