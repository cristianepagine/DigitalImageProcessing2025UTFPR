# Cristiane Regina Pagine

import sys
import timeit
import numpy as np
import math
import cv2

# Caminho da imagem de entrada
CAMINHO_IMAGEM = 'a01 - Original.bmp'

# Tamanho da janela de filtragem (ímpar)
ALTURA_JANELA = 11
LARGURA_JANELA = 15

# Filtro de média usando abordagem ingênua
def filtro_simples(img):
    altura, largura, canais = img.shape
    resultado = img.copy()

    ajuste_linha = 0 if ALTURA_JANELA % 2 == 0 else 1
    ajuste_coluna = 0 if LARGURA_JANELA % 2 == 0 else 1

    for canal in range(canais):
        for linha in range(ALTURA_JANELA // 2, altura - (ALTURA_JANELA // 2)):
            for coluna in range(LARGURA_JANELA // 2, largura - (LARGURA_JANELA // 2)):
                soma = 0.0
                for dy in range(-(ALTURA_JANELA // 2), ALTURA_JANELA // 2):
                    for dx in range(-(LARGURA_JANELA // 2), LARGURA_JANELA // 2):
                        soma += img[linha + dy, coluna + dx, canal]
                media = soma / ((ALTURA_JANELA - ajuste_linha) * (LARGURA_JANELA - ajuste_coluna))
                resultado[linha, coluna, canal] = media

    return resultado

# Filtro simples com somas otimizadas por linha
def filtro_com_soma_otimizada(img):
    altura, largura, canais = img.shape
    saida = img.copy()

    metade_col = LARGURA_JANELA // 2
    metade_lin = ALTURA_JANELA // 2

    for canal in range(canais):
        for linha in range(metade_lin, altura - metade_lin):
            soma = 0.0
            for coluna in range(metade_col, largura - metade_col):
                if coluna == metade_col:
                    for dy in range(-metade_lin, metade_lin):
                        for dx in range(-metade_col, metade_col):
                            soma += img[linha + dy, coluna + dx, canal]
                else:
                    for k in range(-metade_lin, metade_lin):
                        soma += img[linha + k, coluna + metade_col, canal]
                        soma -= img[linha + k, coluna - metade_col, canal]

                media = soma / ((ALTURA_JANELA - 1) * (LARGURA_JANELA - 1))
                saida[linha, coluna, canal] = media

    return saida

# Primeira etapa do filtro separável (horizontal)
def filtro_horizontal(img, destino):
    altura, largura, canais = img.shape
    meio_coluna = LARGURA_JANELA // 2
    ajuste = 0 if LARGURA_JANELA % 2 == 0 else 1

    for canal in range(canais):
        for linha in range(meio_coluna, altura - meio_coluna):
            for coluna in range(meio_coluna, largura - meio_coluna):
                acumulador = 0.0
                for dx in range(-meio_coluna, meio_coluna):
                    acumulador += img[linha, coluna + dx, canal]
                destino[linha, coluna, canal] = acumulador / (LARGURA_JANELA - ajuste)

    return destino

# Segunda etapa do filtro separável (vertical)
def filtro_vertical(img, destino):
    altura, largura, canais = img.shape
    meio_linha = ALTURA_JANELA // 2
    ajuste = 0 if ALTURA_JANELA % 2 == 0 else 1

    for canal in range(canais):
        for coluna in range(meio_linha, largura - meio_linha):
            for linha in range(meio_linha, altura - meio_linha):
                acumulador = 0.0
                for dy in range(-meio_linha, meio_linha):
                    acumulador += img[linha + dy, coluna, canal]
                destino[linha, coluna, canal] = acumulador / (ALTURA_JANELA - ajuste)

    return destino

# Aplica o filtro separável (horizontal + vertical)
def filtro_separavel(img):
    imagem_horizontada = img.copy()
    imagem_horizontada = filtro_horizontal(img, imagem_horizontada)

    imagem_final = imagem_horizontada.copy()
    imagem_final = filtro_vertical(imagem_horizontada, imagem_final)

    return imagem_final

# Gera imagem integral (soma acumulada)
def gerar_imagem_integral(img):
    altura, largura, canais = img.shape
    for canal in range(canais):
        for linha in range(altura):
            acumulado = 0
            for coluna in range(largura):
                acumulado += img[linha, coluna, canal]
                img[linha, coluna, canal] = acumulado
                if linha > 0:
                    img[linha, coluna, canal] += img[linha - 1, coluna, canal]
    return img

# Filtro usando imagem integral
def filtro_integral(img):
    acumulada = gerar_imagem_integral(img.copy())
    resultado = img.copy()
    altura, largura, canais = img.shape

    for canal in range(canais):
        for linha in range(altura):
            for coluna in range(largura):
                top = min(linha, ALTURA_JANELA // 2)
                bottom = min(altura - linha - 1, ALTURA_JANELA // 2)
                left = min(coluna, LARGURA_JANELA // 2)
                right = min(largura - coluna - 1, LARGURA_JANELA // 2)

                y1 = linha - top
                x1 = coluna - left
                y2 = linha + bottom
                x2 = coluna + right

                A = acumulada[y1, x1, canal]
                B = acumulada[y1, x2, canal]
                C = acumulada[y2, x1, canal]
                D = acumulada[y2, x2, canal]

                area = (top + bottom) * (left + right)
                media = (D + A - B - C) / area if area != 0 else 0
                resultado[linha, coluna, canal] = media

    return resultado

# Função principal
def principal():
    img = cv2.imread(CAMINHO_IMAGEM, cv2.IMREAD_COLOR)
    if img is None:
        print("Erro ao carregar a imagem.")
        sys.exit()

    img = img.astype(np.float32) / 255.0
    img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))

    cv2.imshow("00 - Original", img)

    inicio = timeit.default_timer()
    resultado1 = filtro_simples(img)
    print("Tempo (ingênuo): %.6f" % (timeit.default_timer() - inicio))
    cv2.imshow("a02 - Filtro Ingênuo", resultado1)
    cv2.imwrite("a02 - filtro_inge.png", resultado1 * 255)

    inicio = timeit.default_timer()
    resultado2 = filtro_separavel(img)
    print("Tempo (separável): %.6f" % (timeit.default_timer() - inicio))
    cv2.imshow("a03 - Filtro Separável", resultado2)
    cv2.imwrite("a03 - filtro_sep.png", resultado2 * 255)

    inicio = timeit.default_timer()
    resultado3 = filtro_integral(img)
    print("Tempo (integral): %.6f" % (timeit.default_timer() - inicio))
    cv2.imshow("a04 - Filtro Integral", resultado3)
    cv2.imwrite("a04 - filtro_int.png", resultado3 * 255)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    principal()
