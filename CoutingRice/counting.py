import sys
import cv2
import numpy as np
import math
import statistics

# Configurações
INPUT_IMAGE = '82.bmp'
ALTURA_MIN = 10
LARGURA_MIN = 10
N_PIXELS_MIN = 100

def flood_fill(img, x, y, label, info):
    stack = [(x, y)]
    info.update({'T': y, 'B': y, 'L': x, 'R': x, 'n_pixels': 0})
    
    while stack:
        x, y = stack.pop()
        if img[y, x] != -1:
            continue

        img[y, x] = label
        info['n_pixels'] += 1
        info['T'] = min(info['T'], y)
        info['B'] = max(info['B'], y)
        info['L'] = min(info['L'], x)
        info['R'] = max(info['R'], x)

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < img.shape[1] and 0 <= ny < img.shape[0] and img[ny, nx] == -1:
                stack.append((nx, ny))
    
    return info

def rotula(img, largura_min, altura_min, n_pixels_min):
    img_out = np.where(img > 0, -1, 0).astype(np.int32)
    label = 1
    componentes = []

    for y in range(img_out.shape[0]):
        for x in range(img_out.shape[1]):
            if img_out[y, x] == -1:
                info = flood_fill(img_out, x, y, label, {})
                w, h = info['R'] - info['L'], info['B'] - info['T']
                if info['n_pixels'] >= n_pixels_min and w >= largura_min and h >= altura_min:
                    info['label'] = label
                    componentes.append(info)
                label += 1
    return componentes

def preprocessar_imagem(caminho):
    img = cv2.imread(caminho)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(norm, (11, 11), 0)
    binarizada = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 75, -7)
    return img, binarizada

def contar_arroz(componentes):
    tamanhos = [c['n_pixels'] for c in componentes]
    mediana = int(statistics.median(tamanhos))
    total = 0

    for pixels in tamanhos:
        total += math.ceil(pixels / mediana) if (pixels / mediana) % 1 > 0.5 else int(pixels / mediana)
    
    return total, mediana, sorted(tamanhos)

def main():
    sys.setrecursionlimit(100000)
    imagem_original, imagem_segmentada = preprocessar_imagem(INPUT_IMAGE)
    componentes = rotula(imagem_segmentada, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    quantidade, mediana, tamanhos_ordenados = contar_arroz(componentes)

    print("Quantidade estimada de grãos de arroz:", quantidade)
    print("Mediana dos tamanhos:", mediana)
    print("Tamanhos ordenados:", tamanhos_ordenados)

    bordas = cv2.Canny(imagem_segmentada, 100, 145)
    cv2.imshow("Original", imagem_original)
    cv2.imshow("Segmentada", imagem_segmentada)
    cv2.imshow("Bordas", bordas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
