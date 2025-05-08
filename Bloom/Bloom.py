import cv2
import numpy as np

# Caminho para a imagem de entrada
PATH = 'b01 - Original.bmp'
# Limiar de brilho para o filtro bright-pass
THR = 200
# Valores de sigma para o filtro Gaussiano
SIGMA = [5, 20, 50]
# Tamanhos de kernel para o filtro Box Blur
KERNEL = [5, 15, 35]

# Função para extrair as regiões mais brilhantes da imagem
def brilho(img, lim):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = hsv[:, :, 2] > lim
    return img * np.expand_dims(mask, 2)

# Função para aplicar o efeito bloom usando filtro Gaussiano
def gauss_bloom(img, lim, sigmas):
    bright = brilho(img, lim).astype(np.float32)
    bl = np.zeros_like(bright)
    for s in sigmas:
        bl += cv2.GaussianBlur(bright, (0, 0), sigmaX=s, sigmaY=s)
    return cv2.addWeighted(img.astype(np.float32), 1, bl, 1, 0).astype(np.uint8)

# Função para aplicar o efeito bloom usando filtro Box Blur
def box_bloom(img, lim, kernels):
    bright = brilho(img, lim).astype(np.float32)
    bl = np.zeros_like(bright)
    temp = bright
    for k in kernels:
        temp = cv2.blur(temp, (k, k))
        bl += temp
    return cv2.addWeighted(img.astype(np.float32), 1, bl, 1, 0).astype(np.uint8)

if __name__ == '__main__':
   
    orig = cv2.imread(PATH)
    cv2.imshow("Gauss Bloom", gauss_bloom(orig.copy(), THR, SIGMA))
    cv2.imshow("Box Bloom", box_bloom(orig.copy(), THR, KERNEL))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
