#Esse código faz troca de fundo (chroma key), tirando o fundo verde das imagens e colocando outro fundo no lugar.

#Usei OpenCV pra ler as imagens e fazer os cálculos. O que eu fiz foi:
    # Ver qual parte da imagem é verde comparando o canal verde com os outros (vermelho e azul).
    # Depois fiz um tipo de máscara pra saber onde é fundo e onde é a pessoa (ou objeto da frente).
    # Apliquei blur pra suavizar as bordas, senão fica uma borda muito marcada.
    # Também usei um truque pra tirar aquele brilho verde que fica nas bordas.
    # No final, juntei a imagem original com o novo fundo usando essa máscara (com alpha blending).
#O código lê as imagens 0.bmp até 9.bmp e troca o fundo delas pelo fundo que tá no arquivo 'dw.jpg'.


import cv2
import numpy as np
import os

def aplicar_chroma_key(foreground, background):
    foreground = foreground.astype(np.float32) / 255.0
    background = background.astype(np.float32) / 255.0

    background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
    b, g, r = cv2.split(foreground)
    green_score = g - np.maximum(r, b)
    green_score = cv2.GaussianBlur(green_score, (7, 7), 0)
    green_score = cv2.normalize(green_score, None, 0.0, 1.0, cv2.NORM_MINMAX)

    alpha = 1.0 - green_score
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    alpha_3ch = cv2.merge([alpha, alpha, alpha])

    result = foreground * alpha_3ch + background * (1.0 - alpha_3ch)

    # Spill suppression (borda verde)
    mask_borda = (alpha < 0.9).astype(np.float32)
    mask_borda = cv2.GaussianBlur(mask_borda, (3, 3), 0)
    result[:, :, 1] = result[:, :, 1] * (1 - 0.2 * mask_borda)

    result = (result * 255).astype(np.uint8)
    return result

# Cria pasta de saída
os.makedirs("saida", exist_ok=True)

# Fundo
fundo = cv2.imread('dw.jpg')
if fundo is None:
    raise FileNotFoundError("Imagem 'dw.jpg' não encontrada.")

# Loop de imagens 0.bmp a 9.bmp
for i in range(10):
    nome_entrada = f'{i}.bmp'
    imagem = cv2.imread(nome_entrada)
    if imagem is None:
        print(f"⚠️ Imagem {nome_entrada} não encontrada. Pulando.")
        continue

    resultado = aplicar_chroma_key(imagem, fundo)

    nome_saida = f'saida/{i}_saida.jpg'
    cv2.imwrite(nome_saida, resultado)
    print(f"✅ Imagem processada e salva: {nome_saida}")

    # Exibe o resultado
    cv2.imshow(f'Resultado {i}', resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
