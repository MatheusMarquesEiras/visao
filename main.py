import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def segmentar_kmeans(caminho_imagem="input.jpg", k=4, nome_saida="output.jpg"):
    imagem = cv2.imread(str(caminho_imagem))
    if imagem is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {caminho_imagem}")

    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    pixel_values = np.float32(imagem_rgb.reshape((-1, 3)))
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centros = cv2.kmeans(pixel_values, k, None, criterios, 10, cv2.KMEANS_RANDOM_CENTERS)
    centros = np.uint8(centros)
    imagem_segmentada = centros[labels.flatten()].reshape(imagem_rgb.shape)

    imagem_segmentada_bgr = cv2.cvtColor(imagem_segmentada, cv2.COLOR_RGB2BGR)
    cv2.imwrite(nome_saida, imagem_segmentada_bgr)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(imagem_rgb)
    axs[0].set_title("Imagem Original")
    axs[0].axis("off")
    axs[1].imshow(imagem_segmentada)
    axs[1].set_title(f"Segmentação K-means (k={k})")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    caminho_input = input("Digite o caminho da imagem (Enter para usar 'input.jpg'): ")
    caminho = Path(caminho_input) if caminho_input else Path("input.jpg")
    k_input = input("Digite o número de clusters K (Enter para usar 4): ")
    k = int(k_input) if k_input else 4
    segmentar_kmeans(caminho_imagem=caminho, k=k)
