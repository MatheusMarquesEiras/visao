import cv2
import numpy as np

def segmentar_kmeans(caminho_imagem, k=3, nome_saida="imagem_segmentada.jpg"):
    """
    Segmenta uma imagem usando K-means baseado em cores e salva o resultado.

    Parâmetros:
        caminho_imagem (str): Caminho da imagem a ser segmentada.
        k (int): Número de clusters/grupos.
        nome_saida (str): Nome do arquivo de saída.
    """
    # 1. Carregar imagem
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem: {caminho_imagem}")

    # 2. Converter para RGB (opcional)
    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

    # 3. Transformar em array de pixels
    pixel_values = imagem_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # 4. Definir critérios de parada
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # 5. Aplicar K-means
    _, labels, centros = cv2.kmeans(pixel_values, k, None, criterios, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 6. Reconstruir imagem segmentada
    centros = np.uint8(centros)
    imagem_segmentada = centros[labels.flatten()]
    imagem_segmentada = imagem_segmentada.reshape(imagem_rgb.shape)

    # 7. Converter de volta para BGR antes de salvar
    imagem_segmentada_bgr = cv2.cvtColor(imagem_segmentada, cv2.COLOR_RGB2BGR)

    # 8. Salvar resultado
    cv2.imwrite(nome_saida, imagem_segmentada_bgr)
    print(f"Imagem segmentada salva como: {nome_saida}")


# Exemplo de uso
if __name__ == "__main__":
    segmentar_kmeans("rune.jpg", k=4, nome_saida="rune1.jpg")
