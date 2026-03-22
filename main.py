"""
Script para processamento de imagens de dados e contagem de pips (os pontinhos que denotam o valor daquela face)
"""
from typing import Tuple
from itertools import product
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import json

# Isola os pips (brancos) do resto da imagem para facilitar a detecção
def preprocessar(img_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Função de pré-processamento de imagem.

    Cada imagem possui um contexto diferente de iluminação,
    ângulo, posição, etc
    Neste caso, a melhor opção é de utilizar dos pips, que são brancos
    para segmentação, retirando todos resto da imagem

    Args:
        img_path: string contendo path até a imagem a ser pré-processada
    Returns:
        Tupla contendo a imagem original e imagem processada
    """
    # carregar imagem
    img = cv2.imread(img_path)

    # BGR -> GRAYSCALE
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # THRESHOLDING (todos funcionaram bem exceto OTSU)
    _, img_binaria = cv2.threshold(img_gray, 250, 255, cv2.THRESH_TOZERO)

    # Passar um blur inicial de mediana
    # medianblur remove ruído e mantém as bordas
    img_filtrada = cv2.medianBlur(img_binaria, 7)

    # checar elipses, os pips
    elipses = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # erodir para remover ruídos
    erodir = cv2.erode(img_filtrada, elipses, iterations=2)

    # expandir pips para facilitar contagem
    dilatar = cv2.dilate(erodir, elipses, iterations=3)
    return img, dilatar

# Conta círculos na imagem via transformada de Hough e retorna o total
def contar(proc: int,
        dp: int,
        minDist: int,
        param1: int,
        param2: int,
        minRadius: int,
        maxRadius: int) -> int:
    """
    Função para contagem de pips

    Utiliza-se do método HoughCircles, que pode ser representado matematicamente
    por (x - x_centro)^2 + (y - y_centro)^2 = r^2, onde x_centro e y_centro
    referem-se ao centro do círculo. A biblioteca OpenCV utiliza o método do
    Gradiente de Hough (21HT).

    Args:
        proc:       imagem pré-processada
        dp:         razão inversa da resolução do acumulador em relação à
                    resolução da imagem. dp=1 mantém a mesma resolução;
                    dp=2 reduz o acumulador à metade — valores maiores
                    aceleram o processamento mas diminuem a precisão
        minDist:    distância mínima (em pixels) entre os centros de dois
                    círculos detectados. Valor baixo pode gerar detecções
                    duplicadas; valor alto pode fundir círculos distintos
        param1:     limiar superior do detector de bordas Canny interno
                    (o limiar inferior é automaticamente param1 / 2).
                    Valores altos detectam apenas bordas mais fortes
        param2:     limiar de votos no acumulador para um candidato ser
                    considerado um círculo. Valores menores detectam mais
                    círculos, porém com mais falsos positivos
        minRadius:  raio mínimo (em pixels) dos círculos a detectar.
                    Use 0 para não aplicar limite inferior
        maxRadius:  raio máximo (em pixels) dos círculos a detectar.
                    Use 0 para não aplicar limite superior
    Returns:
        Número inteiro com a quantidade de círculos (pips) detectados
    """
    pips = cv2.HoughCircles(proc, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                            param1=param1, param2=param2,
                            minRadius=minRadius, maxRadius=maxRadius)
    return len(pips[0]) if pips is not None else 0

# Testa todas as combinações de hiperparâmetros e salva a melhor em resultados.txt
def treinar_modelo():
    # Dados organizados em uma pasta -> aqui vão todas as imagens
    pasta_dados = os.path.join(os.getcwd(), "dados")

    # Colocando quanto foi contado de fato, isso é nosso ground truth para otimizar nosso "modelo" de contagem de pips
    contagem_real = {"img1.jpg": 11, "img2.jpg": 4, "img3.jpg": 15, "img4.jpg": 7}

    # Pré-processar todas as imagens presentes na pasta
    preprocessed_imgs = {
        name: preprocessar(os.path.join(pasta_dados, name))
        for name in contagem_real
    }

    grid = {
        "dp":        [1], # Imagem pequena
        "minDist":   [3, 5, 8, 12], # Distancias entre pontos
        "param1":    [5, 10, 15],
        "param2":    list(range(4, 25)), # Mais problematico de ser otimizado
        "minRadius": [1, 3, 5], # Raio minimo para ser contado
        "maxRadius": [20, 30, 40, 50], # Raio maximo para ser contado
    }

    keys = list(grid.keys()) # Nomes de hiperparametros
    best_params, best_score = None, -1

    # Combinatoria de todos os possiveis parametros
    for combo in product(*grid.values()):
        params = dict(zip(keys, combo))
        # Quantos corretos do total?
        score = sum(
            contar(proc, **params) == contagem_real[name]
            for name, (_, proc) in preprocessed_imgs.items()
        )
        # Se melhor que o anterior, substituir
        if score > best_score:
            best_score, best_params = score, params
        if best_score == len(contagem_real):
            break

    # Printar no terminal
    print(f"Melhor escore: {best_score}/{len(contagem_real)}")
    print(f"Melhor combinação: {best_params}")
    for name, (_, proc) in preprocessed_imgs.items():
        n = contar(proc, **best_params)
        gt = contagem_real[name]
        status = "Correto" if n == gt else "Errado"
        print(f"  {status} {name}: detectado={n}, esperado={gt}")
    # Salvar parâmetros
    with open("resultados.txt", "w") as file:
        json.dump(best_params, file, indent = 4)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Plottar imagem contendo pips detectados
    for ax, (name, (orig, proc)) in zip(axes, preprocessed_imgs.items()):
        display = cv2.cvtColor(orig.copy(), cv2.COLOR_BGR2RGB)
        pips = cv2.HoughCircles(proc, cv2.HOUGH_GRADIENT, **best_params)
        if pips is not None:
            for x, y, r in np.round(pips[0]).astype("int"):
                cv2.circle(display, (x, y), r, (255, 0, 0), 2)
                cv2.circle(display, (x, y), 2, (0, 255, 0), 3)
        n = len(pips[0]) if pips is not None else 0
        gt = contagem_real[name]
        ax.imshow(display)
        ax.set_title(f"{name}  Detectado={n}  Esperado={gt}", color="green" if n == gt else "red")
        ax.axis("off")
    plt.savefig("output.png", dpi = 300)
    plt.tight_layout()
    plt.show()

# Pré-processa e conta os pips de uma única imagem usando os parâmetros salvos
def analisar_imagem(nome_imagem: str):
    pasta_dados = os.path.join(os.getcwd(), "dados")
    img_path = os.path.join(pasta_dados, nome_imagem)

    if not os.path.exists(img_path):
        print(f"Imagem '{nome_imagem}' não encontrada em {pasta_dados}")
        return

    if not os.path.exists("resultados.txt"):
        print("Arquivo resultados.txt não encontrado. Execute 'uv run main.py treinar' primeiro.")
        return

    with open("resultados.txt", "r") as f:
        params = json.load(f)

    orig, proc = preprocessar(img_path)

    pips = cv2.HoughCircles(proc, cv2.HOUGH_GRADIENT, **params)
    n = len(pips[0]) if pips is not None else 0
    print(f"{nome_imagem}: {n} pip(s) detectado(s)")

    display = cv2.cvtColor(orig.copy(), cv2.COLOR_BGR2RGB)
    if pips is not None:
        for x, y, r in np.round(pips[0]).astype("int"):
            cv2.circle(display, (x, y), r, (255, 0, 0), 2)
            cv2.circle(display, (x, y), 2, (0, 255, 0), 3)

    plt.figure()
    plt.imshow(display)
    plt.title(f"{nome_imagem}  Detectado={n}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Contagem de pips em dados")
    subparsers = parser.add_subparsers(dest="comando", required=True)

    subparsers.add_parser("treinar", help="Treinar o modelo e salvar os parâmetros em resultados.txt")

    analisar_parser = subparsers.add_parser("analisar", help="Analisar uma imagem usando os parâmetros salvos")
    analisar_parser.add_argument("--nome_imagem", required=True, help="Nome do arquivo de imagem dentro da pasta dados")

    args = parser.parse_args()

    if args.comando == "treinar":
        treinar_modelo()
    elif args.comando == "analisar":
        analisar_imagem(args.nome_imagem)
