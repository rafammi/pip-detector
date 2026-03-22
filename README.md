# Contagem de Pips em Dados

Detecção e contagem automática de pips (pontos) em faces de dados usando visão computacional.
O pipeline combina pré-processamento morfológico com Transformação Circular de Hough,
otimizando os hiperparâmetros via grid search exaustiva.

---

## Estrutura do Projeto

```
.
├── dados/           # Imagens de entrada (img1.jpg … img4.jpg)
├── main.py          # Script principal (CLI)
├── main.ipynb       # Notebook de exploração e prototipagem
├── resultados.txt   # Hiperparâmetros ótimos (gerado pelo treino)
├── output.png       # Visualização das detecções
└── pyproject.toml   # Dependências (gerenciadas com uv)
```

---

## Pipeline

```
Imagem RGB
    │
    ▼
Conversão para Escala de Cinza
    │
    ▼
Limiarização (Threshold)
    │
    ▼
Filtro de Mediana
    │
    ▼
Erosão Morfológica
    │
    ▼
Dilatação Morfológica
    │
    ▼
Hough  →  Contagem de pips
```

---

## Fundamentos Matemáticos

### 1. Limiarização `THRESH_TOZERO`

Isola os pips (brancos) suprimindo pixels escuros:

$$
f(x, y) =
\begin{cases}
I(x,y) & \text{se } I(x,y) > T \\
0       & \text{caso contrário}
\end{cases}
\quad T = 250
$$

onde $I(x,y)$ é a intensidade do pixel na posição $(x,y)$.
Pips têm reflexo próximo a 255; a face do dado e o fundo ficam próximos de zero e são eliminados.

---

### 2. Filtro de Mediana

Para uma vizinhança $\mathcal{N}(x,y)$ de tamanho $k \times k$:

$$
\hat{I}(x,y) = \text{mediana}\bigl\{I(p,q) \mid (p,q) \in \mathcal{N}(x,y)\bigr\}
$$

Usado com $k = 7$. Remove ruído sal-e-pimenta preservando bordas — ideal para pips
com bordas nítidas.

---

### 3. Morfologia Matemática

O elemento estruturante utilizado é uma **elipse** de $3 \times 3$ pixels,
adequada à forma circular dos pips.

**Erosão** (remove ruídos menores que o elemento estruturante $B$):

$$
(I \ominus B)(x,y) = \min_{(u,v) \in B}\, I(x+u,\; y+v)
$$

**Dilatação** (expande as regiões restantes, tornando os pips mais proeminentes):

$$
(I \oplus B)(x,y) = \max_{(u,v) \in B}\, I(x-u,\; y-v)
$$

A sequência *erosão (×2) → dilatação (×3)* é uma **abertura** seguida de
crescimento controlado, limpando espúrios e realçando os pips para a etapa seguinte.

---

### 4. Transformação Circular de Hough (21HT)

Cada pip é aproximado por um círculo. A equação implícita de um círculo é:

$$
(x - x_c)^2 + (y - y_c)^2 = r^2
$$

O OpenCV implementa o **Gradiente de Hough em Duas Etapas (21HT)**:

1. **Detecção de bordas (Canny):** localiza pixels de borda via gradiente
   $\nabla I$ com limiar superior `param1` (limiar inferior = `param1 / 2`).

2. **Votação no espaço acumulador:** cada pixel de borda $(x, y)$ com gradiente
   $\nabla I(x,y)$ vota em candidatos a centro $(x_c, y_c)$ ao longo da direção
   do gradiente:

$$
(x_c, y_c) = (x, y) + t \cdot \hat{\nabla}I(x,y), \quad t \in [r_{\min},\, r_{\max}]
$$

3. **Seleção de picos:** centros com votos $\geq$ `param2` são aceitos como
   círculos. O raio é estimado como a mediana das distâncias entre o centro
   aceito e os pixels de borda que votaram nele.

O parâmetro `dp` controla a resolução do acumulador:

$$
\text{resolução do acumulador} = \frac{\text{resolução da imagem}}{\texttt{dp}}
$$

`dp = 1` mantém resolução plena; valores maiores trocam precisão por velocidade.

---

### 5. Busca em Grade (*Grid Search*)

Os hiperparâmetros do `HoughCircles` são otimizados exaustivamente sobre o
conjunto de treinamento. A função objetivo é a **acurácia por imagem**:

$$
\text{score}(\theta) = \sum_{i=1}^{N} \mathbf{1}\bigl[\hat{c}_i(\theta) = c_i\bigr]
$$

onde $\hat{c}_i(\theta)$ é a contagem detectada com parâmetros $\theta$,
$c_i$ é o valor real (*ground truth*) e $N$ é o número de imagens.

O espaço de busca é:

| Parâmetro   | Valores testados           | Total de candidatos |
|-------------|----------------------------|---------------------|
| `dp`        | `[1]`                      | 1                   |
| `minDist`   | `[3, 5, 8, 12]`            | 4                   |
| `param1`    | `[5, 10, 15]`              | 3                   |
| `param2`    | `4 … 24`                   | 21                  |
| `minRadius` | `[1, 3, 5]`                | 3                   |
| `maxRadius` | `[20, 30, 40, 50]`         | 4                   |

$$
|\Theta| = 1 \times 4 \times 3 \times 21 \times 3 \times 4 = 3\,024 \text{ combinações}
$$

A busca para antecipadamente quando `score = N` (acerto total), evitando
iterações desnecessárias.

---

## Resultados

Hiperparâmetros ótimos encontrados:

```json
{
    "dp": 1,
    "minDist": 8,
    "param1": 10,
    "param2": 9,
    "minRadius": 1,
    "maxRadius": 20
}
```

Desempenho no conjunto de treinamento (**4/4 corretos**):

| Imagem    | Detectado | Esperado | Status  |
|-----------|-----------|----------|---------|
| img1.jpg  | 11        | 11       | Correto |
| img2.jpg  | 4         | 4        | Correto |
| img3.jpg  | 15        | 15       | Correto |
| img4.jpg  | 7         | 7        | Correto |

---

## Uso

### Instalação

```bash
# Instalar dependências com uv
uv sync
```

### Treinar (otimizar hiperparâmetros)

Percorre o grid, encontra os melhores parâmetros, salva em `resultados.txt`
e gera `output.png` com as detecções visualizadas.

```bash
uv run main.py treinar
```

### Analisar uma imagem

Usa os parâmetros salvos em `resultados.txt` para contar os pips de uma
imagem específica.

```bash
uv run main.py analisar --nome_imagem img1.jpg
```

> **Pré-requisito:** executar `treinar` ao menos uma vez antes de `analisar`.

---

## Dependências

| Biblioteca       | Finalidade                              |
|------------------|-----------------------------------------|
| `opencv-python`  | Processamento de imagem e Hough         |
| `numpy`          | Operações vetoriais                     |
| `matplotlib`     | Visualização dos resultados             |
