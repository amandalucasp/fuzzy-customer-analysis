# Customer Analysis using Fuzzy C-Means

Esse projeto foi desenvolvido como trabalho final da disciplina "Lógica Fuzzy" da pós-graduação em Engenharia Elétrica da PUC-Rio.

A proposta do projeto é comparar a performance do K-Means e do Fuzzy C-Means em um conjunto de dados de consumo de clientes de uma dada empresa, buscando encontrar uma forma satisfatória de particionar os mesmos. Adicionalmente, é realizada uma análise da influência do valor do coeficiente nebuloso no ajuste do modelo aos dados. 

### Etapas do projeto

1. Limpeza dos dados
2. Análise exploratória dos dados
3. Pré-processamento dos dados
4. Split em treino/validação/teste
5. Normalização dos dados
6. Experimentos com os algoritmos testados
  * Baseline: usando os dados "crus"
  * Utilizando PCA (Principal Component Analysis), variando o número de componentes utilizadas
  * Utilizando diferentes valores para c/k (número de clusters)
  * Análise da influência do coeficiente nebuloso m

### K-Means e Fuzzy C-Means

No K-Means, o conjunto de dados é particionado em k grupos, sendo que cada grupo tem como seu centro a média das amostras pertencentes a este. O parâmetro k que define o número de clusters deve ser definido pelo especialista. Por atribuir cada amostra a um único cluster, o K-Means é um algoritmo do tipo "hard clustering".

Já o Fuzzy C-Means é um algoritmo do tipo "soft clustering", e atribui para cada objeto valores diferentes da função de pertinência. Os valores de pertinência de uma amostra são referentes a cada cluster, de forma que um mesmo objeto possa pertencer a um ou mais grupos. Métodos desse tipo são interessantes em aplicações que envolvam dados onde se espera um "overlap" das classes/clusters finais. 

### Base de Dados

O conjunto de dados utilizado para o presente trabalho é o "Customer Personality Analysis" disponível no Kaggle. É formado por 2240 amostras, que consistem em informações relativas aos gastos de clientes da empresa, separados por setores: compra de vinhos, frutas, carne, peixe, doces e ouro.

- [Download dos dados](https://www.kaggle.com/imakash3011/customer-personality-analysis)
