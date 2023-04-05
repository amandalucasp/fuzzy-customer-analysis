# Customer Analysis using Fuzzy C-Means and K-Means

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

### Base de Dados

O conjunto de dados utilizado para o presente trabalho é o "Customer Personality Analysis" disponível no Kaggle. É formado por 2240 amostras, que consistem em informações relativas aos gastos de clientes da empresa, separados por setores: compra de vinhos, frutas, carne, peixe, doces e ouro.

- [Download dos dados](https://www.kaggle.com/imakash3011/customer-personality-analysis)


Esse projeto foi desenvolvido como trabalho final da disciplina "Lógica Fuzzy" da pós-graduação em Engenharia Elétrica da PUC-Rio.
