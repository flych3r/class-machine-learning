# Trabalho 4 - K-means, PCA, Árvores de Decisão

## Implementação e Relatório

1. k-means
   * Implemente o k-means usando a distância euclidiana.
   * Execute o k-means para k ={2,3,4,5}
     * a. Plote a distância média de cada ponto para o seu centroide em um gráfico linha em função de k (média sobre 20 rodadas)
     * b. Discuta qual seria o k ideal a ser usado
2. PCA
    * Implemente o PCA
        * a. Você deve implementar a função de calcular a matriz de covariância
        * b. A função de achar os autovetores e os autovalores pode ser usado pronto do [numpy](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html)
    * Reduza o conjunto de dados original em um conjunto com apenas duas variáveis (2 componentes principais de maior autovalor)
        * a. Reporte quanto de variância foi preservado
        * b. Plote cada ponto do conjunto transformado em um gráfico de dispersão 2d  atribuindo uma cor para cada uma das classes (3 classes no total).

3. Árvores de decisão
    * Implemente a árvore de decisão usando o coeficiente de Gini como mostrado em sala
    * Reporte o erro de classificação para o k-fold com k=5
        * a. Pode usar o k-fold que foi implementado em atividades passadas ou pode usar pronto do scikit-learn
        * b. Erro de classificação pode usar pronto do scikit-learn também

## Conjunto de dados

Carregar trab4.data
