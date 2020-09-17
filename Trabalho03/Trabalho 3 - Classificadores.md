# Trabalho 3

## Classificadores 2

Implementação

  1. Implemente os seguintes métodos:
      * a. Rede MLP para classificação
        * i. Apenas uma camada oculta (recebe o tamanho dessa camada como parâmetro)
        * ii. Pode assumir que tem apenas um neurônio de saída
      * b. KNN
        * i. Recebe k como parâmetro
  2. Usaremos a funções acurácia, plot_confusion_matrix e plot_boundaries do trabalho passado
  3. Implementar função k_fold(X, y, k, método) que execute a validação cruzada k-fold sobre o conjunto de dados X,y usando o e reportando o erro usando função acurácia (usar k=5). Não precisa implementar parte de validação e teste, implementar somente o fluxo principal (como está no primeiro slide sobre k-fold)

Dado

  1. Carregar data1.txt
  2. As duas primeiras colunas são as características e a última coluna é a variável alvo

Relatório

  1. Reporte o que se pede usando os métodos KNN (com k=1,2 e 3) e MLP (como número de neurônios na camada oculta 2, 3 e 4):
        * a. O erro do 5-fold
        * b. O dado em um gráfico de dispersão com as fronteiras de separação produzidas pelo método treinado com o conjunto de dados inteiro
