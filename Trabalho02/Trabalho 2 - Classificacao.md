# Trabalho 2

## Classificadores

Implementação

  1. Implemente os seguintes métodos:
        * a. Regressão Logística – Gradiente Descendente
        * b. Naive Bayes Gaussiano
        * c. Discriminante Quadrático Gaussiano

  2. Implemente a função acurácia(y_true, y_pred) que retorna o a porcentagem de acerto de y_true.

        ```python
        >>> y_true  = [1,2,3,2,3,1]
        >>> y_pred = [1,3,3,1,3,2]
        >>> acurácia(y_true, y_pred)
        0.5
        ```

  3. Implemente uma função que receba um classificador e o conjunto de testes e exiba uma matriz de confusão:
assinatura: plot_confusion_matrix(X, y, clf)
Código exemplo: <https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html>

  4. Implemente uma função que receba o classificador e o conjunto de teste e exiba o conjunto de testes em um gráfico de dispersão juntamente com as fronteiras de separação do classificador.
assinatura: plot_boundaries(X, y, clf)
Código exemplo: <https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html>

Conjunto de dados

  1. Carregar trab2.data

  2. A duas primeiras colunas são as características e a última coluna é a variável alvo

  3. Usar 70% do conjunto para treino e 30% para teste

Relatório

  1. Para cada uma das técnicas de classificação apresente:

     * a. A porcentagem de predições corretas para o conjunto de teste usando a função acurácia.
     * b. A matriz de confusão.
     * c. O dado em um gráfico de dispersão com as fronteiras de separação.

  2. O dado parece se linearmente separável ou não?

  3. Quais dos métodos produziu fronteiras lineares?

  4. Qual teve melhor acurácia?
