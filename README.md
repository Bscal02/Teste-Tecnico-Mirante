# Teste Técnico – Data Scientist Mirante

## Resumo

O problema em questão envolve a predição de churn dos clientes de uma telefônica, sendo então o objetivo deste case a criação de um modelo de Machile Learning capaz de realizar a classificação de forma razoável dos clientes em possíveis alvos em uma campanha para impedir o churn. Durante a EDA não foram identificados dados nulos ou outliers, portanto o pré-processamento deles se limitou à encodificação de features categóricas, utilizando a estretégia ordinal, e à normalização de todas as features. Em termos de modelos candidatos, foram avaliados seis, logístico, Random Forest, k-nearest neighbors, GradientBoosting e DecisionTree, e após a seleção, utilizando como critério a AUC, o melhor modelo identificado foi o GradientBoosting. Também foi feito um tunning dos hiper parâmetros do modelo escolhido através de uma busca extensa de alguns candidatos de valores, resultando nos valores de learning_rate igual a 0.1, max_depth igual a 1 e n_estimators igual a 100. Durante esse processo também foi usado um algoritmo de seleção de features nativo do pacote sklearn. No fim o melhor modelo resultou em um AUC de teste de 0.64 e uma acurácia de 0.65. Em termos de classes, o grupo que não realizou churn obteve uma precisão de 0.65, recall de 1.00 e F1-score de 0.79, porém o grupo que realizou churn obteve uma precisão de 0.56, recall de 0.01 e F1-score de 0.01, indicando que o modelo não foi bem sucedido em identificar essa classe. Através da análise dos SHAP values, a feature mais importante identificado foi a DeviceProtection, tendo mais que o dobro da importância do segundo lugar, indicando que a contratação ou não da segurança de dispositivos é um fator importante para o churn, porem os scores em geral do SHAP foram baixos, incluindo vários com praticamente zero importância, refletindo o baixo desempenho do modelo.

Em conclusão, podemos dizer que os dados disponíveis para análise não carregam informação suficiente para uma modelagem satisfatória do comportamento de churn dos clientes, sendo nescessário uma investigação mais profuunda sobre outros possíveis fatores que melhor representam essa tendência.

## Construído com

* [pandas](https://github.com/pandas-dev/pandas) Manipulação de dados
* [numpy](https://github.com/numpy/numpy) Operações matemáticas
* [matplotlib](https://github.com/matplotlib/matplotlib) Grafícos
* [seaborn](https://github.com/mwaskom/seaborn) Grafícos
* [sklearn](https://github.com/scikit-learn/scikit-learn) Modelos e métricas
* [shap](https://github.com/shap/shap) Métricas

## Autor

**Bernardo Souza Scaldaferri**