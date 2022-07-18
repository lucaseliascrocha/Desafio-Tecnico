# Desafio Técnico - Americanas

Abaixo descreverei os processos de desenvolvimento do modelo de aprendizado de máquina para classificar as amostras do conjunto de dados `dataset_cdjr.parquet.gzip`.
    
## Análise Exploratória
(Detalhes em `Analise_exploratoria.ipynb`)
  
Verifica-se que não há dados faltantes e todos são numéricos. Agora, com o auxílio da ferramenta Pandas Profiling, podemos analisar alguns aspectos dos dados no arquivo `Análise Exploratória.html`.

### Aspectos que chamaram atenção
- Boa distribuição de classes;
  - 0: 44,2%
  - 1: 55,8%
- Nenhuma feature com variância zero;
- Features com grande proporção de valores distintos (acima de 90%):
  - feature3: 98.3%
  - feature4: 94.4%
  - feature7: 100%
  - feature9: 99.8%
  - feature10: 95.7%

Apesar dessa grande proporção de valores distintos, não vemos correlação desse comportamento com a simetria da distribuição de valores. No geral, as features apresentam uma assimetria em suas distribuições de valores, em destaque para as features feature7 e feature11. Essa característica pode resultar em complicações para alguns tipos de algoritmos de ML.

- Alta correlação entre features:
  - feature0 e feature15
  - feature0 e feature4
  - feature2 e feature15
  - feature4 e feature15
  - feature6 e feature9  

![alt text](/imagens/Correla%C3%A7%C3%A3o.png)

- As classes se concentram nas mesmas regiões de valores para cada feature. Porém, podemos observar outliers.

## Preparação dos dados
(Detalhes em `Preparação_dos_Dados.ipynb`)
  
### Remoção de Outliers
Considerei efetura remoção de outlier para as features por meio de intervalo interquatil. Porém, no total, muitas instâncias seriam removidas e, se tratando de uma coleção pequena, restaria pouco dado.

### Transformação dos dados
Através da biblioteca Pycaret aplicaremos o seguinte pré-processamento dos dados:
- Normalização (zscore);
- Transformação (quantile).

### Seleção de Features

Dado a ocorrência de features correlatas, considerei pertinente aplicar uma seleção de feature buscando evitar redundância de informação para o modelo. Outro ponto é buscar eliminar features ruidosas.
  
Inicialmente, analisei a importância das features para o algoritmo Random Forest. Contudo teria um problema: feature importance usando mean decrease in impurity para algoritmos baseados em árvores possue a tendência em favorecer features com grande proporção de valores distintos. Como vimos na análise exploratória, temos algumas features com essa característica. Assim, optei por adicionar seleção de feature de acordo com outros algoritmos. A escolha desses algoritmos se deu na tentativa de experimentar diferentes abordagens:
- Algoritmo baseado em árvores: Random Forest;
- Support vectors: SVM;
- Gradiente: SGD;
- Modelo linear: Ridge.

Logo, cheguei em 4 grupos de features selecionadas:
- Conjunto de features selecionadas a partir da feature importance do algoritmo Random Forest
  - feature3, feature4, feature5, feature6, feature7, feature9, feature10
- Conjunto de features selecionadas a partir da feature importance do algoritmo SVM
  - feature0, feature2, feature5, feature7, feature10, feature13, feature15
- Conjunto de features selecionadas a partir da feature importance do algoritmo SGD
  - feature4, feature7, feature8, feature10, feature12, feature15
- Conjunto de features selecionadas a partir da feature importance do algoritmo Ridge
  - feature2, feature4, feature6, feature7, feature14, feature15

## Modelagem
(Detalhes em `Modelagem_e_Avliação.ipynb`)
  
Para realização da modelagem utilizei a ferramenta Pycaret visto que optei por não executar algoritmos neurais e a ferramenta Pycaret disponibiliza disversos algoritmos não neurais, implementando tarefas como pré processamento dos dados e otimização de parâmetros dos algoritmos.  
A decisão de não executar modelos neurais passam por 2 aspectos:
- Tempo insuficiente para tunning do modelo: modelos neurais possuem nessecidade de tunning e alto custo de parametrização;
- Pequeno conjunto de dados: alto risco de overffiting com modelos neurais.

Como trabalhado na etapa de preparação dos dados, experimentei cinco conjuntos de features:
- Todas as features;
- Conjunto de features selecionadas a partir da feature importance do algoritmo Random Forest;
- Conjunto de features selecionadas a partir da feature importance do algoritmo SVM;
- Conjunto de features selecionadas a partir da feature importance do algoritmo SGD;
- Conjunto de features selecionadas a partir da feature importance do algoritmo Ridge.

Assim como na etapa de preparação dos dados, na experimentação avaliei 5 algoritmos escolhidos para contemplar diferentes abordagens. Os algoritmos experimentados foram:
- Random Forest (RF);
- Gradient Boosting Classifier (GBC);
- Support Vector Machine (SVM);
- Ridge;
- Logistic Regression (LR).

Cada algoritmo foi avaliado com todos os conjuntos de features da seleção de features.

Abaixo temos o melhor modelo obtido com uma divisão dos dados em 80% treino e 20% validação e 5-fold cross validation. O resultado dos outros modelos pode ser encontrado [aqui](https://docs.google.com/spreadsheets/d/1Z2nioEfYBve8F2uLDxee3aMmD04uB289PIVw8K1nZhI/edit?usp=sharing).

- Algoritmo: Gradient Boosting Classifier (GBC)
- Parâmetros:
  - ccp_alpha: 0.0
  - criterion: friedman_mse
  - learning_rate: 0.15
  - loss: deviance
  - max_depth: 1
  - max_features: log2
  - min_impurity_decrease: 0.5
  - min_samples_leaf: 3
  - min_samples_split: 2
  - min_weight_fraction_leaf: 0.0
  - n_estimators: 270
  - subsample: 0.55
  - tol: 0.0001
  - validation_fraction: 0.1
  - verbose: 0
  - warm_start: False

## Avaliação da Performance do Modelo
(Detalhes em `Modelagem_e_Avliação.ipynb`)

O modelo final apresenta o seguinte desempenho para os dados de validação:

| Accuracy | AUC | Recall | Prec. | F1 |
|----------|-----|--------|-------|----|
| 0.7071 | 0.7378 | 0.7683 | 0.7223 | 0.7444 |	

Dentre os 25 modelos avaliados, o modelo final apresentou a melhor acurácia, AUC e F1. 
  
![alt text](/imagens/class_report.png)

Como podemos observar no gráfico acima, o modelo tem um desempenho melhor para identificar a ocorrência do evento do que a não ocorrência. Essa diferença se dá principalmente na métrica de precisão. Ou seja, a ocorrência de falso positivo é menor do que a ocorrência de falso negativo.

## Entrega do modelo
(Modelo salvo no arquivo `modelo.pkl`)

### Executar o modelo
Para executar o modelo é necessário utilizar a biblioteca Pycaret. Por meio dela carregamos o modelo e executamos a predição diretamente sobre os dados originais da instância de entrada. Outra possibilide é executar a predição sobre um dataframe. A seguir temos um exemplo:

```python
import pandas as pd
from pycaret.regression import load_model, predict_model

sample = {
    "feature0": 200.0,
    "feature1": 2,
    "feature2": 662.28,
    "feature3": 39.0,
    "feature4": -188.55,
    "feature5": 0.246978,
    "feature6": 761,
    "feature7": 0.004548,
    "feature8": 3.523703,
    "feature9": 167326,
    "feature10": 33441.06,
    "feature11": 0.019804,
    "feature12": 26.850,
    "feature13": 0.009198,
    "feature14": 94.611429,
    "feature15": 7
  }

model = load_model('/content/drive/MyDrive/Americanas/modelo')

predict = predict_model(
                    model,
                    data=pd.DataFrame([sample])
            )['Label'][0]

print(f'Valor predito: {predict}')
```

### API
Também implementei uma API onde, por meio de um post em https://desafio-tecnico-americanas.herokuapp.com/predict podemos obter a predição do modelo. Segue um exemplo de execução da api:

```Curl
curl -X 'POST' \
  'https://desafio-tecnico-americanas.herokuapp.com/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "feature0": 200.0,
  "feature1": 2,
  "feature2": 662.28,
  "feature3": 39.0,
  "feature4": -188.55,
  "feature5": 0.246978,
  "feature6": 761,
  "feature7": 0.004548,
  "feature8": 3.523703,
  "feature9": 167326,
  "feature10": 33441.06,
  "feature11": 0.019804,
  "feature12": 26.850,
  "feature13": 0.009198,
  "feature14": 94.611429,
  "feature15": 7
}'
```

Outra forma de utilizar a api é por meio do [link](https://desafio-tecnico-americanas.herokuapp.com/docs#/default/get_predict_predict_post) na aba POST /predict, clicando em "Try it out", definindo os valores de cada feature e executando.