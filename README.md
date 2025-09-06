# **Detecção de Fraudes em Transações de Cartão de Crédito**

![Capa do Projeto](https://github.com/VlaadX/Fraud-Detection-ML-Pipeline/blob/main/imgs/cover.png)  

O problema de **detecção de fraudes** é um dos desafios mais importantes e de alto valor no setor financeiro. A cada dia, bilhões de transações são realizadas, e a capacidade de identificar e bloquear atividades fraudulentas em tempo real é crucial para bancos, fintechs e e-commerce.

Este projeto foca em construir um pipeline completo de Machine Learning para a detecção de fraudes em transações de cartão de crédito, abordando os desafios inerentes de um dataset altamente desbalanceado.

---
## **1.0 O Problema do Negócio e a Falsa Acurácia de 99%**

No mundo das transações financeiras, a detecção de fraude é um problema crítico. A prioridade é identificar transações suspeitas com o mínimo de falsos positivos, para evitar prejuízos financeiros e garantir a confiança do cliente.

Um dos maiores desafios é o **desbalanceamento de classes**. Em nosso dataset, as transações fraudulentas representam apenas **0.172%** do total. Isso cria uma armadilha: um modelo ingênuo poderia prever "não-fraude" para todas as transações, atingindo uma acurácia de mais de 99%, mas sendo completamente inútil na prática.

Para resolver isso, o projeto focou em métricas de avaliação que realmente importam:
- **Precision:** Das transações previstas como fraude, quantas são realmente fraude? (minimiza falsos alarmes)
- **Recall:** De todas as fraudes reais, quantas o modelo capturou? (minimiza perdas financeiras)
- **F1-Score:** O equilíbrio entre Precision e Recall.

---
## **2.0 Análise e Pré-processamento dos Atributos**

A análise exploratória (EDA) nos permitiu entender a natureza dos dados e tomar decisões de pré-processamento cruciais.

### **2.1 Insights da Análise Exploratória (EDA)**

* **Distribuição de Valores (`Amount`):** A análise de densidade mostrou que as fraudes tendem a ter valores de transação menores, um insight valioso para os modelos.
![Gráfico de Densidade - Distribuição de Valor](https://github.com/VlaadX/Fraud-Detection-ML-Pipeline/blob/main/imgs/Distribui%C3%A7%C3%A3o%20de%20Valores.png)

* **Distribuição de Tempo (`Time`):** Transações legítimas seguem padrões de picos e vales diários, enquanto as fraudes ocorrem de forma mais aleatória, sem um padrão de tempo definido.
![Gráfico de Distribuição - Distribuição de Tempo](https://github.com/VlaadX/Fraud-Detection-ML-Pipeline/blob/main/imgs/Distribui%C3%A7%C3%A3o%20de%20Tempo.png)

### **2.2 O Processo de Pré-processamento**

As colunas `Time` e `Amount` foram padronizadas com o `StandardScaler`. Em seguida, o dataset foi dividido em conjuntos de treino e teste de forma **estratificada** (`StratifiedShuffleSplit`), garantindo que a proporção de fraudes fosse preservada em ambos os conjuntos, o que assegura uma avaliação justa e precisa do modelo.

---
### **3.0 Comparação e Avaliação dos Modelos**

A alma do projeto reside na comparação entre modelos. Nosso objetivo não era apenas encontrar o modelo com o melhor desempenho técnico, mas sim aquele que oferece a solução mais pragmática e benéfica para o negócio. Para todos os modelos, utilizamos técnicas como `class_weight` e `scale_pos_weight` para mitigar o viés do desbalanceamento.

### **3.1 Análise Detalhada dos Modelos Tradicionais**

Comparar a performance de modelos tradicionais como a Regressão Logística, Random Forest e XGBoost nos permitiu identificar a abordagem mais promissora para a detecção de fraude. Cada modelo revelou um comportamento único ao lidar com o problema.

| **Modelo** | **Precision** | **Recall** | **F1-Score** | **ROC-AUC** | **AP** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.9605** | 0.7448 | 0.8390 | 0.9529 | 0.8542 |
| **XGBoost** | 0.8817 | **0.8367** | **0.8586** | **0.9682** | **0.8800** |
| **Regressão Logística** | 0.0609 | 0.9183 | 0.1143 | 0.9720 | 0.7189 |

* **Regressão Logística: O Agente Hiper-agressivo.** Este modelo demonstrou um comportamento extremista em sua busca por fraudes. Com um **Recall de 91.83%**, ele foi extremamente eficaz em capturar quase todas as fraudes. No entanto, sua **Precision de apenas 6.09%** revela um grande problema de falsos positivos: de cada 100 alertas de fraude, mais de 93 seriam falsos. Para um negócio, isso resultaria em uma avalanche de bloqueios de transações legítimas e clientes insatisfeitos.

* **Random Forest: O Guardião Cauteloso.** Em contrapartida, o Random Forest se mostrou o mais preciso. Com uma **Precision de 96.05%**, ele errou muito pouco quando previu uma fraude. O trade-off, no entanto, foi seu **Recall de 74.48%**, que indica que ele deixou passar uma parte significativa das fraudes reais. Embora não gere tantos falsos alarmes, este modelo ainda permite que uma boa parcela das atividades fraudulentas não seja detectada.

* **XGBoost: O Mestre do Equilíbrio.** O XGBoost demonstrou um desempenho superior e equilibrado. Com um **Recall de 83.67%** e uma **Precision de 88.17%**, ele conseguiu capturar a maioria das fraudes sem gerar um número excessivo de falsos alarmes. Esta combinação ideal de métricas resultou no maior **F1-Score de 0.8586**, provando ser a solução mais robusta e pragmática para a tarefa.

### **3.2 Análise da Rede Neural**

Uma rede neural densa (`DNN`) foi construída com **Keras** e **TensorFlow** para verificar se uma abordagem de Deep Learning poderia superar o XGBoost.

![Logs de treinamento](https://github.com/VlaadX/Fraud-Detection-ML-Pipeline/blob/main/imgs/rede.png)

| **Modelo** | **Precision** | **Recall** | **F1-Score** | **ROC-AUC** | **AP** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Rede Neural** | 0.1250 | **0.8980** | 0.2195 | **0.9761** | 0.7431 |

* **Alto Desempenho Teórico, Baixa Usabilidade Prática.** O modelo de Deep Learning obteve um **Recall impressionante de 89.80%**, superando o XGBoost, e a melhor **ROC-AUC**, indicando uma excelente capacidade teórica de diferenciar as classes. No entanto, sua **Precision de apenas 12.50%** é inaceitável para uso em produção. Sua estratégia de "capturar tudo" resultaria em um volume tão grande de falsos positivos que o modelo seria descartado na prática.

### **3.3 Conclusão Final**
  A análise comparativa deixa claro que, para o problema de detecção de fraudes, a solução não está apenas em obter a maior pontuação em uma única métrica. O modelo **XGBoost** provou ser a melhor escolha por oferecer o balanço perfeito entre **Precision** e **Recall**, minimizando tanto o risco de perdas financeiras (fraudes) quanto os prejuízos de relacionamento com o cliente (falsos alarmes).

---
## **4.0 Estrutura do Projeto e Tecnologias**

O projeto foi estruturado em um pipeline de 4 etapas claras, documentadas em notebooks Jupyter para garantir a reprodutibilidade.

* **Tecnologias:** Python 3.10+, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, TensorFlow, Keras.
* **Estrutura:**
    * `notebooks/` - Contém os notebooks para cada etapa (EDA, Pré-processamento, Modelos Tradicionais e Deep Learning).
    * `data/` - Armazena o dataset original e os arquivos processados (`X_train.csv`, etc.).

---
## **5.0 Como Executar o Projeto**

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/VlaadX/Fraud-Detection-ML-Pipeline
    cd Fraud-Detection-ML-Pipeline
    ```
2.  **Instale as dependências:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow
    ```
3.  **Execute os notebooks na ordem correta:**
    * `01_eda.ipynb`
    * `02_preprocessing.ipynb`
    * `03_traditional_models.ipynb`
    * `04_deep_learning.ipynb`
