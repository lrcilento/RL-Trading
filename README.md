# Projeto 4 - Inteligência Artificial Aplicada

## Requisitos:
  - Python 3.7
  - Dependências em 'requirements.txt'

## Metodologia:
1. As documentações das bibliotecas gym_anytrading e stable-baselines foram lidas extensivamente.
2. Ambas bibliotecas foram integradas afim de utilizar os algoritimos de RL da stable-baselines.
3. Foram testados os seguintes algorimitos da biblioteca, cada um com pelo menos 50 combinações de parâmetros diferentes, tanto no ginásio 'forex-v0' quanto no 'stocks-vo':
  - A2C
  - ACER
  - ACKTR
  - DQN
  - PPO1
  - PPO2

## Resultados:
Os resultados obtidos foram parcialmente satisfatórios, a melhor combinação consegue chegar consistentemente marginalmente acima de **1.01x** o valor inicial, sendo que o agente aleatório varia entre **0.89x e 0.92x**, porém o valor máximo hipotético varia entre **1.25x e 1.3x** dependendo do conjunto selecionado, portanto ainda existe muito espaço para melhoria.

## Definição dos Arquivos:
  - **Random**.py: Agente aleatório usado para controle de resultados.
  - **Baseline Example**.py: Primeira iteração do agente integrado com a biblioteca stable-baselines, pré-treino.
  - **Attempt Training**.py: Arquivo utilizado para testar as mais de 200 combinações de algoritimos, policies, ginásios e parâmetros.
  - **DQN Never-Buy**.py: Achei interessante incluir este pois foi uma das combinações que achei curiosas, enquanto no ginásio de ações, o algoritimos DQN, dada iterações de aprendizado suficientes, sempre opta por nunca comprar nada e terminar a simulação com o valor exato de **1.0x do valor** inicial.
  - **Final**.py: Se trata da versão final do algoritimo que desejo apresentar como resultado deste projeto, como dito anteriormente ele trabalha com a melhor combinação de variáveis que eu encontrei para conseguir um resultando consistente de pelo menos **1.01x do valor inicial**.


  
