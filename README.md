﻿## Inteligência Artificial 2023.1

Autores: João Gabriel Alvez de Souza, João Pedro Silva Cunha

Nesse repositorio você encontra o código fonte e um breve relatório (KNN-IA.pdf) do experimento aplicado.

Link para apresentação: https://docs.google.com/presentation/d/1_FIpQYelhgiPyLowwOVkEWHCS25-TRzQoDAvn4MEAoY/edit?usp=sharing

# Algoritmo 

KNN - Algoritmo de classificação muito usado em aprendizado de máquina que usa o conceito de aprendizado supervisionado. O algoritmo opera tomando um conjunto como base de conhecimento, e a vizinhança do elemento atual e de assim tomar previsões.

# Como executar 

Na pasta fonte tem-se uma pasta com as bases de dados escolhidas, um arquivo com funções implementadas desde o algoritmo propriamente dito até funções auxiliares, esse arquivo vai servir como módulo. E um arquivo principal que deverá ser executado para rodar o programa.

```
/bin/python3 main.py
```

# O programa

Ao executar o programa o usuário deverá escolher:

* Um número de vizinhos a considerar (K).

* A porcentagem dos dados que serão usados como base de conhecimento.

feito isso se espera uma saída semelhante a isso...

```
Num Vizinhos:  10
Base de conhecimento:  50 %
Tempo de execucao =  0.12923383712768555

Acertos :  74 
Taxa de acertos:  98.66666666666667 %
```

em seguida ao digitar qualquer tecla, retorna-se ao ínicio da execução do programa.
