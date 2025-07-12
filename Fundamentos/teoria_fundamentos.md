# Fundamentos de Machine Learning - Teoria

Este documento serve como apoio conceitual para o módulo **Fundamentos de Machine Learning**. Ele explica os conceitos fundamentais por trás dos modelos supervisionados de classificação, com foco em entendimento prático e matemático.

---

## 1. O que é Aprendizado Supervisionado

- Definição geral do problema
- Conceito de amostras (`X`) e rótulos (`y`)
- Problemas de classificação vs. regressão
- Objetivo: aprender uma função \$f(X) \rightarrow y\$

### Exemplo:

- Diagnóstico de diabetes baseado em variáveis clínicas

---

## 2. Overfitting e Underfitting

- **Underfitting:** modelo muito simples, não aprende o padrão
- **Overfitting:** modelo aprende até o ruído dos dados
- Bias (viés) e Variance (variância)

### Gráfico:

- Inserir gráfico ilustrando:
  - Curva de erro no treino e validação vs. complexidade do modelo

---

## 3. Pipeline de Treinamento

1. Pré-processamento dos dados
2. Separação em treino/teste
3. Escolha de modelo
4. Treinamento (`fit`)
5. Predição (`predict`)
6. Avaliação de desempenho

---

## 4. Métricas de Avaliação

### Para classificação binária:

- Acurácia
- Precisão
- Recall (Sensibilidade)
- F1-score
- Curva ROC e AUC

### Tabela de Confusão:

- Verdadeiro Positivo (VP), Falso Positivo (FP), etc.

### Fórmulas:

```
Precisão = VP / (VP + FP)
Recall = VP / (VP + FN)
F1 = 2 * (Precisão * Recall) / (Precisão + Recall)
```

---

## 5. Validação Cruzada

- K-Fold Cross-Validation
- Redução da variância na avaliação
- Quando e por que usar

---

## 6. Modelos Clássicos de Classificação

### 1. Regressão Logística

- Interpretação probabilística
- Equação de decisão: \$\sigma(wX + b)\$

### 2. Árvore de Decisão

- Regras baseadas em perguntas
- Interpretação simples

### 3. Random Forest

- Conjunto de árvores
- Redução de overfitting

### 4. KNN (K-vizinhos mais próximos)

- Baseado em distância
- Simples, mas custoso em produção

### Comparação:

| Modelo              | Vantagens              | Desvantagens                  |
| ------------------- | ---------------------- | ----------------------------- |
| Regressão Logística | Simples, interpretável | Não lida com não-linearidades |
| Árvore              | Interpretação clara    | Overfitting fácil             |
| Random Forest       | Robusto e preciso      | Mais lento para predição      |
| KNN                 | Intuitivo              | Custo computacional alto      |

---

## 7. Datasets Balanceados vs. Desequilibrados

- Problemas com acurácia em datasets desbalanceados
- Estratégias:
  - Ajuste de pesos
  - Amostragem (oversampling / undersampling)

---

## 8. Considerações Éticas

- Viés nos dados
- Responsabilidade no uso
- Transparência de modelos

---

## 9. Referências

- [Hands-On ML com Scikit-Learn, Keras e TensorFlow — Aurélien Géron]
- [Curso de Machine Learning — Andrew Ng (Coursera)]
- [Documentação oficial do Scikit-learn](https://scikit-learn.org/stable/)

---
