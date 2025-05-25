# 📚 Sistema de Recomendação de Livros com KNN

### Este projeto é um sistema de recomendação de livros desenvolvido em Python que utiliza Machine Learning (KNeighborsClassifier) para sugerir livros aos usuários com base em suas preferências de leitura anteriores.

### O sistema combina dados de três fontes principais (books.csv, ratings.csv e users.csv) para construir um modelo de recomendação. Após selecionar um usuário e um dos livros que ele já leu, o sistema utiliza o algoritmo K-Nearest Neighbors (KNN) para encontrar livros semelhantes que outros usuários com gostos parecidos também leram.

## 🧠 Funcionalidades
### Leitura e pré-processamento de dados (remoção de duplicatas, limpeza de colunas).

### Treinamento do modelo com o algoritmo KNN.

### Recomendação personalizada de livros com base no livro selecionado pelo usuário.

### Interface interativa construída com Streamlit.

### Visualização das capas dos livros recomendados.

## 🗂️ Arquivos esperados
### books.csv: Contém informações sobre os livros (título, autor, ano de publicação, etc.).

### ratings.csv: Avaliações que os usuários deram aos livros.

### users.csv: Informações sobre os usuários (localização, idade, etc.).

## 🚀 Como executar

