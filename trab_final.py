import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image
import requests

url='books.csv'
df_book = pd.read_csv(url,sep=',')

#Rating
url='ratings.csv'
df_rating = pd.read_csv(url,sep=',')

#Users
url='users.csv'
df_users = pd.read_csv(url,sep=',')

print(df_book.shape, df_rating.shape, df_users.shape)



duplicados = df_book[df_book.duplicated(keep='first')]
print(duplicados)

# excluindo repetições
df_book.drop_duplicates(subset ='ISBN', keep='first', inplace=True) 
df_book.drop_duplicates(subset ='Book_Title', keep='first', inplace=True)
#df_book.drop_duplicates(subset ='Book_Author', keep='first', inplace=True)
#df_book.drop_duplicates(subset ='Year_Of_Publication', keep='first', inplace=True)
df_book.drop_duplicates(subset ='Publisher', keep='first', inplace=True)



print(df_book.isna().sum())

# juntando os dataframes ratings e book pela coluna ISBN
ratings_with_name = df_rating.merge(df_book, on='ISBN')

ratings_with_name.head()

data = ratings_with_name.merge(df_users.drop("Age", axis=1), on="user_id")
data.head()

data['Location'] = data['Location'].str.split(',').str[-1].str.strip()

data.Location.value_counts()

data.head()


# dividir o dataframe que foi juntado em conjunto de treinamento e conjunto de teste
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# selecionar as colunas relevantes para treinamento
X_train = train_data[['ISBN', 'Book_Rating', 'Year_Of_Publication']]
y_train = train_data['Book_Title']

# selecionar as colunas relevantes para teste
X_test = test_data[['ISBN', 'Book_Rating', 'Year_Of_Publication']]
y_test = test_data['Book_Title']

from sklearn.neighbors import KNeighborsClassifier



# utilizando o parametro n_neighbors = 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Precisao: {:.2f}%".format(accuracy * 100))


# treinar o modelo utilizando o algoritmo KNN
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train, y_train)

def recommendation(user_ratings, k=10):
    # obter os índices dos k vizinhos mais próximos
    distances, indices = knn.kneighbors(user_ratings)
    
    # contar quantas vezes cada livro foi recomendado
    book_count = {}
    for i in range(len(indices)):
        for j in range(len(indices[i])):
            book = X_train.iloc[indices[i][j]]['ISBN']
            if book in book_count:
                book_count[book] += 1
            else:
                book_count[book] = 1
    
    # selecionar os 10 livros mais recomendados
    top_books = sorted(book_count.items(), key=lambda x: x[1], reverse=True)[:k]
    
    # retornar a lista de livros recomendados
    return [book[0] for book in top_books]


livros = pd.read_csv('novo_dados.csv')

st.title('Recomendação de Livros')

usuarios = livros['user_id'].drop_duplicates()
usuarios_id = st.selectbox('Selecione um Usuário:', sorted(usuarios))

info = data[data['user_id'] == usuarios_id]


t_livros = info['Book_Title'].drop_duplicates()
u_livros = st.selectbox('Este são os livros que o usuário escolhido leu. Selecione uma opção:',t_livros)

st.text("OBS: Alguns livros estão sem imagem.")


# informando o id do usuário e o nome de um livro para ter o retornos dos livros recomendados
user_id = usuarios_id 
book_title = u_livros 
book_ISBN = data[data['Book_Title'] == book_title]['ISBN'].values[0] 

user_ratings = data[data['user_id'] == user_id][['ISBN', 'Book_Rating', 'Year_Of_Publication']]
user_ratings.loc[len(user_ratings)] = [book_ISBN, 0, 0] 

user_recommendations = recommendation(user_ratings)
print("Usuário {}: Recomendacoes de Livros com base em \"{}\" -> {}".format(user_id, book_title, user_recommendations))


image_urls = data[data['ISBN'].isin(user_recommendations)]['Image_URL'].tolist()
unique_values, counts = np.unique(image_urls, return_counts=True)



num_cols = 3
num_rows = (len(unique_values) + num_cols - 1) // num_cols

cols = st.columns(num_cols)


for i, url in enumerate(unique_values):
    col_index = i % num_cols
    row_index = i // num_cols
    with cols[col_index]:
        st.image(url, width=150)


