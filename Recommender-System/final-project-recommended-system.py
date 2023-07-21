#!/usr/bin/env python
# coding: utf-8

# # Sistem Rekomendasi Buku
# 
# Nama: I Gede Ari Wisnu Sanjaya
# 
# Asal: Bali

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Data UnderStanding
# 
# Memberikan informasi seperti jumlah data, kondisi data, dan informasi mengenai data yang digunakan.
# 
# Menuliskan tautan sumber data (link download). https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset
# 
# Menguraikan seluruh variabel atau fitur pada data.
# 
# Melakukan beberapa tahapan yang diperlukan mengenai data, contohnya teknik visualisasi data beserta insight atau exploratory data analysis.

# In[3]:


cd ../input/book-recommendation-dataset


# In[4]:


import pandas as pd
books = pd.read_csv('./Books.csv')
ratings = pd.read_csv('./Ratings.csv')
users = pd.read_csv('./Users.csv')

books.head()


# In[5]:


ratings.head()


# In[6]:


users.head()


# In[7]:


books.info()


# In[8]:


ratings.info()


# In[9]:


users.info()


# In[10]:


print("Shape Books Data : ", books.shape)
print("Shape Ratings Data : ", ratings.shape)
print("Shape Users Info Data : ", users.shape)


# In[11]:


import numpy as np
df_all = np.concatenate([
    ratings["User-ID"].unique(),
    users["User-ID"].unique()
])

df_all = np.sort(np.unique(df_all))

print("Jumlah seluruh datasets: ", len(df_all))


# In[12]:


# Merge

df = pd.merge(books, ratings, on='ISBN', how='left')

df = pd.merge(df, users, on='User-ID', how='left')
df.head()


# In[13]:


df.info()


# In[14]:


dtypes = pd.DataFrame(df.dtypes,columns=["Data Type"])

dtypes["Unique Values"]=df.nunique().sort_values(ascending=True)

dtypes["Null Values"]=df.isnull().sum()

dtypes["% null Values"]=df.isnull().sum()/len(df)

dtypes.sort_values(by="Null Values" , ascending=False).style.background_gradient(axis=0)


# ## Data Preparation
# 
# Menerapkan dan menyebutkan teknik data preparation yang dilakukan.
# 
# Teknik yang digunakan pada notebook dan laporan harus berurutan.
# 
# Menjelaskan proses data preparation yang dilakukan.
# 
# Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.
# 
# Handling null Value, ini perlu sebab data yang digunakan jauh dari kata bersih / clean

# In[15]:


df.isnull().sum()


# In[16]:


df = df.dropna()


# In[17]:


df.isnull().sum()


# In[18]:


df.info()


# In[19]:


df.shape


# Filter Column yang akan kita gunakan, ini perlu karena tidak semua kolom akan kita gunakan, hanya kolom yang dirasa akan dibutuhkan oleh requirement tim pengembang

# In[20]:


used_columns = ['ISBN','Book-Title','Book-Author','Publisher','Book-Rating','User-ID','Age',]
new_df = df[used_columns]


# In[21]:


new_df.isnull().sum()


# Banyak value yang unbalance, misal di bagian Publisher, ada 1 Publisher berisi 20.000 value sedangkan ada Publisher yang hanya berisi 1 value... maka dari itu akan kita proses Publisher yang value nya kurang dari 1000 akan kita drop atau hapus

# In[22]:


for column in new_df.columns:
 print("\n" + column)
 print(new_df[column].value_counts())


# ### Problem
# Data terlalu banyak jadi cosine smiliarity sangat memakan waktu lama jadi akan kita buat conditions ketika Publisher kurang dari 500 akan kita drop, lalu akan kita drop juga ISBN yang sama agar mengurangi data

# In[23]:


df_new = new_df.drop_duplicates(subset='ISBN')
df_new.shape


# In[24]:


df_new.shape


# In[25]:


counts = df_new['Publisher'].value_counts()
df = df_new[~df_new['Publisher'].isin(counts[counts < 50].index)]


# In[26]:


df.shape


# In[27]:


# counts = df['Book-Title'].value_counts()
# df = df[~df['Book-Title'].isin(counts[counts < 50].index)]


# ### EDA

# In[28]:


df.shape


# In[29]:


df.info()


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns

cat_features = ["Publisher","Book-Title","Book-Author",'Age']

for col in range(len(cat_features)):
    plt.figure()
    plt.xticks(rotation=90)
    plt.title(f'Count Plot Column {cat_features[col]}')
    sns.countplot(x = cat_features[col],data = df, order = df[cat_features[col]].value_counts().iloc[:20].index)


# label encoder untuk mengubah data kategorikal menjadi data number

# In[31]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['ISBN_le'] = le.fit_transform(df['ISBN'])
df['Publisher_le'] = le.fit_transform(df['Publisher'])
df['Book-Author_le'] = le.fit_transform(df['Book-Author'])
df['Book-Title_le'] = le.fit_transform(df['Book-Title'])


# In[32]:


# Сorrelation matrix
plt.figure(figsize=[15,10])
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[108]:


df.reset_index(drop=True)


# ## Modeling and Result
# 
# ### Consine Similiarity/Content Base Similiarity
# 
# Membuat dan menjelaskan sistem rekomendasi untuk menyelesaikan permasalahan
# 
# Menyajikan top-N recommendation sebagai output.
# 
# Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
# 
# Menjelaskan kelebihan dan kekurangan pada pendekatan yang dipilih.
# 
# Cosine similarity mengukur kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama. Ia menghitung sudut cosinus antara dua vektor. Semakin kecil sudut cosinus, semakin besar nilai cosine similarity.
# 
# Metrik ini sering digunakan untuk mengukur kesamaan dokumen dalam analisis teks. Sebagai contoh, dalam studi kasus ini, cosine similarity digunakan untuk mengukur kesamaan nama restoran dan nama masakan.
# 
# Maka dari itu saya gunakan Similarity untuk mencari kesamaan _Publisher_ dan _age_ untuk rekomendasi

# In[34]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# cv = CountVectorizer(lowercase=False)

# cosim_df = cv.fit_transform(df['Book-Author'])
# cosine_sim = cosine_similarity(cosim_df)


# In[107]:


df.isnull().sum()


# In[36]:


df = df.dropna()


# In[106]:


df.isnull().sum()


# In[104]:


def recommend_books_to_you(df_resm, publisher, age):
 try:

    age_publisher_data = df_resm[(df_resm['Age'] == age) & (df_resm['Publisher_le'] == publisher)]
    similar_books = df_resm.copy()

    book_prop = similar_books.loc[:, ['ISBN_le']]

    similar_books['Similarity with books'] = cosine_similarity(book_prop, book_prop.to_numpy()[age_publisher_data.index[0], None]).squeeze()

    similar_books.rename(columns={'Publisher_le': f'Similar with Publisher id {publisher}'}, inplace=True)
    similar_books['Book-Author_le'] = le.inverse_transform(similar_books['Book-Author_le'])
    similar_books['Book-Title_le'] = le.inverse_transform(similar_books['Book-Title_le'])
    similar_books = similar_books.sort_values(by='Similarity with books', ascending=False)
    similar_books = similar_books[['ISBN_le', f'Similar with Publisher id {publisher}', 'Book-Author_le', 'Age','Book-Title_le']]

    similar_books.reset_index(drop=True, inplace=True)

    return similar_books.iloc[1:7]
 except Exception as e:
    print('error: ', e)


# top recommended Books

# In[105]:


recommend_books_to_you(df,370,46)


# ## Colaborative Filtering
# Untuk Colaborative filtering yang akan saya gunakan adalah User-ID, Book-Title , Book-Rating, Book-Author, dan Age

# baca datasets

# In[44]:


import pandas as pd
books = pd.read_csv('./Books.csv')
ratings = pd.read_csv('./Ratings.csv')
users = pd.read_csv('./Users.csv')

books.head()


# In[45]:


ratings.head()


# In[46]:


users.head()


# merge semua data yang diperlukan

# In[63]:


df = pd.merge(books, ratings, on='ISBN', how='left')

df = pd.merge(df, users, on='User-ID', how='left')
df.head()


# handling null value

# In[64]:


df.isnull().sum()


# In[66]:


df = df.dropna()


# In[67]:


df.isnull().sum()


# ambil 100.000 sample

# In[68]:


df = df.sample(n=100000)


# In[69]:


df = df.reset_index(drop=True)


# pilih column yang diperlukan

# In[71]:


df = df[['Book-Rating', 'User-ID', 'Book-Title']]


# In[72]:


df


# Encode fitur User-ID dan Book-Title 

# In[73]:


user_ids = df['User-ID'].unique().tolist()
book_ids = df['Book-Title'].unique().tolist()


# Menyandikan (encode) fitur ‘User-ID’ dan ‘Book-Title’ ke dalam indeks integer.
# 
# Memetakan ‘User-ID’ dan ‘Book-Title’ ke dataframe yang berkaitan.
# 
# Mengecek beberapa hal dalam data seperti jumlah user, jumlah buku, kemudian mengubah Book-Rating menjadi float.

# In[74]:


# Melakukan encoding msno
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
 
# Melakukan proses encoding angka ke ke msno
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}

# Melakukan proses encoding song
book_to_book_encoded = {x: i for i, x in enumerate(book_ids)}
 
# Melakukan proses encoding angka ke song
book_encoded_to_book = {i: x for i, x in enumerate(book_ids)}

# Mapping user ke dataframe user
df['User-ID'] = df['User-ID'].map(user_to_user_encoded)
 
# Mapping song ke dataframe song
df['Book-Title'] = df['Book-Title'].map(book_to_book_encoded)

# Mengubah target menjadi nilai float
df['Book-Rating'] = df['Book-Rating'].values.astype(np.float32)

# Mendapatkan jumlah user
num_users = len(user_to_user_encoded)
print(num_users)
 
# Mendapatkan jumlah book
num_book = len(book_encoded_to_book)
print(num_book)


# In[92]:


# Mengacak Datasets
df = df.sample(frac=1, random_state=42)
df


# Split dengan rasio 8:2
# 
# 80.000 data train
# 
# 20.000 data val

# In[76]:


# Membuat variabel x untuk mencocokkan data user dan book menjadi satu value
x = df[['User-ID', 'Book-Title']].values
 
# Membuat variabel y untuk membuat rating dari hasil 
y = df['Book-Rating']
 
# Membagi menjadi 80% data train dan 20% data validasi
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)


# Membuat Recommender Net berikut example nya
# 
# https://keras.io/examples/structured_data/collaborative_filtering_movielens/

# In[89]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class RecommenderNet(tf.keras.Model):
 
  # Insialisasi fungsi
  def __init__(self, num_users, num_book, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_book = num_book
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
    self.book_embedding = layers.Embedding( # layer embeddings resto
        num_book,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.book_bias = layers.Embedding(num_book, 1) # layer embedding book bias
 
  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    book_vector = self.book_embedding(inputs[:, 1]) # memanggil layer embedding 3
    book_bias = self.book_bias(inputs[:, 1]) # memanggil layer embedding 4
 
    dot_user_book = tf.tensordot(user_vector, book_vector, 2) 
 
    x = dot_user_book + user_bias + book_bias
    
    return tf.nn.sigmoid(x) # activation sigmoid


# Compile model dengan loss RMSE

# In[ ]:


model = RecommenderNet(num_users, num_book, 50) # inisialisasi model
 
# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)


# In[93]:


# Memulai training
 
history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 8,
    epochs = 100,
    validation_data = (x_val, y_val)
)


# ## Evaluation
# Menyebutkan metrik evaluasi yang digunakan. RMSE
# 
# Menjelaskan hasil proyek berdasarkan metrik evaluasi.
# 
# Menjelaskan metrik evaluasi yang digunakan untuk mengukur kinerja model (formula dan cara metrik tersebut bekerja).
# 
# rmse = $\sqrt{\sum\nolimits_{n=1}^n \left((y_i - ŷ_i) ^ 2 \over n \right) }$
#  
# 
# Visualisasi Metrix untuk Melihat error dalam bentuk grafik

# In[109]:


import matplotlib.pyplot as plt

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[240]:


books = pd.read_csv('./Books.csv')
ratings = pd.read_csv('./Ratings.csv')
df_test =pd.merge(books, ratings, on='ISBN', how='left')
df_test = df_test[['Book-Rating','User-ID', 'Book-Title']]
df_test.head()


# In[241]:


user_id = df_test['User-ID'].sample(1).iloc[0]
book_read_by_user = df_test[df_test['User-ID'] == user_id]

book_read_by_user


# In[242]:


import pandas as pd
test_books = pd.read_csv('./Books.csv')
test_ratings = pd.read_csv('./Ratings.csv')
test_users = pd.read_csv('./Users.csv')

df = pd.merge(test_books, test_ratings, on='ISBN', how='left')

df = pd.merge(df, test_users, on='User-ID', how='left')
df.head()
df = df.sample(n=100000)
df = df.reset_index(drop=True)
df = df[['Book-Rating', 'ISBN', 'User-ID', 'Book-Title']]
df.isnull().sum()


# In[243]:


book_df = df


# In[244]:


book_df


# In[245]:


book_not_read = book_df[~book_df['Book-Title'].isin(book_read_by_user['Book-Title'].values)]['Book-Title'] 
book_not_read = list(
    set(book_not_read)
    .intersection(set(book_to_book_encoded.keys()))
)
len(book_not_read)


# In[246]:


book_not_read = [[book_to_book_encoded.get(x)] for x in book_not_read]

user_encoder = user_to_user_encoded.get(user_id)
user_book_array = np.hstack(
    ([[user_encoder]] * len(book_not_read), book_not_read)
)


# In[247]:


len(user_book_array) - len([x for x in user_book_array if x is not None])


# In[248]:


ratings = model.predict(user_book_array).flatten()
ratings


# In[287]:


top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_book_ids = [
    book_encoded_to_book.get(book_not_read[x][0]) for x in top_ratings_indices
]
 
print('Showing recommendations for users: {}'.format(user_id))
print('===' * 9)
print('book with high ratings from user')
print('----' * 8)
 
top_book_user = (
    book_read_by_user.sort_values(
        by = 'Book-Rating',
        ascending=False
    )
    .head(5)
)
print(top_book_user['Book-Title']) 
book_df_rows = book_df[book_df['ISBN'].isin(top_book_user)]
for row in book_df_rows.head(5).itertuples():
    print(row.name)
 
print('----' * 8)
print('Top 10 book recommendation')
print('----' * 8)
 
recommended_book = book_df[book_df['Book-Title'].isin(recommended_book_ids)]
for row in recommended_book.head(10).itertuples():
    print(row._4)

