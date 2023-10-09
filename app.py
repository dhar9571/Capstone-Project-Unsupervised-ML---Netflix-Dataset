import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import re
from nltk.stem.porter import PorterStemmer
import matplotlib as mpl
import sklearn

# Web Page title
st.title("Netflix Content Cluster Prediction")

# Importing required pickle files
xg = pickle.load(open('xggrid.pkl','rb'))
sc = pickle.load(open('sc.pkl','rb'))
df1 = pickle.load(open('df1.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))
sc = pickle.load(open('sc.pkl','rb'))
pca = pickle.load(open('pca.pkl','rb'))

# creating selection boxes for required variables
director = st.text_input("Enter Director Names separated by Comma:", "")
cast = st.text_input("Enter Cast Names separated by Comma:", "")
country = st.text_input("Enter Country of the Content Origin:", "")
rating = st.selectbox("Enter Rating:", [0,1,2,3,4,5,6,7,8,9,10,11,12,13])
listed_in = st.multiselect("Enter Genres separated by commas:", sorted(df1['listed_in'].unique()))

is_movie = 0
movie_duration=0
show_duration=0

type = st.selectbox("Select Content Type:",["Movie", "TV Show"])

if type=="Movie":
    is_movie=1
    movie_duration = st.number_input("Select Movie Duration:",step=1)
else:
    is_movie=0
    show_duration = st.selectbox("Select Total Seasons of TV Show:", range(1,17))

description = st.text_input("Enter Content Desciption:")

# defining user input dictionary:
user_dict = {'director':[director],'cast':[cast],'country':[country],'rating':[rating],
             'listed_in':[listed_in],'is_movie':[is_movie],'movie_duration':[movie_duration],
             'show_duration':[show_duration],'description':[description]}

# creating dataframe using user inputs
user_df = pd.DataFrame(user_dict)

# applying NLP with user input text values
user_df['text'] = str(director)+" "+str(cast)+" "+str(country)+" "+str(listed_in)+" "+str(description)


stop_words = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
    "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
    "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now"
]

corpus=[]
for i in range(len(user_df)):
    tags = re.sub('[^a-zA-Z]', ' ', user_df['text'][i])
    tags = tags.lower()
    tags = tags.split()
    ps = PorterStemmer()
    tags = [ps.stem(word) for word in tags if not word in set(stop_words)]
    tags = ' '.join(tags)
    corpus.append(tags)

# applying vectorization using tfidf
X = tfidf.transform(corpus).toarray()

user_df = pd.concat([user_df[['rating','is_movie','movie_duration','show_duration']],pd.DataFrame(X)],axis=1)

user_df["movie_duration"] = sc.transform(user_df['movie_duration'].values.reshape(-1,1))

# applying PCA
X_pca = pca.transform(user_df)

result = xg.predict(X_pca)

# creating button with HTML code for color change
button_style = """
        <style>
        div.stButton > button:first-child {
            background-color: green !important;
            color: black !important;
        }
        </style>
    """

st.markdown(button_style, unsafe_allow_html=True)

# creating if else condition when clicked on predict button, it will show the result of cluster with plots:
if st.button("Predict"):
    st.write("Cluster:", str(int(result)))

    st.title("Properties of the Cluster")

    st.subheader("Countries Distribution")
    cluster = df1[df1["kmeans_cluster"] == int(result)]

    # Create a countplot using Streamlit's native charting
    country_counts = cluster['country'].value_counts().head(5)

    font_size = 4

    mpl.rcParams['font.size'] = font_size

    plt.figure(figsize=(3, 2))
    ax = country_counts.plot(kind='bar')
    plt.xticks(rotation=45)

    # Display counts on the bars
    for i, count in enumerate(country_counts):
        ax.text(i, count + 1, str(count), ha='center', va='bottom')
        ax.set_ylabel("Count", fontsize=font_size)

    st.pyplot(plt)

    st.subheader("Ratings Distribution in Cluster 0")

    # Create a countplot using Streamlit's native charting
    ratings_counts = cluster['rating'].value_counts()

    font_size = 5

    mpl.rcParams['font.size'] = font_size

    plt.figure(figsize=(4, 3))
    ay = ratings_counts.plot(kind='bar')
    plt.xticks(rotation=45)

    # Display counts on the bars
    for i, count in enumerate(ratings_counts):
        ay.text(i, count + 1, str(count), ha='center', va='bottom')

    ay.set_xlabel("Ratings", fontsize=font_size)
    ay.set_ylabel("Count", fontsize=font_size)

    st.pyplot(plt)

    st.subheader("Movies vs TV Shows")

    # Create a countplot using Streamlit's native charting
    content_counts = cluster['is_movie'].value_counts()

    font_size = 5

    mpl.rcParams['font.size'] = font_size

    plt.figure(figsize=(4, 3))
    az = content_counts.plot(kind='bar')
    plt.xticks(ticks=[0, 1], labels=["TV Shows", "Movies"], rotation=0)

    # Display counts on the bars
    for i, count in enumerate(content_counts):
        az.text(i, count + 1, str(count), ha='center', va='bottom')

    az.set_ylabel("Count", fontsize=font_size)

    st.pyplot(plt)

    st.subheader("TV Show Seasons Distribution")

    # Create a countplot using Streamlit's native charting
    season_counts = cluster[cluster['show_duration']!=0]['show_duration'].value_counts().head(10)

    font_size = 5

    mpl.rcParams['font.size'] = font_size

    plt.figure(figsize=(4, 3))
    ai = season_counts.plot(kind='bar')

    # Display counts on the bars
    for i, count in enumerate(season_counts):
        ai.text(i, count + 1, str(count), ha='center', va='bottom')

    ai.set_xlabel("Seasons", fontsize=font_size)
    ai.set_ylabel("Count", fontsize=font_size)

    st.pyplot(plt)


    st.subheader("Movie Duration Distribution")

    # Create a histogram with KDE plot using Streamlit
    fig, aj = plt.subplots(figsize=(4, 3))
    sns.histplot(data=cluster, x='movie_duration', kde=True, ax=aj)
    aj.set_xlabel("Movie Duration")
    aj.set_ylabel("Frequency")

    # Display the plot in Streamlit
    st.pyplot(fig)

