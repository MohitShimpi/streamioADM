import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np
import base64
import datetime
from urllib.parse import urlencode
import json
import re
import sys
import itertools
from scipy.spatial import distance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
from skimage import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from math import sqrt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PIL import Image
import requests
from io import BytesIO


add_selectbox = st.sidebar.radio(
    "Select One",
    ("Introduction","RFM- Customer Segmentation","Churn Rate Analysis","Recommendation System")
)



if add_selectbox == 'Introduction':
    new_title = '**<p style="font-family:Georgia; color:Brown; bold: True;font-size: 33px;"> Movie Recommendation System & Market Analytics</p>**'
    st.markdown(new_title, unsafe_allow_html=True)
    st.image('cover.jpg')




    st.markdown(
        """
        Personalized Recommendation according to genres and user id, Customer Segmentation using RFM, and Churn Rate Information which will help to view information of customers all in one place. 
        """
    )
    
    st.markdown(
        """
                _Project by Team 9_
        """
    )
    
    
    
    
elif add_selectbox == 'RFM- Customer Segmentation':

    new_title = '**<p style="font-family:Georgia; color:Brown; text-align: center;font-size: 33px;">Customer Segmentation using RFM</p>**'
    st.markdown(new_title, unsafe_allow_html=True)
    def get_rfm():
        return pd.read_csv(
         'https://raw.githubusercontent.com/MohitShimpi/Team-9_INFO7374_Algorithmic_Digital_Marketing_/main/Final_Project/7.Streamlit/rfm_level_ag.csv')

    rfm = get_rfm()
   
    values = rfm['MonetaryValue']
    labels = rfm['Customer Segment']

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(height=500)
    st.plotly_chart(fig,height=500)

   
   
   
elif add_selectbox == 'Churn Rate Analysis':
    new_title = '**<p style="font-family:Georgia; color:Brown; text-align:center;bold: True;font-size: 33px;"> Churn Prediction</p>**'
    st.markdown(new_title, unsafe_allow_html=True)
    st.markdown(
        """
         Purchase Frequency (2019- 2021) : 106.04
        
         Repeat Rate : 0.45 
        
        Churn Rate : 0.54
        
        Since churn rate is more than repeate rate, company is loosing people than gaining people
        """
                )
    
    st.write('_______________________________________________________________________________________')

    def get_sales():
        return pd.read_csv('https://raw.githubusercontent.com/MohitShimpi/Team-9_INFO7374_Algorithmic_Digital_Marketing_/main/Final_Project/7.Streamlit/sale.csv')

    def get_rfm_segment():
        return pd.read_csv(
          'https://raw.githubusercontent.com/MohitShimpi/Team-9_INFO7374_Algorithmic_Digital_Marketing_/main/Final_Project/7.Streamlit/rfm_lsegment.csv')

    sales = get_sales()
    rfm2 = get_rfm_segment()

    id = st.selectbox("Select User Id : ",sales['user_id'])


    if id:
        count = -1
        for i in sales['user_id']:
            count = count + 1
            if id == i:
                new_title = '**<p style="font-family:Georgia; color:Blue; text-align:left;bold: True;font-size: 20px;"> Customer Lifetime Value- </p>**'
                st.markdown(new_title, unsafe_allow_html=True)
                new_title = sales['CLV'][count]
                st.markdown(new_title, unsafe_allow_html=True)
                new_title = rfm2['Customer Segment'][count]
                st.markdown(new_title, unsafe_allow_html=True)         
       
       
               
elif add_selectbox == 'Recommendation System':     
	
	def get_sales():
		return pd.read_csv(
		 'https://raw.githubusercontent.com/MohitShimpi/Team-9_INFO7374_Algorithmic_Digital_Marketing_/main/Final_Project/7.Streamlit/sale.csv')
	def get_movies():
		return pd.read_csv(
		 'https://raw.githubusercontent.com/MohitShimpi/Team-9_INFO7374_Algorithmic_Digital_Marketing_/main/Final_Project/7.Streamlit/movies.csv')
	# Reading ratings file
	ratings = pd.read_csv(
	 'https://raw.githubusercontent.com/MohitShimpi/Team-9_INFO7374_Algorithmic_Digital_Marketing_/main/Final_Project/7.Streamlit/ratings.csv'
	 , sep=',', encoding='latin-1', usecols=['userId','movieId','rating','timestamp'])

	# Reading movies file
	movies = pd.read_csv(
	 'https://raw.githubusercontent.com/MohitShimpi/Team-9_INFO7374_Algorithmic_Digital_Marketing_/main/Final_Project/7.Streamlit/movies.csv'
	  , sep=',', 
	encoding='latin-1',usecols=['movieId','title','genres'])

	df_movies = movies 
	df_ratings = ratings 

	merge_ratings_movies = pd.merge(df_movies, df_ratings, on='movieId', how='inner')

	merge_ratings_movies = merge_ratings_movies.drop('timestamp', axis=1)

	ratings_grouped_by_users = merge_ratings_movies.groupby('userId').agg([np.size, np.mean])

	ratings_grouped_by_users = ratings_grouped_by_users.drop('movieId', axis = 1)

	ratings_grouped_by_movies = merge_ratings_movies.groupby('movieId').agg([np.mean], np.size)

	ratings_grouped_by_movies = ratings_grouped_by_movies.drop('userId', axis=1)

	# Define a TF-IDF Vectorizer Object.
	tfidf_movies_genres = TfidfVectorizer(token_pattern = '[a-zA-Z0-9\-]+')

	#Replace NaN with an empty string
	df_movies['genres'] = df_movies['genres'].replace(to_replace="(no genres listed)", value="")

	#Construct the required TF-IDF matrix by fitting and transforming the data
	tfidf_movies_genres_matrix = tfidf_movies_genres.fit_transform(df_movies['genres'])
	cosine_sim_movies = linear_kernel(tfidf_movies_genres_matrix, tfidf_movies_genres_matrix)

	def get_recommendations_based_on_genres(movie_title, cosine_sim_movies=cosine_sim_movies):
	    """
	    Calculates top 2 movies to recommend based on given movie titles genres. 
	    :param movie_title: title of movie to be taken for base of recommendation
	    :param cosine_sim_movies: cosine similarity between movies 
	    :return: Titles of movies recommended to user
	    """
	    # Get the index of the movie that matches the title
	    idx_movie = df_movies.loc[df_movies['title'].isin([movie_title])]
	    idx_movie = idx_movie.index
	    
	    # Get the pairwsie similarity scores of all movies with that movie
	    sim_scores_movies = list(enumerate(cosine_sim_movies[idx_movie][0]))
	    
	    # Sort the movies based on the similarity scores
	    sim_scores_movies = sorted(sim_scores_movies, key=lambda x: x[1], reverse=True)

	    # Get the scores of the 10 most similar movies
	    sim_scores_movies = sim_scores_movies[1:3]
	    
	    # Get the movie indices
	    movie_indices = [i[0] for i in sim_scores_movies]
	    
	    # Return the top 2 most similar movies
	    return df_movies['title'].iloc[movie_indices]

	df_movies_ratings=pd.merge(df_movies, df_ratings)


	ratings_matrix_items = df_movies_ratings.pivot_table(index=['movieId'],columns=['userId'],values='rating').reset_index(drop=True)
	ratings_matrix_items.fillna( 0, inplace = True )

	def item_similarity(movieName): 
	    """
	    recomendates similar movies
	   :param data: name of the movie 
	   """
	    try:
	    	user_inp=movieName
	    	inp=df_movies[df_movies['title']==user_inp].index.tolist()
	    	inp=inp[0]
	    	df_movies['similarity'] = ratings_matrix_items.iloc[inp]
	    	df_movies.columns = ['movie_id', 'title', 'release_date','similarity']
				
	    except:
	    	print("Sorry, the movie is not in the database!")
		 


	def recommendedMoviesAsperItemSimilarity(user_id):
	    """
	     Recommending movie which user hasn't watched as per Item Similarity
	    :param user_id: user_id to whom movie needs to be recommended
	    :return: movieIds to user 
	    """
	    user_movie= df_movies_ratings[(df_movies_ratings.userId==user_id) & df_movies_ratings.rating.isin([5,4.5])][['title']]
	    user_movie=user_movie.iloc[0,0]
	    item_similarity(user_movie)
	    sorted_movies_as_per_userChoice=df_movies.sort_values( ["similarity"], ascending = False )
	    sorted_movies_as_per_userChoice=sorted_movies_as_per_userChoice[sorted_movies_as_per_userChoice['similarity'] >=0.45]['movie_id']
	    recommended_movies=list()
	    df_recommended_item=pd.DataFrame()
	    user2Movies= df_ratings[df_ratings['userId']== user_id]['movieId']
	    for movieId in sorted_movies_as_per_userChoice:
		    if movieId not in user2Movies:
		        df_new= df_ratings[(df_ratings.movieId==movieId)]
		        df_recommended_item=pd.concat([df_recommended_item,df_new])
		    best10=df_recommended_item.sort_values(["rating"], ascending = False )[1:10] 
	    return best10['movieId']
	    
	def movieIdToTitle(listMovieIDs):
	    """
	     Converting movieId to titles
	    :param user_id: List of movies
	    :return: movie titles
	    """
	    movie_titles= list()
	    for id in listMovieIDs:
	    	movie_titles.append(df_movies[df_movies['movie_id']==id]['title'])
	    return movie_titles

	new_title = '**<p style="font-family:Georgia; color:Brown; text-align:center;bold: True;font-size: 33px;"> Recomendation based on genres</p>**'
	st.markdown(new_title, unsafe_allow_html=True)

	movies = get_movies()
	name_list = st.selectbox("Select Movie : ",movies['title'])
        
	st.write(get_recommendations_based_on_genres(name_list))
	st.write('_______________________________________________________________________________________')
	new_title = '**<p style="font-family:Georgia; color:Brown; text-align:center;bold: True;font-size: 33px;"> Recomendation based on user</p>**'
	st.markdown(new_title, unsafe_allow_html=True)

	sales = get_sales()
	user_id = st.selectbox("Select UserID : ",sales['user_id'])
	st.write(movieIdToTitle(recommendedMoviesAsperItemSimilarity(user_id).unique()))

