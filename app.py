import streamlit as st
import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD
from joblib import load

# Load model
model = load('svd_model.pkl')

# Load data
ratings = pd.read_csv('Data/ratings.dat', delimiter='::', header=None, names=['userId', 'movieId', 'rating', 'Timestamp'],encoding='ISO-8859-1')
movies = pd.read_csv('Data/movies.dat', delimiter='::', header=None, names=['movieId', 'title' , 'categories'],encoding='ISO-8859-1' )

# Tạo một từ điển để lưu trữ các gợi ý phim
predicted_ratings_dict = {}
# Hàm gợi ý phim
def recommend_movies(model, user_id, ratings, movies, top_n=10):
    global predicted_ratings_dict
    # Kiểm tra nếu userID là None hoặc lớn hơn 6040, gợi ý 10 phim có rating trung bình cao nhất
    if user_id is None or user_id > 6040:
        # Tính toán rating trung bình cho mỗi phim
        avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
        avg_ratings = avg_ratings.sort_values(by='rating', ascending=False)[:top_n]
        
        
        # Merge với thông tin của phim
        top_movies = pd.merge(avg_ratings, movies, on='movieId')        
        
        # Hiển thị các phim được gợi ý
        st.header("Top 10 Recommended Movies")
        for i, row in top_movies.iterrows():
            st.write(f"<div style='border: 1px solid black; padding: 10px;'>{i+1}. {row['title']} (Average Rating: {row['rating']})</div>", unsafe_allow_html=True)
    
    # Ngược lại, gợi ý phim dựa trên user_id được cung cấp
    else:
        if user_id not in predicted_ratings_dict:
        # Tạo từ điển gợi ý phim
            predicted_ratings = {}

        # dự đoán rating của tất cả các movie mà người dùng chưa đánh giá
            for movie_id in ratings['movieId'].unique():
                if not ratings[(ratings['userId'] == user_id) & (ratings['movieId'] == movie_id)].empty:
                    continue
                predicted_rating = model.predict(user_id, movie_id).est
                predicted_ratings[movie_id] = predicted_rating
            predicted_ratings_dict[user_id] = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
        # Lấy ra từ điển gợi ý phim cho user_id từ từ điển global
        top_movies = predicted_ratings_dict[user_id]

        # hiển thị các phim được gợi ý 
        st.header("Top 10 Recommended Movies")
        for i, (movie_id, rating) in enumerate(top_movies):
            movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
            st.write(f"<div style='border: 1px solid black; padding: 10px;'>{i+1}. {movie_title} (Predicted Rating: {rating})</div>", unsafe_allow_html=True)


# Streamlit app
st.title("Movie Recommender System")

# nhập user id
user_id = st.number_input("Enter User ID:", min_value=1,value=None,   step=1)
# Kiểm tra xem user_id đã được nhập vào chưa
# Recommend movies for the given user ID
if st.button("Recommend"):
    recommend_movies(model, user_id, ratings, movies)
else:
    if user_id is  None:
        user_id = None
        recommend_movies(model, user_id, ratings, movies)



