# README

## Project Title: **Movie Recommendation System using Content-Based Filtering**

### Description:
This project implements a movie recommendation system using content-based filtering. The system recommends movies based on the similarity of their content, including plot, genre, actors, and director. It utilizes TF-IDF (Term Frequency-Inverse Document Frequency) to extract features from the content and computes cosine similarity between movies to provide recommendations.

### Requirements:
- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `nltk`
  - `sklearn`
  
  You can install the required libraries using the following command:
  ```bash
  pip install pandas numpy nltk scikit-learn
  ```

### Dataset:
The dataset used for this project is `IMDB_Top250Engmovies2_OMDB_Detailed.csv`, which includes details of the top 250 English movies from IMDb. Each movie entry contains information like title, genre, plot, actors, and director. You can replace the dataset file with your own movie dataset if needed.

### Script Overview:
1. **Preprocessing the Data**:
   - The movie plots are cleaned by removing non-alphabetical characters and extra spaces.
   - Tokenization and removal of stopwords are applied to plot descriptions.
   - The 'Genre', 'Actors', and 'Director' columns are cleaned and tokenized.

2. **Feature Extraction**:
   - The `TF-IDF Vectorizer` is used to convert the text data (plot, genre, actors, director) into numerical features that can be used to compute similarity.
   
3. **Cosine Similarity**:
   - Cosine similarity is calculated between movies based on the features extracted from the text data.

4. **Movie Recommendation**:
   - A recommendation function is implemented to find the top 10 movies most similar to a given movie based on content similarity.

### How to Use:
1. Clone or download the repository.
2. Install the required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the `IMDB_Top250Engmovies2_OMDB_Detailed.csv` file in the same directory as the script (or modify the file path in the code).
4. Run the Python script to get movie recommendations based on the title.

### Recommendation Function:
- The main function to get recommendations is `recommend_movies(title)`.
- Input a movie title to get the top 10 most similar movies.
- Example usage:
  ```python
  print(recommend_movies('The Dark Knight Rises'))
  print(recommend_movies('The Shawshank Redemption'))
  print(recommend_movies('Spider-Man: Homecoming'))
  ```

### Output:
The function `recommend_movies(title)` returns a list of 10 movies that are most similar to the provided movie title based on the content features (plot, genre, actors, director). For example:
```python
recommend_movies('The Dark Knight Rises')
```
Output:
```python
['The Dark Knight', 'Inception', 'Batman Begins', 'The Prestige', 'Interstellar', ...]
```

### Features:
- **Content-Based Filtering**: The recommendations are based on the content of movies, including plot, genre, actors, and director.
- **TF-IDF and Cosine Similarity**: Uses TF-IDF for feature extraction and cosine similarity for computing the similarity between movies.
- **Tokenization and Stopwords Removal**: Cleaned and preprocessed movie descriptions for better accuracy.

### Notes:
- Make sure to download the `IMDB_Top250Engmovies2_OMDB_Detailed.csv` file and place it in the same directory as the script.
- The dataset should contain columns such as `Title`, `Plot`, `Genre`, `Actors`, and `Director`.

### License:
MIT License
