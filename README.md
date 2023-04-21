<h1> SC1015 DSAI Project: YoutubeInsights </h1>

<h2> Our Motivation: </h2>

<li>
Youtube videos are a popular form of entertainment for people of all ages. However, it is close to or almost impossible for content creators to predict whether a particular video will go viral. 
</li>
<li>
YouTube has 2.1 billion monthly active users based all around the world. That number shows no signs of slowing down, with the projected amount of users increasing each year. In terms of daily active users, YouTube sees approximately 122 million users per day.
</li>
<li>
Hence, it is important for content creators to know and consider what are some factors that can make a video go viral and whether there are any trends that can be found in this data analysis. 
</li>

<h2> Project Goal: </h2>

<li>
This project aims to examine what are some factors that make a video go viral by seeing if the genre of video published, when the video is published affects viewership. 
</li>
<li>
This will help content creators maximize profits on youtube videos they produce by predicting which type of content will go viral before production.
</li>

<h2> Dataset Used: </h2>

We used one dataset from Kaggle and one dataset from Github Repository, cleaned and processed it for Exploratory Data Analysis and Machine Learning.
</li> https://www.kaggle.com/datasets/thedevastator/revealing-insights-from-youtube-video-and-channe</li>
</li>https://gitlab.com/thebrahminator/Youtube-View-Predictor/-/tree/master/dataset</li>

<h2> Jupyter Notebooks: </h2>

<li>
<a href = "https://github.com/DeuelT/SC1015_Mini_Project/blob/main/SC1015%20Mini-Project%20-%20Data%20Cleaning%20and%20Preprocessing.ipynb"> Data Cleaning and Preprocessing </a>
</li>
<li>
<a href = "https://github.com/DeuelT/SC1015_Mini_Project/blob/main/SC1015%20Mini-Project%20-%20EDA%20and%20Visualization.ipynb"> Exploratory Data Analysis & Visualization </a>
</li>
<li>
<a href = "https://github.com/DeuelT/SC1015_Mini_Project/blob/main/SC1015%20Mini-Project%20-%20Regressions.ipynb"> Regressions </a>
</li>
<li>
<a href = "https://github.com/DeuelT/SC1015_Mini_Project/blob/main/SC1015%20Mini-Project%20-%20Classification%20and%20Clustering.ipynb"> Classification/Clustering </a>
</li>
<li>
<a href = "https://github.com/DeuelT/SC1015_Mini_Project/blob/main/SC1015%20Mini-Project%20-%20Deep%20Learning%20%26%20Machine%20Learning.ipynb"> Deep Learning and Machine Learning </a>
</li>

<h2> Slide Deck: </h2>
Presentation Slides

<h2> Overview of DataScience Pipeline:</h2> 

<h3> 1. Data collection: </h3> 
<li>Used two different datasets</li>
<li>Merging of two datasets</li>
<h3> 2. Data cleaning and preprocessing:</h3> 
<li>Handling missing values by merging two different datasets </li>
<li>Creating ‘date difference’ time series data</li>
<li>Export data as csv</li>
<li>Minmax Scalar</li>
<h3> 3. EDA & Visualization:</h3> 
Explored, visualized, and generated insights for the following:
<li>Sum of ‘Title’ count to see Top Genre for no. of videos posted</li>
<li>‘channelViewCount’ sorted according to sum of ‘Title’ to see the top Genre for viewership</li>
<li>‘elapsedtime’ sorted according to sum of ‘Title’ to see the top Genre for viewership</li>
<h3> 4. Regression:</h3> 
Models:
<li>Linear Regression</li>
<li>Lasso Regression (Best)</li>
<li>Ridge Regression </li>
<li>Elastic Net Regression</li>
Metrics:
<li>Explained Variance (R^2)</li>
<li>Mean Squared Error </li>
<h3> 5. Classification/Clustering:</h3> 
Models:
<li>K-Means</li>
<li>KNN (K-Nearest-Neighbors</li>
Metrics:
<li>TPR, TNR, Confusion Matrix</li>
<li>Precision, Recall (TPR), F-score</li>
<li>Elbow Plot</li>
<li>Optimal K-Value</li>

<h2> 6. Key Insights & Recommendations:</h2>

<h3> Content Creators should: </h3>

<li>Focus creating videos within these genres: Music, Entertainment & Gaming</li>
<li>Focus on increasing subscribers by generating more views, more likes per view and more likes than dislikes</li>
<li>Focus on Video like count</li>
<li>Not focus on the number of videos that they upload by the quality of videos uploaded</li>
<li>Not focus on the number of comments on a video</li>
<li>Not focus on the total time spent watching their videos</li>

<h3> Important features that determine the success of a youtube video: </h3>

<li>‘subscriberCount’</li>
<li>‘channelViewCount’</li>
<li>like/view</li>
<li>‘likes/dislike’</li>
<li>‘Title’ - Genre</li>
<li>‘videolikeCount’</li>

<h2>What we learnt from this project:</h2>

<h3>Data cleaning and preprocessing:</h3>

<li>Merging Datasets together</li>
<li>Generating time-series data</li>
<li>MinMax Scalar (for transposing data points)</li>

<h3>EDA & Visualization:</h3>

Visualization plots with large number of datapoints</li>
By summing data using a specific variable - genre</li>
‘genres’ time-series EDA</li>

<h3>Machine Learning:</h3>
<h3>Machine Learning Models:</h3>

<li>Ridge Regression, Lasso Regression, Elastic Net</li>
<li>K-means, KNN (K-Nearest-Neighbour)</li>
<li>Classification </li>

<h3>Performance Metrics:</h3>

<li>F-score (Precision & Recall)</li>
<li>Elbow Plot</li>

<h2>Contributions:</h2>
<li>Data Collection: Aaron, Deuel, Russell </li>
<li>Data cleaning and preprocessing: Aaron, Deuel</li>
<li>EDA and visualization: Aaron, Deuel</li>
<li>Regression: Aaron, Deuel</li>
<li>Classification/Clustering: Aaron</li>
<li>Presentation Script: Aaron, Deuel, Russell</li>
<li>Presentation Voice Over + Editing: Aaron, Deuel, Russell</li>
<li>Slides Deck: Aaron, Deuel, Russell</li>
<li>GitHub ReadMe: Aaron, Deuel</li>

<h2> References:</h2>
<h3>Datasets</h3>
https://www.kaggle.com/datasets/thedevastator/revealing-insights-from-youtube-video-and-channe
https://gitlab.com/thebrahminator/Youtube-View-Predictor/-/blob/master/datasets/categoryID.csv 

<h2> Libraries <h2>
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html 
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

<h2> Machine Learning</h2>
<h3>General</h3>
https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501 
<h3>K-Means</h3>
https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/#What_Is_K-Means_Clustering?
<h3>K-Nearest-Neighbors (KNN)</h3>
https://realpython.com/knn-python/ 
<h3>Lasso Regression</h3>
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html 
