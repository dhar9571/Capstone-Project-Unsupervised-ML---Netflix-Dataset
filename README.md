# Project Name - Netflix Movies and TV Shows Clustering

Link to the Live Server of my Netflix Cluster Prediction Web Application:

https://capstone-project-unsupervised-ml-netflix-dataset-dharmendra.streamlit.app/

# Business Use Case:
The business use case of project is to provide Netflix with valuable insights into its content library. These insights can inform content acquisition, production decisions, marketing strategies, and enhance the overall user experience along with creating a Recommender System for movies and TV Shows.

# Potential Impact:
The potential impact of project is significant. By understanding content trends, preferences, and clustering, Netflix can make data-driven decisions to improve its content offerings. This can lead to increased viewer satisfaction, higher engagement, and potentially reduced churn rates, positively impacting Netflix's bottom line.

# Approach:

Dataset Explanation: The dataset is Netflix movies and tv shows data which comprises of over 7500 rows and 12 features with no duplicate values and some null values in 5 of the features.

Feature Engineering: Data preprocessing involved creating separate datasets for both movies and TV shows for recommender system and NLP steps to convert content description into vectors such as TF-IDF. This step was crucial for preparing the data for machine learning modelling.

Algorithms: Applied clustering algorithms such as K-means, Hierarchical, DBSCAN to group content into clusters among which, K Means was working optimal with highest silhouette and lowest WCSS (Inertia). Used evaluation metrics like silhouette score and the elbow method to determine the optimal number of clusters.

Training: After training the K means model, acquired the cluster labels of all the movies and TV Show records using kmeans_cluster. After this, changed the clustering problem into classification problem and trained the XGBoost Classifier model with 92% accuracy considering the cluster label column as target feature.

End Output: The end output of the project is to develop a web application with trained classification model based on clusters which can also show meaningful insights of the particular cluster.

# Challenges Faced:
Missing values: The dataset contained a large number of missing values, particularly for the "cast" and "director" features. In order to handle this, I removed null value observations for the features having less than 5% of total length. In some features I replaced with UNKNOWN keyword due to incorrect information.

Deployment: Another challenge was to deploy the model as large files were not getting added on github respository. Therefore, proceed with uploading them using LFS (Large Files Storage) using command line.

# Future Scope: 
Training the model with more future data for better and vast prediction with improved insights. The insights from the analysis could be used to improve Netflix's services in a number of ways. For example, Netflix could use the insights to improve its content recommendations, personalize its marketing campaigns, and enhance its content duration.
