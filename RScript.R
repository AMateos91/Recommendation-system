## Abraham Mateos
## MovieLens Rating Prediction Project 
## HarvardX: PH125.9x - Capstone Project

#################################################
# MovieLens Rating Prediction Project Code 
################################################

## Introduction
## - https://rafalab.github.io/dsbook/ - Chapter 34.7 - Recommendation systems
## - https://courses.edx.org/courses/course-v1:HarvardX+PH125.9x+2T2018

#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes in order to load the required packages:
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
#library(hexbin)
 
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# Checking downloads
datafile <- "MovieLens.RData"
if(!file.exists("MovieLens.RData"))
{
  print("Download")
  dl <- tempfile()
  download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
  
  ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                        col.names = c("userId", "movieId", "rating", "timestamp"))
  
  movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
  colnames(movies) <- c("movieId", "title", "genres")
  
  movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                             title = as.character(title),
                                             genres = as.character(genres))
  movielens <- left_join(ratings, movies, by = "movieId")
  
  # The Validation set will be 10% of the MovieLens data:
  
  set.seed(1, sample.kind = "Rounding")
  test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
  edx <- movielens[-test_index,]
  temp <- movielens[test_index,]
  
  #Make sure userId and movieId in validation set are included as well in edx set:
  validation <- temp %>%
    semi_join(edx, by = "movieId") %>%
    semi_join(edx, by = "userId")
  
  # Add rows removed from the validation set back into the edx set:
  removed <- anti_join(temp, validation)
  edx <- rbind(edx, removed)
  rm(dl, ratings, movies, test_index, temp, movielens, removed)
  save(edx, validation, file = datafile)
  
} else {
  load(datafile)
}

#### Methods and Analysis

### Data Analysis 

# Summarise Data
head(edx) %>%
  print.data.frame()
summary(edx)

# Number of unique movies, users and genres in the edx dataset 
edx %>% summarise(
  n_users = n_distinct(userId), 
  n_movies = n_distinct(movieId),
  n_genres = n_distinct(genres))

# Ratings distribution histogram
edx %>%
  ggplot(aes(rating)) +
  geom_bar(binwidth = 0.5, color = "black") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +	
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  xlab("Rating") +
  ylab("Count") +
  ggtitle("Rating distribution") +
  theme(plot.title = element_text(hjust = 0.5))

# Ratings of Movies - Number of Ratings
edx %>% 
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  xlab("Number of Ratings") +
  ylab("Number of Movies") +
  ggtitle("Number of Ratings per movie") +
  theme(plot.title = element_text(hjust = 0.5))

# Table 20 movies rated only once
edx %>%
  group_by(movieId) %>%
  summarise(count = n()) %>%
  filter(count == 1) %>%
  left_join(edx, by = "movieId") %>%
  group_by(title) %>%
  summarise(rating = rating, n_rating = count) %>%
  slice(1:20) %>%
  knitr::kable()

# Ratings of Users - Number of Ratings
edx %>% 
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  xlab("Number of Ratings") +
  ylab("Number of Users") +
  ggtitle("Users Number of Ratings") +
  theme(plot.title = element_text(hjust = 0.5))

# Ratings of Movies - Mean
edx %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarise(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) +
  geom_bar(bins = 30, color = "black") +
  xlab("Ratings mean") +
  ylab("Number of users") +
  ggtitle("Movie ratings mean per users") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  theme(plot.title = element_text(hjust = 0.5))

# Ratings of Users - Mean by Number with Curve Fitted
edx %>%
  group_by(userId) %>%
  summarise(mu_user = mean(rating), number = n()) %>%
  ggplot(aes(x = mu_user, y = number)) +
  geom_point( ) +
  scale_y_log10() +
  geom_smooth(method = lm) +
  ggtitle("Users Ratings Mean per Number of Rated Movies") +
  xlab("Ratings Mean") +
  ylab("Number of Rated Movies") +
  theme(plot.title = element_text(hjust = 0.5))

edx %>%
  group_by(userId) %>%
  summarise(mu_user = mean(rating), number = n()) %>%
  ggplot(aes(x = mu_user, y = number)) +
  geom_bin2d( ) +
  scale_fill_gradientn(colors = grey.colors(10)) +
  labs(fill="User Count") +
  scale_y_log10() +
  geom_smooth(method = lm) +
  ggtitle("Users Ratings Mean per Number of Rated Movies") +
  xlab("Ratings Mean") +
  ylab("Number of Rated Movies") +
  theme(plot.title = element_text(hjust = 0.5))


### Modelling Approach ###

## Average movie rating model ##

# References to the methods followed are below:
# From https://rafalab.github.io/dsbook/ 
# - 34.7 Recommendation systems
# - 34.9 Regularization

# Loss Function - RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Calculate the dataset's mean rating
mu <- mean(edx$rating)
mu

#Test results based on simple prediction
rmse_naive <- RMSE(validation$rating, mu)
rmse_naive

# Save prediction in data frame
rmse_results = data_frame(method = "Average movie rating model", RMSE = rmse_naive)
rmse_results%>% knitr::kable()

## Movie Effects Model
# Simple model taking into account the movie effects, b_i
# Plot number of movies with the computed b_i
# Test and save rmse results

movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"),
                     ylab = "Number of movies", main = "Number of movies with the computed b_i")

predicted_ratings <- mu + validation %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
rmse_model_movie_effects <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results, 
                          data_frame(method="Movie Effects Model",
                                     RMSE = rmse_model_movie_effects))
rmse_results %>% knitr::kable()

## Movie and user effect model
# Plot penaly term user effect 
#Test and save rmse results

user_avgs<- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarise(b_u = mean(rating - mu - b_i))
user_avgs%>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = I("black"))

user_avgs <- edx %>%
  left_join(movie_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse_model_user_effects <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results, 
                          data_frame(method="User Effects Model",
                                     RMSE = rmse_model_user_effects))
rmse_results %>% knitr::kable()

## Regularisation
# Regularized movie and user effect model 

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n() +l))
  
  b_u <- edx %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- validation %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))  
})
rmse_regularisation <- min(rmses)

# Plot RMSE vs Lambdas to find optimal lambda
qplot(lambdas, rmses)
lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results, 
                          data_frame(method="Regularisation",
                                     RMSE = rmse_regularisation))

rmse_results %>% knitr::kable()

## Appendix
print("Operating System info:")
version