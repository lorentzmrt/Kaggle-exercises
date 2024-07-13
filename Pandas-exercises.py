import pandas as pd

reviews = pd.read_csv("wine-reviews/winemag-data-130k-v2.csv")


# function to recenter the average score around 0
def remean_points(row):
    row.points = row.points - row.points.mean()
    return row

reviews.apply(remean_points, axis='columns')

# I'm an economical wine buyer. Which wine is the "best bargain"? 
# Create a variable bargain_wine with the title of the wine with the 
# highest points-to-price ratio in the dataset.

def find_bargain(reviews)
	return reviews.title.iloc[(reviews.points/reviews.price).idxmax()]

bargain_wine = find_bargain(reviews)

# There are only so many words you can use when describing a bottle of wine. 
# Is a wine more likely to be "tropical" or "fruity"? 
# Create a Series descriptor_counts counting how many times each of these two 
# words appears in the description column in the dataset. 
#(For simplicity, let's ignore the capitalized versions of these words.)'''

def count_descriptors(reviews, desc1, desc2):
    """
    Counts the number of descriptions containing specific descriptors.

    Parameters:
    reviews (pd.DataFrame): A DataFrame containing a 'description' column with text data.
    desc1 (str): The first descriptor string to search for.
    desc2 (str): The second descriptor string to search for.

    Returns:
    pd.Series: A Series with counts of the specified descriptors.
    """
    n_desc1           = reviews.description.map(lambda desc: desc1 in desc).sum()
    n_desc2           = reviews.description.map(lambda desc: desc2 in desc).sum()
    descriptor_counts = pd.Series([n_desc1, n_desc2], index=[desc1, desc2])
    
    return descriptor_counts

# Example usage (commented out)
# reviews = pd.DataFrame({'description': ["This wine has tropical flavors.",
# 										  "This wine is fruity.", 
#                                         "A wonderful blend."]})
# print(count_descriptors(reviews, "tropical", "fruity"))
print(  count_descriptors(reviews, "tropical", "fruity"))


# What is the best wine I can buy for a given amount of money? 
# Create a `Series` whose index is wine prices and whose values is the maximum number 
# of points a wine costing that much was given in a review. 
# Sort the values by price, ascending (so that `4.0` dollars is at the top and `3300.0` 
# dollars is at the bottom).


def best_value_for_given_index(df, index_col, value_col):
    """
    Creates a Series whose index is unique values from the specified index column and whose values 
    are the maximum values from the specified value column for each unique index. The Series is 
    sorted by the index in ascending order.

    Parameters:
    df (pd.DataFrame): A DataFrame containing the columns specified by index_col and value_col.
    index_col (str): The name of the column to be used as the index of the Series.
    value_col (str): The name of the column to be used as the values of the Series.

    Returns:
    pd.Series: A Series with the maximum value for each unique index, 
    sorted by the index in ascending order.
    """
    best_value_per_index = df.groupby(index_col)[value_col].max().sort_index()
    return best_value_per_index

best_rating_per_price = best_value_for_given_index(reviews, 'price', 'points')

# What are the minimum and maximum prices for each variety of wine? 
# Create a DataFrame whose index is the variety category from the dataset 
# and whose values are the min and max values thereof.

def value_extremes_per_category(df, category_col, value_col):
    """
    Creates a DataFrame whose index is unique values from the specified category column 
    and whose values are the minimum and maximum values from the specified value column 
    for each unique category.

    Parameters:
    df (pd.DataFrame):  A DataFrame containing the columns specified by category_col and value_col.
    category_col (str): The name of the column to be used as the category 
    				    for the index of the DataFrame.
    value_col (str):    The name of the column to be used as the values for the min and max
    				    aggregation.

    Returns:
    pd.DataFrame: A DataFrame with the min and max values for each unique category.
    """
    value_extremes = df.groupby(category_col)[value_col].agg(['min', 'max'])
    return value_extremes

# Example usage (commented out)
# df = pd.DataFrame({'variety': ['A', 'B', 'A', 'B', 'C'], 'price': [10, 20, 5, 25, 15]})
# print(value_extremes_per_category(df, 'variety', 'price'))

price_extremes = value_extremes_per_category(reviews, 'variety', 'price')

# What are the most expensive wine varieties? 
# Create a variable sorted_varieties containing a copy of the dataframe 
# from the previous question where varieties are sorted in descending order 
# based on minimum price, then on maximum price (to break ties).

sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending = False)

# Create a Series whose index is reviewers and whose values is the average review score
# given out by that reviewer. Hint: you will need the taster_name and points columns.

def average_rating_per_reviewer(df, index_col, value_col):
    """
    Creates a Series whose index is unique reviewers from the specified reviewer column 
    and whose values are the average review scores from the specified score column for 
    each unique reviewer.

    Parameters:
    df (pd.DataFrame): A DataFrame containing the columns specified by reviewer_col and score_col.
    reviewer_col (str): The name of the column to be used as the reviewer for the index of the Series.
    score_col (str): The name of the column to be used as the review scores for averaging.

    Returns:
    pd.Series: A Series with the average review score for each unique reviewer.
    """
    reviewer_mean_ratings = df.groupby(index_col)[value_col].mean()
    return reviewer_mean_ratings

# Example usage (commented out)
# df = pd.DataFrame({'taster_name': ['Alice', 'Bob', 'Alice', 'Bob', 'Charlie'], 'points': [90, 85, 88, 92, 80]})
# print(average_rating_per_reviewer(df, 'taster_name', 'points'))

reviewer_mean_ratings = average_rating_per_reviewer(reviews, 'taster_name', 'points')


# What combination of countries and varieties are most common? 
# Create a Series whose index is a MultiIndexof {country, variety} pairs. 
# For example, a pinot noir produced in the US should map to {"US", "Pinot Noir"}. 
# Sort the values in the Series in descending order based on wine count.

def count_combinations(df, col1, col2):
    """
    Creates a Series whose index is a MultiIndex of unique pairs from the specified columns
    and whose values are the counts of each combination, sorted in descending order.

    Parameters:
    df (pd.DataFrame): A DataFrame containing the columns specified by col1 and col2.
    col1 (str): The name of the first column to be used for the MultiIndex.
    col2 (str): The name of the second column to be used for the MultiIndex.

    Returns:
    pd.Series: A Series with counts of each combination of col1 and col2, sorted in descending order.
    """
    country_variety_counts = df.groupby([col1, col2]).size().sort_values(ascending=False)
    return country_variety_counts

# Example usage (commented out)
# df = pd.DataFrame({'country': ['US', 'France', 'US', 'France', 'Italy'], 
#                    'variety': ['Pinot Noir', 'Merlot', 'Pinot Noir', 'Chardonnay', 'Merlot']})
# print(count_combinations(df, 'country', 'variety'))


country_variety_counts = count_combinations(reviews, 'country', 'variety')

# rename columns
renamed = reviews.rename(columns={'region_1' : 'region', 'region_2':'locale'})

# rename axis
reindexed = reviews.rename_axis("wines", axis = 'rows')


#-#-#-#-#–#-#-#-#-#–#-#-#-#-#–#-#-#-#-#–#-#-#-#-#–#-#-#-#-#–#-#-#-#-#–#-#-#-#-#–#-#-#-#-#–#-#-#-#-#–

'''## 3.
The [Things on Reddit](https://www.kaggle.com/residentmario/things-on-reddit/data) dataset 
includes product links from a selection of top-ranked forums ("subreddits") on reddit.com. 
Run the cell below to load a dataframe of products mentioned on the */r/gaming* subreddit 
and another dataframe for products mentioned on the *r//movies* subreddit.'''
gaming_products = pd.read_csv("data/things-on-reddit/top-things/top-things/reddits/g/gaming.csv")
gaming_products['subreddit'] = "r/gaming"
movie_products = pd.read_csv("data/things-on-reddit/top-things/top-things/reddits/m/movies.csv")
movie_products['subreddit'] = "r/movies"

# Create a DataFrame of products mentioned on either subreddit.
combined_products = pd.concat([gaming_products, movie_products])
