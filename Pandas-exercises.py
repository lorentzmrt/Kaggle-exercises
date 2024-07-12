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
