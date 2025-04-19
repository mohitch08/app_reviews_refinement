# Installing Necessary Libraries
!pip install google-play-scraper

# Importing Necessary Libraries
from google_play_scraper import app
import pandas as pd
import numpy as np
from google_play_scraper import app, reviews_all

# Replace 'APP_PACKAGE' with the package name of the app you want to scrape
app_package = 'com.whatsapp'# Whatsapp Reviews Extraction
#app_package = 'com.ubercab' # Uber Reviews Extraction
#app_package = 'com.spotify.music' # Spotify Reviews Extraction
#app_package = 'com.twitter.android' # Twitter Reviews Extraction
#app_package = 'com.google.android.youtube' # Youtube Reviews Extraction
#app_package = 'com.netflix.mediaclient' # Netflix Reviews Extraction
#app_package = 'com.king.candycrushsaga' # Candy Crush Saga Reviews Extraction
#app_package = 'com.amazon.mShop.android.shopping' # Amazon Reviews Extraction
#app_package = 'com.duolingo' # Duolingo Reviews Extraction
#app_package = 'com.google.android.apps.fitness' # Google Fit Reviews Extraction

# Fetching the app details to obtain the app ID
app_details = app(app_package)
app_id = app_details['appId']

# Initialize variables
desired_reviews = 1000
reviews = []
filtered_reviews = []

# Function to get reviews that meet the criteria
def get_reviews():
    global reviews
    reviews = reviews_all(
        app_id,
        lang='en',  # Filter for English reviews
        count=desired_reviews*2,  # Fetch more than needed to filter out insufficient reviews
        sort=Sort.NEWEST # Sort by newest reviews
    )

    # Filter reviews with more than 10 words
    filtered_reviews.extend([
        review for review in reviews
        if review.get('content') and len(review['content'].split()) > 10
    ])

# Loop until we have enough reviews
while len(filtered_reviews) < desired_reviews:
    get_reviews()

# Take the first 1000 reviews that meet the criteria
filtered_reviews = filtered_reviews[:desired_reviews]

df_review = pd.DataFrame(np.array(filtered_reviews),columns=['review'])
df_review = df_review.join(pd.DataFrame(df_review.pop('review').tolist()))
#print(len(df_review))
#df_review.head()

# Extracting two columns and renaming them and adding the App name
Final_df = df_review[['content', 'score']].rename(columns={'content': 'App_Review', 'score': 'Rating'})
app_name = 'Whatsapp'  # Replace with the actual app name for which you are extracting the review
Final_df['App_Name'] = app_name

# Rearranging columns so that 'App Name' appears first
Final_df = Final_df[['App_Name', 'App_Review', 'Rating']]

# Save in the excel file
Final_df.to_excel('Whatsapp.xlsx', index=False)