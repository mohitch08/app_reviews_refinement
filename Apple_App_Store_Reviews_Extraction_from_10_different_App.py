#Installing Necessary Libraries
!pip install app_store_scraper pandas numpy json

#Importing Necessary Libraries
import pandas as pd
import numpy as np
import json
from app_store_scraper import AppStore


#Creating appstore objects of 10 APPS
WhatsApp=AppStore(country='in', app_name='whatsapp-messenger', app_id = '310633997')
Uber=AppStore(country='in', app_name='uber-request-a-ride', app_id = '368677368')
Spotify=AppStore(country='in', app_name='spotify-music-and-podcasts', app_id = '324684580')
Twitter=AppStore(country='in', app_name='x', app_id = '333903271')
YouTube=AppStore(country='in', app_name='youtube-watch-listen-stream', app_id = '544007664')
Netflix=AppStore(country='in', app_name='netflix', app_id = '363590051')
Candy_Crush_Saga=AppStore(country='in', app_name='candy-crush-saga', app_id = '553834731')
Amazon=AppStore(country='in', app_name='amazon-india-shop-pay-minitv', app_id = '1478350915')
Duolingo=AppStore(country='in', app_name='duolingo-languages-more', app_id = '570060128')
Google_Fit=AppStore(country='in', app_name='google-fit-activity-tracker', app_id = '1433864494')

#Extracting 1500 appstore reviews each for 10 APPS
WhatsApp.review(how_many=1500)
Uber.review(how_many=1500)
Spotify.review(how_many=1500)
Twitter.review(how_many=1500)
YouTube.review(how_many=1500)
Netflix.review(how_many=1500)
Candy_Crush_Saga.review(how_many=1500)
Amazon.review(how_many=1500)
Duolingo.review(how_many=1500)
Google_Fit.review(how_many=1500)

#Filtering data which is not required and creating final dataframe for each app

#Creating dataframe for Whatsapp reviews
whatsappdf = pd.DataFrame(np.array(WhatsApp.reviews),columns=['review'])
whatsappdf1 = whatsappdf.join(pd.DataFrame(whatsappdf.pop('review').tolist()))
whatsappdf1.drop(["date","isEdited","title","userName","developerResponse"],axis=1,inplace=True)
WhatsAPP=pd.DataFrame()
WhatsAPP["AppName"]=""
WhatsAPP=pd.concat([WhatsAPP, whatsappdf1], axis=1)
WhatsAPP["AppName"]="WhatsAPP"

#Creating dataframe for Uber reviews
uberappdf = pd.DataFrame(np.array(Uber.reviews),columns=['review'])
uberappdf1 = uberappdf.join(pd.DataFrame(uberappdf.pop('review').tolist()))
uberappdf1.drop(["date","isEdited","title","userName","developerResponse"],axis=1,inplace=True)
Uber=pd.DataFrame()
Uber["AppName"]=""
Uber=pd.concat([Uber, uberappdf1], axis=1)
Uber["AppName"]="Uber"

#Creating dataframe for Spotify reviews
spotifyappdf = pd.DataFrame(np.array(Spotify.reviews),columns=['review'])
spotifyappdf1 = spotifyappdf.join(pd.DataFrame(spotifyappdf.pop('review').tolist()))
spotifyappdf1.drop(["date","isEdited","title","userName"],axis=1,inplace=True)
Spotify=pd.DataFrame()
Spotify["AppName"]=""
Spotify=pd.concat([Spotify, spotifyappdf1], axis=1)
Spotify["AppName"]="Spotify"

#Creating dataframe for Twitter reviews
twitterappdf = pd.DataFrame(np.array(Twitter.reviews),columns=['review'])
twitterappdf1 = twitterappdf.join(pd.DataFrame(twitterappdf.pop('review').tolist()))
twitterappdf1.drop(["date","isEdited","title","userName"],axis=1,inplace=True)
Twitter=pd.DataFrame()
Twitter["AppName"]=""
Twitter=pd.concat([Twitter, twitterappdf1], axis=1)
Twitter["AppName"]="Twitter"

#Creating dataframe for Youtube reviews
youtubeappdf = pd.DataFrame(np.array(YouTube.reviews),columns=['review'])
youtubeappdf1 = youtubeappdf.join(pd.DataFrame(youtubeappdf.pop('review').tolist()))
youtubeappdf1.drop(["date","isEdited","title","userName"],axis=1,inplace=True)
YouTube=pd.DataFrame()
YouTube["AppName"]=""
YouTube=pd.concat([YouTube, youtubeappdf1], axis=1)
YouTube["AppName"]="YouTube"

#Creating dataframe for Netflix reviews
netflixappdf = pd.DataFrame(np.array(Netflix.reviews),columns=['review'])
netflixappdf1 = netflixappdf.join(pd.DataFrame(netflixappdf.pop('review').tolist()))
netflixappdf1.drop(["date","isEdited","title","userName"],axis=1,inplace=True)
Netflix=pd.DataFrame()
Netflix["AppName"]=""
Netflix=pd.concat([Netflix, netflixappdf1], axis=1)
Netflix["AppName"]="Netflix"

#Creating dataframe for Candy_Crush_Saga reviews
candycrushappdf = pd.DataFrame(np.array(Candy_Crush_Saga.reviews),columns=['review'])
candycrushappdf1 = candycrushappdf.join(pd.DataFrame(candycrushappdf.pop('review').tolist()))
candycrushappdf1.drop(["date","isEdited","title","userName"],axis=1,inplace=True)
Candy_Crush_Saga=pd.DataFrame()
Candy_Crush_Saga["AppName"]=""
Candy_Crush_Saga=pd.concat([Candy_Crush_Saga, candycrushappdf1], axis=1)
Candy_Crush_Saga["AppName"]="Candy_Crush_Saga"

#Creating dataframe for Amazon reviews
amazonappdf = pd.DataFrame(np.array(Amazon.reviews),columns=['review'])
amazonappdf1 = amazonappdf.join(pd.DataFrame(amazonappdf.pop('review').tolist()))
amazonappdf1.drop(["date","isEdited","title","userName"],axis=1,inplace=True)
Amazon=pd.DataFrame()
Amazon["AppName"]=""
Amazon=pd.concat([Amazon, amazonappdf1], axis=1)
Amazon["AppName"]="Amazon"


#Creating dataframe for Duolingo reviews
duolingoappdf = pd.DataFrame(np.array(Duolingo.reviews),columns=['review'])
duolingoappdf1 = duolingoappdf.join(pd.DataFrame(duolingoappdf.pop('review').tolist()))
duolingoappdf1.drop(["date","isEdited","title","userName"],axis=1,inplace=True)
Duolingo=pd.DataFrame()
Duolingo["AppName"]=""
Duolingo=pd.concat([Duolingo, duolingoappdf1], axis=1)
Duolingo["AppName"]="Duolingo"


#Creating dataframe for Google_Fit reviews
googlefitappdf = pd.DataFrame(np.array(Google_Fit.reviews),columns=['review'])
googlefitappdf1 = googlefitappdf.join(pd.DataFrame(googlefitappdf.pop('review').tolist()))
googlefitappdf1.drop(["date","isEdited","title","userName"],axis=1,inplace=True)
Google_Fit=pd.DataFrame()
Google_Fit["AppName"]=""
Google_Fit=pd.concat([Google_Fit, googlefitappdf1], axis=1)
Google_Fit["AppName"]="Google_Fit"


#function for calculating words in review
def word_count(sentence):
    return len(sentence.split())

#Filtering all reviews with length less than 10 from dataframes of all apps

WhatsApp= WhatsApp[WhatsApp['review'].apply(word_count) >= 10]
Uber= Uber[Uber['review'].apply(word_count) >= 10]1
Spotify= Spotify[Spotify['review'].apply(word_count) >= 10]2
Twitter= Twitter[Twitter['review'].apply(word_count) >= 10]
YouTube= YouTube[YouTube['review'].apply(word_count) >= 10]
Netflix= Netflix[Netflix['review'].apply(word_count) >= 10]
Candy_Crush_Saga= Candy_Crush_Saga[Candy_Crush_Saga['review'].apply(word_count) >= 10]
Amazon= Amazon[Amazon['review'].apply(word_count) >= 10]
Duolingo= Duolingo[Duolingo['review'].apply(word_count) >= 10]
Google_Fit= Google_Fit[Google_Fit['review'].apply(word_count) >= 10]


#Removing duplicates from dataframes of all apps if present
WhatsApp= WhatsApp.drop_duplicates()
Uber = Uber.drop_duplicates()
Spotify= Spotify.drop_duplicates()
Twitter= Twitter.drop_duplicates()
YouTube= YouTube.drop_duplicates()
Netflix= Netflix.drop_duplicates()
Candy_Crush_Saga= Candy_Crush_Saga.drop_duplicates()
Amazon= Amazon.drop_duplicates()
Duolingo= Duolingo.drop_duplicates()
Google_Fit= Google_Fit.drop_duplicates()


#Extracting Finally filtered 1000 reviews for each app 
WhatsAPP=WhatsAPP[:1000]
Uber=Uber[:1000]
Spotify=Spotify[:1000]
Twitter=Twitter[:1000]
YouTube=YouTube[:1000]
Netflix=Netflix[:1000]
Candy_Crush_Saga=Candy_Crush_Saga[:1000]
Amazon=Amazon[:1000]
Duolingo=Duolingo[:1000]
Google_Fit=Google_Fit[:1000]

#Combing all dataframes 
App_Storedf=pd.concat(([WhatsAPP,Uber,Spotify,Twitter,YouTube,Netflix,Candy_Crush_Saga,Amazon,Duolingo,Google_Fit], axis=0))

#Storing all extracted reviews 
App_Storedf.to_excel("App_Storedf.xlsx",index=False)