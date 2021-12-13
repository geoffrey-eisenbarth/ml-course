For my capstone project I have chosen the "Video Games Rating by 'ESRB'"
dataset on Kaggle, available here: 

  https://www.kaggle.com/imohtn/video-games-rating-by-esrb

According to the Entertainment Software Ratings Board (ESRB), "ESRB 
ratings provide information about what’s in a video game or app so 
parents can make informed choices about which are right for their 
family."

The dataset contains video games, along with their console and various
features that factor into an ESRB rating, such as presence of profanity,
animated blood, alcohol references, etc. The ESRB assigns one of the
following ratings to each game it reviews:

  * RP  (Rating Pending)
  * EC  (Early Childhood)
  * E   (Everyone)
  * ET  (Everyone 10+)
  * T   (Teen)
  * M   (Mature 17+)
  * A   (Adult Only 18+)

The goal of this model is to predict the ESRB rating provided the
various binary features listed below:

  * Console (either PS4 or PS4+XBoxOne)
  * Alcohol Reference
  * Animated Blood
  * Blood
  * Blood and Gore
  * Cartoon Violence
  * Crude Humor
  * Drug Reference
  * Fantasy Violence
  * Intense Violence
  * (Profane) Language
  * (Profane) Lyrics
  * Mature Humor
  * Mild Blood
  * Mild Cartoon Violence
  * Mild Fantasy Violence
  * Mild Language
  * Mild (Profane) Lyrics
  * Mild Suggestive Themes
  * Mild Violence
  * Nudity
  * Partial Nudity
  * Sexual Content
  * Sexual Themes
  * Simulated Gambling
  * Strong Language
  * Strong Sexual Content
  * Suggestive Themes
  * Use of Alcohol
  * Use of Drugs and Alcohol
  * Violence 

Therefore, this problem falls under the Multiclass Classification
umbrella. The dataset does not contain information for the RP, EC, and
A categories, so our model will be restricted to predicinng a rating of
E, ET, M, or T.

The dataset was already split into test and train sets by Kaggle, so we
use that split in our code instead of combining them and re-splitting.

The source code is organized as follows:

  1) The dataset is contained in the `data/capstone/` directory
  1) The exploratory data analysis and model parameter analysis are located in `hw/capstone/notebook.ipynb`
  2) The creation of the model is located in `hw/capstone/train.py`. This file loads and saves data using
     relative paths, so if you want to run it you should do:
      $ pipenv run ipython
      > from hw.capstone.train import *
  3) All files relevant for setting up the Docker container are located in `servers/capstone/`
   
To run the project:

  1) Clone this repository and navigate to the `servers/capstone` directory
  2) Build the container using the command `docker build -t esrb .`
  3) Run the container using the command `docker run -it  --rm -p 9696:9696 capstone:latest`
  4) In your browser, navigate to http://localhost:9696/ and fill out the form
  5) A simple JSON response is returned containing the predicted ESRB rating
