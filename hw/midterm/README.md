For my midterm I have used the dataset from https://www.kaggle.com/muhammetvarl/laptop-price to predict a laptops price.
  
The source code is organized as follows:
  1) The exploratory data analysis and model parameter analysis are located in `hw/midterm/notebook.ipynb`
  2) The creation of the model (and DictVectorizer) are located in `hw/midterm/train.py`
  3) All files relevant for setting up the Dockert container are located in `servers/midterm/`.
   
To run the project:
  1) Clone this repository and navigate to the `servers/midterm` directory.
  2) Build the container using the command `docker build -t laptop .`
  3) Run the container using the command `docker run -it  --rm -p 9696:9696 laptop:latest`
  4) In your browser, navigate to http://localhost:9696/ and fill out the form (NOTE: The form does not have any data validation: I'm relying on the user to put in valid numerical values for weight (in kg), CPU GHz, and Screen Inches).
  5) A simple JSON response is returned containing the predicted price (in Euros) of the theoretical laptop submitted.
