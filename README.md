# csci4930-group-project-denver-traffic
Group project for CSCI 4930: Machine Learning at University of Colorado.  This project analyzes traffic data from Denver County.

There is a bunch of code in `project.ipynb` for data cleaning and grid searches.  I was using the various code boxes as needed, it can be run sequentially but the grid searches will take several hours.

## Project Structure

- `project.ipynb`: Contains the primary data cleaning pipeline and extensive model grid searches.  This is the finalized version of the code.
- `traffic.ipynb`: Includes preliminary exploratory data analysis (EDA) and preliminary modeling specific to the Denver traffic data.  Sort of the rough draft.
- `evaluate.ipynb`: Used for loading trained models, generating model predictions, and evaluating performance metrics.  Contains the final ensemble model.
- `requirements.txt`: Lists the necessary Python packages and dependencies required to run the project environment.
- `data/`: Contains the raw open data CSV of Denver traffic accidents, alongside the cleaned and pre-processed version used for training.
- `pickle_jar/`: Stores serialized Python objects like computationally expensive trained models and grid search results to save time on subsequent runs.
- 