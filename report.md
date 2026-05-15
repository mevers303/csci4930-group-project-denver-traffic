## Data cleaning
This was a relatively large dataset that strictly contains categorical data.  Some of the columns such as `precinct_id` and `neighborhood_id` initially appeared to be numerical, but really those were numerical codes that have no meaning in terms of magnitude.

It was necessary to clean some nonsense from the original dataset.  There were occasional cells that were just two digit numbers found in columns like `TU1_VEHICLE_TYPE`.  We replaced data like that with `None`.

The hardest decision was what to do with blank columns.  This data is all from police reports, so depending on what actually happened in the accident, many columns would be left blank.  For example, there were two sets of columns that would report on secondary and tertiary collisions, like when one vehicle rear ends a second vehicle, then the second vehicle hits a 3rd vehicle.  There were many reports that were just single vehicle accidents where many columns were left blank.  The most sparse rows came from hit and runs, likely from when someone walked out to their car parked on the street and found it damaged.

## Before One-hot Encoding
The columns before our one-hot encoding were named as follows:
1. 'offense_code'
1. 'district_id'
1. 'precinct_id'
1. 'neighborhood_id'
1. 'bicycle_ind'
1. 'pedestrian_ind'
1. 'HARMFUL_EVENT_SEQ_1'
1. 'HARMFUL_EVENT_SEQ_2'
1. 'HARMFUL_EVENT_SEQ_3'
1. 'road_location'
1. 'ROAD_DESCRIPTION'
1. 'ROAD_CONTOUR'
1. 'ROAD_CONDITION'
1. 'LIGHT_CONDITION'
1. 'TU1_VEHICLE_TYPE'
1. 'TU1_TRAVEL_DIRECTION'
1. 'TU1_VEHICLE_MOVEMENT'
1. 'TU1_DRIVER_ACTION'
1. 'TU1_DRIVER_HUMANCONTRIBFACTOR'
1. 'TU1_PEDESTRIAN_ACTION'
1. 'TU2_VEHICLE_TYPE'
1. 'TU2_TRAVEL_DIRECTION'
1. 'TU2_VEHICLE_MOVEMENT'
1. 'TU2_DRIVER_ACTION'
1. 'TU2_DRIVER_HUMANCONTRIBFACTOR'
1. 'TU2_PEDESTRIAN_ACTION'
1. 'hour'
1. 'day_of_week'
1. 'month'

These columns can be explored in `eda.ipynb`.  The `.value_counts()` for each category are present here.

## The target: `high_risk`
For our target `y` column, we decided to combine two columns called `SERIOUSLY_INJURED` or `FATALITIES`.  This would indicate the seriousness of the accident to indicate if people were hurt or if lives were lost.

## After One-Hot Encoding
After one-hot encoding the data exploded into a massive 976 columns.

## Class imbalance
A big obstacle with this dataset is the class imbalance.  We find that there are `275685` NOT high risk accidents with only `6559` rows that ARE high risk.

## Dimensionality reduction  (or not)
Because the one-hot encoded data has `976` features it would seem that reducing the dimensions of that dataset would be necessary.  Because it was all one-hot encoded, PCA was not an option.  We could have potentially done some called Single Value Decomposition (which was not covered in class).  But after some research, it seemed that the Decision Tree based models we were planning on using may actually have their performance hindered by dimensionality reduction.  So we made the decision to forego reduction and let the models do their thing.

## The search for a model
There were many types of models explored.  We expected the best performance from models based on Decision Trees due to the categorical data.  We did a grid search with the following models:
RandomForrest
ExtraTrees
LogisticRegression
LinearSVC
LightGBM (a model developed by Microsoft)

