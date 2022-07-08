<img src="https://bit.ly/2VnXWr2" alt="Ironhack Logo" width="100"/>

# Project 9 - Shared Mobility - Application of model Prophet - Time series

*Edgar Tomé*
*[DA, Paris & 07/22]*

## Description 

Application of the model prophet to predict the quantity of used veichules by time, taking in consideration the location and the factores of holidays and wheather.

### Business Case
	
- Domain

	- Utilizations of sharing mobility services, by provider, city and type of vehicles.
	- Data scrap from sharing mobility services providers.
	- Provide the combine information of position and number utilizations of sharing mobility services.

- Problem

	- Missing data from the scrap process, during some periods of time.
	- Fill the missing data with values from time series model.
	- Find the best parameters to be used on model Prophet, to be more accurate with reality.
	- Experiment different parameters in different conditions for city and type of sharing .


In Module 3 the study was focused in machine learning. Time series forecasting is an important area of machine learning. Time series forecasting is a technique for predicting future events by analyzing past trends, based on the assumption that future trends will hold similar to historical trends. Forecasting involves using models fit on historical data to predict future values. Prediction problems that involve a time component require time series forecasting, which provides a data-driven approach to effective and efficient planning.

Began by understanding the information of all the data set. Separated the data in diferente data sets, by country and type of sharing service. 
Then started by cleaning and preparing the data to be ready for model fitting and application. Dealt with transform the Date/Time column that had the type object in to datetime64 type, fill he missing values of rain, number of veicules avaible and used veicules by near by values, has each row of the time was by hour.
Dint deleted outliers, that existed on the weather values.

With all of the necessary information and cleaned, prepared data then applied the model for time series prophet.

- Prophet (https://facebook.github.io/prophet/)
		Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

To recieve the best possible outcome of our model (also regarding the precious business in this case --> importanc of finding the best parameters to be applyed on the model, taking in cosideration the quantity of the data and the locations of the event.

Using the model with defautl hyperparameters and perform the test of root mean square error, to be compared with the results of the model with hyperparamters tunning. Then applyed the cross validation to get the best hyperparamters for the model for each data set, and perform the test root mean square error, to compare the models with default hyperparameters and hyperparametrs after tunning.

## Plan
- Create Trello workflow visualization 
- Create repo on GitHub
- Data cleaning and preparation in jupiter
- EDA of all data set
- Sampling data for each city, type of sharing service
- Aplication of model with default parameters
- Cross validation of the model
- Calculate the root mean square error, which is a metric that tells us the average distance between the predicted values from the model and the actual values in the dataset.
- Hyper parameter using cross valdiation
- Aplication of model with hyper parameters
- Calculate the root mean square error
- Comparaison of the diferent paramentes and results
- Prepare presentation

## Deliverables

- Original data in csv format
- Cleaning code
- Clean data in cvv format
- Slides for presentation

## Links to deliverables and additional links

[Repository](https://github.com/Edgart371/Project9)  
[Trello] (https://trello.com/invite/b/0Ew8qBuO/973b0645fe4ed0022d03f103d0a0b61d/prj9)




