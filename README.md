# CMPT353-AnimalCrossing

Follow the instruction bellow to run our code.

### Required libraries

Run the command to install the libraries if you don't have them
- BeautifulSoup:  `pip install beautifulsoup4`
- request: `pip install request`
- lxml: `pip install lxml`
- pandas: `pip install pandas`
- NumPy: `pip install numpy`
- Matplotlib: `pip install matplotlib`
- Seaborn: `pip install seaborn`
- sklearn: `pip install sklearn`
- scipy: `pip install scipy`
- scikit-posthocs: `pip install scikit-posthocs`
- statsmodels: `pip install statsmodels`
- csv, os, sys :all should have already been installed

**To install all libraries, run** 
"`pip install beautifulsoup4 request lxml pandas numpy matplotlib seaborn sklearn scipy scikit-posthocs statsmodels`"

### Running the code

Please run the code in the following order, just run `python3 *filename.py*`:

- `scraping_data.py`: This code scrape the villagers data and their ranking from the internet, 
then store the results as csv files in the "Data" folder.
- `cleaning_data.py`: This code clean the data and join the villager and the ranking data 
together. Result is saved in "Data" folder.
- `plots.py`: Produces the plots used in the report. Resulting plots are stored in 'Plots' folder.
- `analysis.py`: Do the statistical test, also produces some plots that are stored in the 'Plots' folder. 
p-values and other test results will be printed on the terminal when running.
- `prediction_model_V2.py`: run different prediction models to find the best one. Results will be printed on terminal upon running.
- `logit_anova.py`: run logistic regression model with the best parameter that we get from the code above, and performs ANOVA test to determinen feature significance. Produces result in terminal and plot of Top 10 significant features.
- `rf_feature_importance.py`(optional): Find the most importance features in random forest model.
