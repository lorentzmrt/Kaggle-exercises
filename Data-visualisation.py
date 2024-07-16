import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
print("Setup Complete")


# LOAD DATA
# Path of the file to read
flight_filepath = "data/flight_delays.csv"  
# Read the file into a variable flight_data
flight_data = pd.read_csv(flight_filepath, index_col="Month")

##### BAR CHART #####
# Say we'd like to create a bar chart showing the average arrival delay for 
# Spirit Airlines (airline code: NK) flights, by month.

# Set the width and height of the figure
# Barplot_2.png 
plt.figure(figsize=(10,6))

# Add title
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=flight_data.index, y=flight_data['NK'])
# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")  
plt.show()

# Bar chart showing average arrival delay for Spirit Airlines flights by month
# Barplot_2.png 
plt.figure(figsize=(10,6))
plt.title("Average Arrival Delay for the month of May")
sns.barplot(x=flight_data.loc[5], y=flight_data.columns)
# Add label for vertical axis
plt.ylabel("Airlines")  
plt.xlabel("Arrival delay (in minutes)")  
plt.show()


##### HEAT MAP #####
# Set the width and height of the figure
# HeatMap.png 
plt.figure(figsize=(14,7))

# Add title
plt.title("Average Arrival Delay for Each Airline, by Month")

# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=flight_data, annot=True)

# Add label for horizontal axis
plt.xlabel("Airline")
plt.show()


###### SCATTER PLOTS ######    
# Path of the file to read
insurance_filepath = "data/insurance.csv"

# Read the file into a variable insurance_data
insurance_data = pd.read_csv(insurance_filepath)
insurance_data.head()

# ScatterPlot.png 
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
plt.show()

'''The scatterplot above suggests that body mass index (BMI) and insurance charges are positively correlated,
 where customers with higher BMI typically also tend to pay more in insurance costs. 
 (This pattern makes sense, since high BMI is typically associated with higher risk of chronic disease.)

To double-check the strength of this relationship, you might like to add a regression line, 
or the line that best fits the data. We do this by changing the command to sns.regplot.'''

# ScatterPlot-regression.png 
sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])
plt.show()

'''We can use scatter plots to display the relationships between (not two, but...) three variables! 
One way of doing this is by color-coding the points.

For instance, to understand how smoking affects the relationship between BMI and insurance costs, 
we can color-code the points by 'smoker', and plot the other two columns ('bmi', 'charges') on the axes.
'''
# ScatterPlot-colorcode.png 
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
plt.show()

'''This scatter plot shows that while nonsmokers to tend to pay slightly more with increasing BMI, 
smokers pay MUCH more.

To further emphasize this fact, we can use the sns.lmplot command to add two regression lines, 
corresponding to smokers and nonsmokers. 
(You'll notice that the regression line for smokers has a much steeper slope, 
relative to the line for nonsmokers!)
'''
# ScatterPlot-two-regression-lines.png 
sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)
plt.show()

'''The sns.lmplot command above works slightly differently than the commands you have learned about so far:

Instead of setting x=insurance_data['bmi'] to select the 'bmi' column in insurance_data, 
we set x="bmi" to specify the name of the column only.
Similarly, y="charges" and hue="smoker" also contain the names of columns.
We specify the dataset with data=insurance_data.'''

'''Finally, there's one more plot that you'll learn about, 
that might look slightly different from how you're used to seeing scatter plots. 
Usually, we use scatter plots to highlight the relationship between two continuous variables 
(like "bmi" and "charges"). 
However, we can adapt the design of the scatter plot to feature a categorical variable 
(like "smoker") on one of the main axes. 
We'll refer to this plot type as a categorical scatter plot, 
and we build it with the sns.swarmplot command.'''
sns.swarmplot(x=insurance_data['smoker'],
              y=insurance_data['charges'])
plt.show()


##### DISTRIBUTIONS #####
# Path of the file to read
iris_filepath = "data/iris.csv"

# Read the file into a variable iris_data
iris_data = pd.read_csv(iris_filepath, index_col="Id")

# Print the first 5 rows of the data
iris_data.head()


## HISTOGRAMS 
'''Say we would like to create a histogram to see how petal length varies in iris flowers.
We can do this with the sns.histplot command.'''
# Histogram.png
sns.histplot(iris_data['Petal Length (cm)'])
plt.show()


## DENSITY PLOT 
'''The next type of plot is a kernel density estimate (KDE) plot. 
In case you're not familiar with KDE plots, you can think of it as a smoothed histogram.

To make a KDE plot, we use the sns.kdeplot command. 
Setting shade=True colors the area below the curve 
(and data= chooses the column we would like to plot).''' 
# KDEplot.png
sns.kdeplot(data=iris_data['Petal Length (cm)'], fill=True)
plt.show()

## 2D KDE PLOTS
'''We're not restricted to a single column when creating a KDE plot. 
We can create a two-dimensional (2D) KDE plot with the sns.jointplot command.

In the plot below, the color-coding shows us how likely we are to see different 
combinations of sepal width and petal length, where darker parts of the figure are more likely.'''
# KDEplot2D.png
sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde") 
plt.show()

