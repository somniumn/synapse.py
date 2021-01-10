import numpy as np
import pandas as pd
from pycomp.viz.insights import *
import os
from warnings import filterwarnings


filterwarnings('ignore')

# Project variables
# DATA_PATH = '../input/titanic'
DATA_PATH = ''
TRAIN_FILENAME = 'train.csv'
TEST_FILENAME = 'test.csv'

# Reading training data
df = pd.read_csv(os.path.join(DATA_PATH, TRAIN_FILENAME))
df.head()

# Survival rate
survived_map = {1: 'Survived', 0: 'Not Survived'}
survived_colors = ['crimson', 'darkslateblue']
# plot_donut_chart(df=df, col='Survived', label_names=survived_map, colors=survived_colors,
#                 title='Absolute Total and Percentual of Passengers \nwho Survived Titanic Disaster')

# Countplot for gender
gender_colors = ['lightskyblue', 'lightcoral']
gender_map = {'male': 'Male', 'female': 'Female'}
# plot_countplot(df=df, col='Sex', palette=gender_colors, label_names=gender_map,
#                title='Total Passengers by Its Gender')

# Survival rate by gender
# plot_countplot(df=df, col='Survived', hue='Sex', label_names=survived_map, palette=gender_colors,
#                title="Could gender had some influence on\nsurviving from Titanic shipwreck?")

# Plotting a double donut chart
# plot_double_donut_chart(df=df, col1='Survived', col2='Sex', label_names_col1=survived_map,
#                        colors1=['crimson', 'navy'], colors2=['lightcoral', 'lightskyblue'],
#                        title="Did the passenger's gender influence \nsurvival rate?")

# Number of passengers for each class
pclass_map = {1: 'Upper Class', 2: 'Middle Class', 3: 'Lower Class'}
# plot_pie_chart(df=df, col='Pclass', colors=['brown', 'gold', 'darkgrey'],
#                explode=(0.03, 0, 0), label_names=pclass_map,
#               title="Total Passengers by Social Class")

# Relação entre sobrevivência e classe social
# plot_countplot(df=df, col='Pclass', hue='Survived', label_names=pclass_map, palette=survived_colors,
#               title="Survival Analysis by Social Class")

# plot_pct_countplot(df=df, col='Pclass', hue='Survived', palette='rainbow_r',
#                   title='Social Class Influence on Survival Rate')

# Relationship between SibSp/Parch and Survived
# plot_countplot(df=df, col='SibSp', hue='Survived', orient='v', palette=survived_colors,
#               title='Survival Analysis by SibSp')

# Relationship between SibSp/Parch and Survived
# plot_countplot(df=df, col='Parch', hue='Survived', orient='v', palette=survived_colors,
#               title='Survival Analysis by Parch')

# Distribution of age variable
# plot_distplot(df=df, col='Age', title="Passenger's Age Distribution")

# plot_distplot(df=df, col='Age', hue='Survived', kind='kde',
#             title="Is there any relationship between age distribution\n from survivors and non survivors passengers?")

plot_distplot(df=df, col='Fare', title='Fare Distribution')

# Fare distribution by social class
plot_distplot(df=df, col='Fare', hue='Pclass', kind='strip', label_names=pclass_map,
              palette=['gold', 'silver', 'brown'],
              title="Fare Distribution by Social Class")

plot_distplot(df=df, col='Fare', hue='Sex', kind='boxen', palette=gender_colors,
              title="What's the Fare distribution by Gender?")

plot_distplot(df=df, col='Fare', hue='Survived', kind='kde', palette=survived_colors,
              title="What's the relationship between the \nfare paid and survivor?")

cat_cols = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
plot_multiple_countplots(df=df, col_list=cat_cols, orient='v')

plot_cat_aggreg_report(df=df, cat_col='Embarked', value_col='Fare', title3='Statistical Analysis',
                       desc_text=f'A statistical approach for Fare \nusing the data available',
                       stat_title_mean='Mean', stat_title_median='Median', stat_title_std='Std', inc_x_pos=10)

plt.show()
