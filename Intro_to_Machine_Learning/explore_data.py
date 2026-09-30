import pandas as pd

# Path of the file to read
home= 'Intro_to_Machine_Learning/melb_data.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(home)

# Print summary statistics in next line
home_data.describe()


#=================#
#     LATIHAN     #
#=================#

# What is the average lot size (rounded to nearest integer)?
avg_lot_size = 10517

# As of today, how old is the newest home (current year - the date in which it was built)
newest_home_age = 16