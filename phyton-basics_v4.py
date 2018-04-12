
'''
# Python is and object-oriented programming language
# This object-oriented programming model provides a number of classes 
# with objects being instances of the class.

'''



# Settings
import pandas as pd



# Email


import smtplib
from email.mime.multipart import MIMEMultipart


def email_end(_subject ):
	_me = 'alejandro.cruzmarcelo@aexp.com'
	
	# Create the container (outer) email message.
	msg = MIMEMultipart()
	msg['Subject'] = '_PT ' + _subject
	# me == the sender's email address
	msg['From'] = _me
	msg['To'] = _me

	s = smtplib.SMTP('localhost')
	s.sendmail(_me, [_me], msg.as_string())
	s.quit()


email_end(_subject = 'Python Test' )



# Create log to track progress
f = open('cnn.log', 'a')
print('Start reading raw data ...')
f.write('Start reading raw data ...\n')
print('====================================================================================================')
f.write('====================================================================================================\n')
f.close() 


# Python classes

# Python for SAS users
http://nbviewer.jupyter.org/github/RandyBetancourt/PythonForSASUsers/tree/master/

# Resources
http://www.scipy-lectures.org/intro/numpy/array_object.html


# Object references
# b_list = a_list
# Point to the same memory location. Different names for the same object 

# Zen of pythone (execute and get the text)
import this
import codecs

print(codecs.decode(this.s, 'rot-13'))

# Line continuation symbol: \
numbers = [2, 4, 6, 8, 11, 13, 21, \
           17, 31]

# Object names are case sensitivity
		   

# Find type of an object
type()

		   
# import multiple objects form a module
from demo import print_a, print_b



# find installed packages  
import pip
installed_packages = pip.get_installed_distributions()
installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
     for i in installed_packages])
print(installed_packages_list)

# find installed modules
help("modules")


# Function definition and return
def miles_to_feet(miles):
    feet = 5280*miles
    return feet

# Call a function
def test(miles):
    print str(miles) + " miles equals",
    print str(miles_to_feet(miles)) + " feet."

test(13)

# Remove space when printing by using comma
def powerball():
    """Prints Powerball lottery numbers."""
    print "Today's numbers are " + str(random.randrange(1, 60)) + ",",
    print str(random.randrange(1, 60)) + ",",
    print str(random.randrange(1, 60)) + ",",
    print str(random.randrange(1, 60)) + ", and",
    print str(random.randrange(1, 60)) + ". The Powerball number is",
    print str(random.randrange(1, 36)) + "."

# Print single quotation inside a comment
print "I'd much rather you 'not'."
# Print double quotation inside a comment
print 'I "said" do not touch this.'	
	
# Convert strings
# Concatenate strings
# Print strings without space
print str(miles) + " miles equals",

# Select substring and apply rule to each member in the series 
d_1['DUM_Jan'] = 0 
d_1.DUM_Jan = 1*(d_1.as_of_dt.str.slice(start = 2, stop = 5, step = 1) == 'JAN')


# Print using multiple variable names with percentage sign
my_name = 'Zed A. Shaw'
my_age = 35 # not a lie
print "Let's talk about %s and %d." % (my_name, my_age)

print "Its fleece was white as %s." % 'snow'

formatter = "%r %r %r %r"
print formatter % (1, 2, 3, 4)

# Print with line breaks
months = "Jan\nFeb\nMar\nApr\nMay\nJun\nJul\nAug"
print "Here are the months: ", months
print ("Here are the months: ", months )

# Print  line spaces with three double quotes
print """
There's something going on here.
With the three double-quotes.
We'll be able to type as much as we like.
Even 4 lines if we want, or 5, or 6.
"""

# print special characters using backslash
backslash_cat = "I'm \\ a \\ cat."
print backslash_cat

# Print with parenthesis 
print('%s already present - Skipping extraction of %s.' % (root, filename))




# Boolean values, capital letter at the beginning
# Logic conditionals: not, and, or
a = True
b = False
print not a
print a and b
print a or b
print (a and b) or (b or a)

# Comparison operators
# Returns True or False
>
<
>=
<=
==
!=
a = 7 > 3
print a
c = "Hello" == "Hello"
print c


# Load libraries
import random
import matplotlib.pyplot as plt
import numpy as np


# if else condition
def favorites(instructor):
    """Return the favorite thing of the given instructor."""
    if instructor == "Joe":
    	return "games"
    elif instructor == "Scott":
    	return "ties"
    elif instructor == "John":
    	return "outdoors"
	else:
		print "Invalid instructor: " , instructor
		
print favorites("John")
print favorites("Jeannie")



# Nested if conditions 
# Smaller quadratic root formula
# Student should enter function on the next lines.
# Comments inside functions
def smaller_root(a, b, c):
    """
    Returns the smaller root of a quadratic equation with the
    given coefficients.
    """
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0 or a == 0:
        print "Error: No real solutions"
    else:
        discriminant_sqrt = discriminant ** 0.5
        # Choose the positive or negative square root that leads to a smaller root.
        if a > 0:
            smaller = - discriminant_sqrt
        else:
            smaller = discriminant_sqrt
        return (-b + smaller) / (2 * a)

# Accesing global variables inside functions
num = 4
def fun():
	global num
	num = 6

# Place holder for functions,  empty block : pass
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    pass  # TODO: Compute and return softmax(x)
	
	
# Find attributes of objects 
dir(x)



# Numpy arrays
# Define arrays 
import numpy as np
>>> d1 = np.array([2.4, -1.5, 3.0, 8.8])
>>> d2 = np.array([(0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11)])
>>> d3 = np.array(
...  [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
...   [[12,13,14,15], [16,17,18,19], [20,21,22,23]]])

>>> print d1
[ 2.4 -1.5  3.   8.8]
>>> print d1.ndim, d1.shape, d1.dtype
1 (4,) float64
>>> print d2
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
>>> print d2.ndim, d2.shape, d2.dtype
2 (3, 4) int32
>>> print d3
[[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]]

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]
>>> print d3.ndim, d3.shape, d3.dtype
3 (2, 3, 4) int32



# Create an array using a low-level method. Array is a way to call ndarray. Not recommended
dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)

# Populating a three dimensional array
dataset[image_index, :, :] = image_data
valid_dataset[start_v:end_v, :, :] = valid_letter

# Select multiple element of an array, randomize the output
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]

							
# Accessing different elements of an array
y[0]    #first array
y[0][0] #first row of first array

							

# transpose array 
np.transpose(x)

# Numpy arrange 
x = np.arange(-2.0, 6.0, 2)  # start, end, step. 

x = np.arange(4).reshape((2,2))
>>> x
array([[0, 1],
       [2, 3]])

# Stack multiple vectors to form an array
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

# Create diagonal array 
a = np.diag(np.arange(3))

# Find number of columns or number of dimensions in the array
nparray.ndim

# Find size for each dimensions
nparray.shape

# Get information on a file 
np.info()

# Data types in python
bool: Boolean (true/false) types. Supported precisions: 8 (default) bits.
int: Signed integer types. Supported precisions: 8, 16, 32 (default) and 64 bits.
uint: Unsigned integer types. Supported precisions: 8, 16, 32 (default) and 64 bits.
float: Floating point types. Supported precisions: 16, 32, 64 (default) bits and extended precision floating point (see note on floating point types).
complex: Complex number types. Supported precisions: 64 (32+32), 128 (64+64, default) bits and extended precision complex (see note on floating point types).
string: Raw string types. Supported precisions: 8-bit positive multiples.
time: Data/time types. Supported precisions: 32 and 64 (default) bits.
enum: Enumerated types. Precision depends on base type.



# Data types
pdt = pd.Timestamp('2016-10-24')
# pandas.tslib.Timestamp


# Create dates
date(year = yyyy, month = mm, day = dd)
ind_day = date(1776, 7, 4)
print(ind_day)

# Find today's date
date.today()

# Get year month day from a date
date.today().year
date.today().month
date.today().day

# Visualize the datam top and last 10 observations
np.head(10)
np.tail(10)

# Use informative names with the columns 
prt = loans[['iz_all', 'iz_grp']]
prt.columns = ['zscore w/ overall mean','zscore with group mean']
prt.sort_values('zscore w/ overall mean', ascending=False).head(10)


# Print the largest and smallest for overall and by group
grp_grd['income'].nlargest(3)
grp_grd['income'].nsmallest(3)



# Convert jupyter notebook into a py file that can be submitted

# Indexers examples
# 1. .iloc() method which is mainly an integer-based method
# 2. .loc() method used to select ranges by labels (either column or row)
# 3. .ix() method which supports a combination of the loc() and iloc() methods


# First row
df.iloc[0]

# last row
df.iloc[-1]

# Range 
# df.iloc[row selection, column selection]
df.iloc[2:4, 0:6]


# show index
df2.index

# Set index 
df2.set_index('id', inplace=True, drop=False)

# Change index, inplace=TRUE -> no copy is made ;
df2.set_index('date', inplace=True)

# Requet a row by label
df2.loc['05/31/16']

# Reset index to the default 
df2.reset_index(inplace=True)

# Check whether the index is increasing monotonically
df.index.is_monotonic_increasing


# Slicing using loc subsetting 
df2.loc['d':'f',['col6','col2']]

# Slicing subsetting based on a condition 
df2.loc[(df2.col3 >=  9) & (df2.col1 == 'cool'), ]
df2.loc[df2.col6.isin([6, 9, 13])]

# Modifying objects based on condition 
df2.loc[df2['col6'] > 50, "col2"] = "VERY FAST"\
df2.loc[: , 'col2']

# ix can combine labels and indexes
df4.ix['b':'e', 6:8] 



# For loop, use the counters

for num in range(10,20):  #to iterate between 10 to 20, not including 20 and starting in 10
   for i in range(2,num): #to iterate on the factors of the number
      if num%i == 0:      #to determine the first factor
         j=num/i          #to calculate the second factor
         print '%d equals %d * %d' % (num,i,j)
         break #to move to the next number, the #first FOR
   else:                  # else part of the loop
      print num, 'is a prime number'


# loop over the components of a list 
change = [1, 'pennies', 2, 'dimes', 3, 'quarters']
for i in change:
    print "I got %r" % i	  

# Maximum and sum by rows and coluns in array
x = arr[7].max()   # Maximum in row 7
y = arr[29].sum()  # Sum of the values in row 29
z = arr[:, 5].sum()  # Sum up all values in column 5.

# loop using the while condition 
i = 0
numbers = []

while i < 6:
    print "At the top i is %d" % i
    numbers.append(i)

    i = i + 1
    print "Numbers now: ", numbers
    print "At the bottom i is %d" % i

# Create a list
list('abc') returns ['a', 'b', 'c'] 
 list( (1, 2, 3) ) returns [1, 2, 3]
	
	
# Create a list by appending elements

elements = []
# then use the range function to do 0 to 5 counts
for i in range(0, 6):
    print "Adding %d to the list." % i
    # append is a function that lists understand
    elements.append(i)


# Operations with arrays
 a = np.arange(5)
 np.sin(a)
 np.log(a)
 np.exp(a)
 2.0*a
 a/(2.0*a)
 
 
# append vectors 

>>> np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
array([1, 2, 3, 4, 5, 6, 7, 8, 9])

>>> np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
	   
>>> a = np.array((1,2,3))
>>> b = np.array((2,3,4))
>>> np.hstack((a,b))
array([1, 2, 3, 2, 3, 4])	   
 
 
 # enumerate elements in a list, it returns the index and the ith element in the loop
 for i, item in enumerate(L):
    # ... compute some result based on item ...
    L[i] = result
 
# Find/ Search one  member in a list
item = 'ale' 
if (item in a_list):
    print('found')
else:
    print('not found')

# Find/ Search multiple  member in a list
item1 = 'eggs' 
item2 = 'cupcakes'
if (item1 and item2 in more_dishes):
    print('found')
else:
    print('not found')	

# Defining object as a set 
months1 = set(['January', 'February', 'March', 'April', 'May', 'June'])	
		
	# Copy as an attribute ;	
	months2 = months1.copy()

	# Remove as an attribute
	months1.remove('February')

	# Add as an attribute
	months2.add('July')
	
	# Intersection
	months1 & months2

	# Test if the set months2 is a super-set of the set months1	
	months2.issuperset(months1)


# Dictionary, Dictionaries  define and add elements
# Get content using the keys
>>> stuff = {'name': 'Zed', 'age': 39, 'height': 6 * 12 + 2}
>>> print stuff['name']
Zed
>>> print stuff['age']
39
>>> print stuff['height']
74

>>> stuff['city'] = "San Francisco"
>>> print stuff['city']
San Francisco

>>> stuff[1] = "Wow"
>>> stuff[2] = "Neato"
>>> print stuff[1]
Wow
>>> print stuff[2]
Neato
>>> stuff
{'city': 'San Francisco', 2: 'Neato', 'name': 'Zed', 1: 'Wow', 'age': 39, 'height': 74}

# Delete elements from a list>>> del stuff['city']
>>> del stuff[1]
>>> del stuff[2]
>>> stuff
{'name': 'Zed', 'age': 39, 'height': 74}

# Extracting items from dictionary 
# Loop with objects including more than one element, 2-tuples
cities = {
    'CA': 'San Francisco',
    'MI': 'Detroit',
    'FL': 'Jacksonville'
}

	
for abbrev, city in cities.items():
    print "%s has the city %s" % (abbrev, city)

	
# Get an element from a dictionary
# It return null if the element is not present
states = {
    'Oregon': 'OR',
    'Florida': 'FL',
    'California': 'CA',
    'New York': 'NY',
    'Michigan': 'MI'
}

state = states.get('Texas')
	

# Class definition and example
class Song(object):

    def __init__(self, lyrics):
        self.lyrics = lyrics

    def sing_me_a_song(self):
        for line in self.lyrics:
            print line

happy_bday = Song(["Happy birthday to you",
                   "I don't want to get sued",
                   "So I'll stop right there"])

bulls_on_parade = Song(["They rally around tha family",
                        "With pockets full of shells"])

happy_bday.sing_me_a_song()

bulls_on_parade.sing_me_a_song()

# Getting information comparing dictionary, classes, and modules

# dict style
mystuff['apples']

# module style
mystuff.apples()
print mystuff.tangerine

# class style
thing = MyStuff()
thing.apples()
print thing.tangerine


# The from...import Statement  - import specific attributes from a module into the current namespace. 
from fib import fibonacci


# Delete Clear variables in a python session 
%reset
%reset -f  # it removes the prompt of yes or no

# select random sample a component in a given array
sample = np.random.choice(os.listdir(i), 1)[0] # this randomly selects a .png file in the given folder


# Add subplots in a grid
# It specifies rows, columns, and number in the grid
ax = fig.add_subplot(2, 5, n + 1) # there are 10 subplots (A-J), a (rows) by b(columns) by c(figure number)


# Return a list containing the names of the entries in the directory given by path. 
image_files = os.listdir(folder)

# Join one or more path components intelligently. 
os.path.join(folder, image)


# Use of raise for errors 
if image_data.shape != (image_size, image_size):
	raise Exception('Unexpected image shape: %s' % str(image_data.shape))
		
# try except and else use for exceptions
for arg in sys.argv[1:]:
    try:
        f = open(arg, 'r')
    except IOError:
        print 'cannot open', arg
    else:
        print arg, 'has', len(f.readlines()), 'lines'
        f.close()

# Picklyng a file 
try:
	with open(set_filename, 'wb') as f:
		pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
	except Exception as e:
		print('Unable to save data to', set_filename, ':', e)


# Find python version 		
	
	
# Comment out multiple lines 
'''

print("We are in a comment")

print ("We are still in a comment")

'''


		
# Making an array read only  
'''a and b are ndarrays of shape (len, 28, 28)'''
  a.flags.writeable = False; b.flags.writeable = False


# With statement 
# Useful when there are related operations
with open('output.txt', 'w') as f:
    f.write('Hi there!')

	
# Create data frames
# By columns
left = pd.DataFrame({'name': ['Gunter, Thomas', 'Harbinger, Nicholas', 'Benito, Gisela', 'Rudelich, Herbert', \
                              'Sirignano, Emily', 'Morrison, Michael', 'Morrison, Michael', 'Onieda, Jacqueline'],
                     'age':          [27, 36, 32, 39, 22, 32, 32, 31],
                     'gender':       ['M', 'M', 'F', 'M', 'F', 'M', 'M', 'F']})

# By rows	
df = pd.DataFrame([['cold','slow', np.nan, 2., 6., 3.], 
                   ['warm', 'medium', 4, 5, 7, 9],
                   ['hot', 'fast', 9, 4, np.nan, 6],
                   ['cool', None, np.nan, np.nan, 17, 89],
                   ['cool', 'medium', 16, 44, 21, 13],
                   ['cold', 'slow', np.nan, 29, 33, 17]],
                   columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6'],
                   index=(list('abcdef')))	

# Merge data frames

Import pandas as pd

Df1 = pd.read_csv(‘path/filename.csv’, header = 0)
pd.read_csv(sio, dtype={"user_id": int, "username": object})

Df2 = 

Var1 = ‘income’
Var2 = ‘email’

Key1 = ‘pcn’

Df3 = Df1.merge(right = Df2[[var1, var2, var3]], how = ‘left’, on = [key1, key2])

Df3.to_csv(‘newPath/newFileName.csv’, header = True, index = False)


# merge two files
both 	= 	pd.merge(left, right, on='name', 	how='inner'		, 	sort=True)
r_outer = 	pd.merge(left, right, 				how='right'		, 	sort=True)  
l_outer = 	pd.merge(left, right, 				how='left'		, 	sort=True)	# This is what I like to use 
nomatch = 	pd.merge(left, right, on='name',	how='outer'		, sort=True		, indicator='in=')  # Indicator could be used as the in = _1 variable 

# merge multiple files 



# drop duplicates based on the column name
df = df.drop_duplicates('name')

# Calculate mean salary for groups based on gender
gb1 = df.groupby('gender')['salary'].mean()

# Create object based on groups using the column gender
# and get unique levels
gb2 = df.groupby('gender')
gb2.count()

# Aggregation method using groupby, definining new object or using attributes 
grp_grd = loans.groupby('grade')
grp_grd['income'].mean()

loans.groupby('grade')['income'].mean()

# Multiple metrics when using groupby
grp_grd['income'].aggregate(['mean', 'std', 'count'])


# Combining groupby and stack with multiple metrics using describe()
# by using unstack the output is presented in one table for multiple variables
grp_grd['income', 'dti'].describe().unstack() 

# Get minimum  maximum and count
print(loans.dti.min())
print(loans.dti.max())
loans.dti.count()

# Binning
bins = [0.0, 10.0, 20.0, 30.0]
names=['Low', 'Medium', 'High']
loans['dti_cat'] = pd.cut(loans['dti'], bins, labels=names)

# To avoid changing the total counts when binning set the right parameter to false 
loans['dti_cat'] = pd.cut(loans['dti'], bins, right=False, labels=names)


# Deciles
loans['inc_cat_dec'] = pd.qcut(loans['income'], q=10)

# Count the number of observations in a grouping and sort the corresponding values 
pd.value_counts(loans['inc_cat_dec'].sort_values())

# Range by group
 def max_min(x):
        return x.max() - x.min()
dti_grd_grp = loans.groupby(['grade', 'dti_cat'])
dti_grd_grp.income.agg(max_min).unstack()


# Apply transformations by group
# It applies the transformation to all the numberic variables in the dataset
zscore = lambda x: (x - x.mean()) / x.std()
type(zscore)
t_loans = grp_inc_cat.transform(zscore)  



# proc freq table similar

pd.crosstab([loans.dti_cat], [loans.inc_cat_dec], \
             values=loans.income, aggfunc='count', margins=True, colnames=['Income Deciles'], \
			 rownames=['Debt/Income Ratio'])


			 
# Create a function to replace nan with group average
func = lambda x: x.fillna(x.mean())
type(func)
trans = gb2.transform(func)

# Drop columns for a dataframe df
df.drop('column_name', axis=1, inplace=True)

# Selecting column by using dot and name
df.columnname


# First and last observation in a group
df.groupby('status').first(()
df.groupby('status').last()


# Selecting specific columns by name
df[['Sex_of_Driver', 'Time']].head(10)

_names_vec = ['Sex_of_Driver', 'Time']
df[_names_vec].head(10)



# Types of missing values
# None vs. np.nan, the latter can work better for arithmetic functions 
# that don't return NaNs when NaNs are present
s1 = np.array([32, None, 17, 109, 201])
s1 = np.array([32, np.nan, 17, 109, 201])

# Count number of cases with missing values ;

df.isnull().sum()

# Create dataset that includes the rows with at lest one missing value 
null_data = df[df.isnull().any(axis=1)]
null_data.head()
	
# Drop rows with at least one NaN
df3 = df2.dropna()

# Drops columns with at least one NaN
df4 = df2.dropna(axis='columns')

# We can define a treshold: we keep rows or columns with more than threshold not missing values
df5 = df2.dropna(thresh=5)

# Populating missing values 
df6 = df2.fillna(0)
df8 = df2[["col3", "col4", "col5"]].fillna(df2.col6.mean()) # Replace with the same computed value 
df9 = df2.fillna(method='ffill')   # Forward fill
df10 = df2.fillna(method='bfill')	# Backwards fill


# Print missing values
for col_name in d_1.columns:
    print (col_name, end="---->")
    print (sum(d_1[col_name].isnull()))
	
for col_name in d_1.columns:
    print (col_name ,  d_1[col_name].dtypes)
