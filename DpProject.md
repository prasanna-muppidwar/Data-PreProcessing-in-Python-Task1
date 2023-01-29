<div class="cell markdown" id="u7f2BEpj39jY">

***Task 1:*** Pick a Dataset from Kaggle or any Platform and Perform
Complete Normalization on it.

</div>

<div class="cell code" data-execution_count="2" data-colab="{&quot;height&quot;:424,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="2QBwDwGZChH-" data-outputId="dce0b882-e883-4d5d-b667-00a6c206507d">

``` python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#load the dataset into a pandas dataframe
myDataFrame = pd.read_csv("/content/IPL_2023-22_Sold_Players.csv")
myDataFrame

#here our only requirement is to normalize 'Price' and 'Season' as without other value there would be no use of normalized numbers in dataset
```

<div class="output execute_result" data-execution_count="2">

``` 
     Season               Name Nationality          Type  \
0      2023    Ajinkya Rahane      Indian        Batter    
1      2023     Bhagath Varma      Indian   All-Rounder    
2      2023     Kyle Jamieson    Overseas        Bowler    
3      2023       Ajay Mandal      Indian   All-Rounder    
4      2023    Nishant Sindhu      Indian   All-Rounder    
..      ...                ...         ...           ...   
279    2022  Fazalhaq Farooqi    Overseas        Bowler    
280    2022       Sean Abbott    Overseas        Bowler    
281    2022         R Samarth      Indian       Batsman    
282    2022    Shashank Singh      Indian   All-Rounder    
283    2022     Saurabh Dubey      Indian        Bowler    

                     Team         Price  
0     Chennai Super Kings    50,00,000   
1     Chennai Super Kings    20,00,000   
2     Chennai Super Kings  1,00,00,000   
3     Chennai Super Kings    20,00,000   
4     Chennai Super Kings    60,00,000   
..                    ...           ...  
279   Sunrisers Hyderabad    50,00,000   
280   Sunrisers Hyderabad  2,40,00,000   
281   Sunrisers Hyderabad    20,00,000   
282   Sunrisers Hyderabad    20,00,000   
283   Sunrisers Hyderabad    20,00,000   

[284 rows x 6 columns]
```

</div>

</div>

<div class="cell code" data-execution_count="19" data-colab="{&quot;height&quot;:2494,&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="LLPRZGc0FTjV" data-outputId="58af8528-5f9d-479e-dd3b-a201df8e7155">

``` python
scale = MinMaxScaler()

#used this function to convert names in proper format if there are any errors as we cant check every name manually
def normalize(Name):
  Name = Name.strip() #used this to remove white spaces
  Name = Name.lower() #used to convert to lowercase
  Name = Name.title() #capitalized first letter
  return Name
myDataFrame['Name'] = myDataFrame['Name'].apply(normalize)
#myDataFrame['Price'] = myDataFrame['Price'].apply(lambda x: x.replace(',','')) 
#used this line to remove commas from values

myDataFrame[['Season','Price']] = scale.fit_transform(myDataFrame[['Season','Price']]) #and here we are performing Normalization
# for Specific Two Features

#as of now we have converted values into categorical data lets 
print(myDataFrame.head())

myDataFrame
```

<div class="output stream stdout">

``` 
   Season            Name Nationality          Type                  Team  \
0     1.0  Ajinkya Rahane     Indian        Batter    Chennai Super Kings   
1     1.0   Bhagath Varma     Indian   All-Rounder    Chennai Super Kings   
2     1.0   Kyle Jamieson   Overseas        Bowler    Chennai Super Kings   
3     1.0     Ajay Mandal     Indian   All-Rounder    Chennai Super Kings   
4     1.0  Nishant Sindhu     Indian   All-Rounder    Chennai Super Kings   

      Price  
0  0.016393  
1  0.000000  
2  0.043716  
3  0.000000  
4  0.021858  
```

</div>

<div class="output execute_result" data-execution_count="19">

``` 
     Season              Name Nationality          Type                  Team  \
0       1.0    Ajinkya Rahane     Indian        Batter    Chennai Super Kings   
1       1.0     Bhagath Varma     Indian   All-Rounder    Chennai Super Kings   
2       1.0     Kyle Jamieson   Overseas        Bowler    Chennai Super Kings   
3       1.0       Ajay Mandal     Indian   All-Rounder    Chennai Super Kings   
4       1.0    Nishant Sindhu     Indian   All-Rounder    Chennai Super Kings   
..      ...               ...         ...           ...                   ...   
279     0.0  Fazalhaq Farooqi   Overseas        Bowler    Sunrisers Hyderabad   
280     0.0       Sean Abbott   Overseas        Bowler    Sunrisers Hyderabad   
281     0.0         R Samarth     Indian       Batsman    Sunrisers Hyderabad   
282     0.0    Shashank Singh     Indian   All-Rounder    Sunrisers Hyderabad   
283     0.0     Saurabh Dubey     Indian        Bowler    Sunrisers Hyderabad   

        Price  
0    0.016393  
1    0.000000  
2    0.043716  
3    0.000000  
4    0.021858  
..        ...  
279  0.016393  
280  0.120219  
281  0.000000  
282  0.000000  
283  0.000000  

[284 rows x 6 columns]
```

</div>

</div>
