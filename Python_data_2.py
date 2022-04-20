#Data export 예제

import pandas as pd
import openpyxl

a=pd.read_excel('excel_data.xlsx') #import
a.to_excel('excel_data_modified.xlsx') #export
b=pd.read_excel('excel_data_modified.xlsx') #다시 import
print(b)