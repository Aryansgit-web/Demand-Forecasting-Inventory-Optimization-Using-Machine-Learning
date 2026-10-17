import pandas as pd
df = pd.read_excel("Online_Retail.xlsx")
df.to_csv("Online_Retail.csv", index=False)
print("Conversion completed!")
