#LINEAR REGRESSION SINGLE VARIABLE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("homeprices.csv", delimiter='\t')
df.columns = df.columns.str.strip()
print(df)

plt.xlabel("area(sqr ft)")
plt.ylabel("price(US$)")
plt.scatter(df.area, df.price, color="red", marker="+")
reg = linear_model.LinearRegression()
reg.fit(df[["area"]], df.price) #Training the linear regression data #Ok. No errors.
print(reg)

print(reg.predict(np.array(3300).reshape(-1, 1)))
print(reg.coef_) #"M" value
print(reg.intercept_) #"B" value

#For any linear equation y = mx + b ==>
# "m" is slope(coeficient)
# "b" is the Y intercept
# Formula => y=m*x+b
y = 135.78767123*3300+180616.43835616432
print(y)

plt.xlabel("area", fontsize=20)
plt.ylabel("price", fontsize=20)
plt.scatter(df.area, df.price, color="red", marker="+")
plt.plot(df.area, reg.predict(df[["area"]]), color="blue")
plt.show()

d = pd.read_csv("areas.csv")
print(d.head())
p = reg.predict(d)
d["prices"] = p #creating a new column to the data frame p
print(d)
d.to_csv("prediction.csv", index=False)

#EXERCISE

canada = pd.read_csv("canada_percapita.csv")
print(canada.head(5))

plt.xlabel("Year", fontsize=20)
plt.ylabel("Per capita income (US$)", fontsize=20)
plt.plot(canada.year, canada.per_capita_income)
plt.show()

model = linear_model.LinearRegression()
model.fit(canada[["year"]], canada.per_capita_income)
print(model)
print(model.predict([[2020]]))



