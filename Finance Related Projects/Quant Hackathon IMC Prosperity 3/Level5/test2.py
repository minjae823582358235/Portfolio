import pandas as pd 
sellerfilereddf=df[df['seller'=='pablo']]
sellerbuyerdf=sellerfilereddf[df['buyer'=='gary']]
pd.to_csv(sellerbuyerdf[[sellerbuyerdf['product'=='fruit']]],'filtered'+buyer+seller+)