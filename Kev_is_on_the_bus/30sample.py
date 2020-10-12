import pandas as pd

df = pd.read_csv('SentimentLatestTotal.csv')

dfNe = df[df['sentiment'] == 'NEGATIVE']
dfNt = df[df['sentiment'] == 'NEUTRAL']
dfPo = df[df['sentiment'] == 'POSITIVE']

dfNeS = dfNe.sample(n=30)
dfNtS = dfNt.sample(n=30)
dfPoS = dfPo.sample(n=30)

df30 = dfPoS.append(dfNtS)
df30 = df30.append(dfNeS)

df30.to_csv('realitydf.csv')
