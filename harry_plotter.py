import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pandas.core.frame import DataFrame

columns = [
    'Data', 'Countries','Local','Industry Sector','Accident Level',
    'Potential Accident Level','Genre','Employee ou Terceiro','Risco Critico'
]

tradutor_num_romanos = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}

columns_dtype = {
    "Data": 'string',
    "Countries": 'category',
    "Local": 'category',
    "Industry Sector": 'category',
    "Accident Level": 'category',
    "Potential Accident Level": 'category',
    "Genre": 'category',
    "Employee ou Terceiro": 'category',
    "Risco Critico": 'category'
}

df = pd.read_csv("accident_data.csv", 
    usecols=columns,
    dtype=columns_dtype
    # Se quiser converter números romanos pra inteiros, descomenta isso aqui
    # converters={
    #     'Accident Level': lambda v: tradutor_num_romanos[v],
    #     'Potential Accident Level': lambda v: tradutor_num_romanos[v]
    # })
)

target = 'Accident Level'
# field = 'Potential Accident Level'
field = 'Genre'

df = df[[target, field]].pivot_table(
    # Trocar essas duas linhas transpõe a matriz
    index=field,columns=target, aggfunc=len)
    # index=target,columns=field, aggfunc=len)

df = df.fillna(0)
# Plot 1: contagem total
fig, ax = plt.subplots()
df.plot.bar(ax=ax)
ax.grid()
ax.tick_params('x', labelrotation=0)
ax.set_ylabel('count')
ax.set_title("Accident count per group")

priori = df.sum()
priori /= priori.sum()
priori = DataFrame(priori).T
priori.index = ["Priori"]

# Plot 2: Proporções a priori e a posteriori
for i in range(len(df.index)):
    df.iloc[i] /= df.iloc[i].sum()

df = pd.concat([priori,df], axis=0)

fig, ax = plt.subplots()
df.plot.bar(ax=ax)
ax.grid()
ax.tick_params('x', labelrotation=0)
ax.set_ylabel('Proportion relative to group')
ax.set_title('A posteriori accident probability')

ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])

# Plot 3: Mudanças na proporção do a priori para o a posteriori
print(df)
df.iloc[1:] = (df.iloc[1:] - df.loc["Priori"])/df.loc["Priori"]
df = df.iloc[1:]

fig, ax = plt.subplots()
df.plot.bar(ax=ax)
ax.grid()
ax.tick_params('x', labelrotation=0)
ax.set_title('Relative proportion change compared to priori')
ax.set_ylabel('Change in proportion')
ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])


plt.show()
