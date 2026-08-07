import pandas as pd

ferma_main = pd.read_csv('/Users/test/Desktop/Яндекс Практикум/Проекты/Ферма/ferma_main.csv', sep=';')
ferma_dad = pd.read_csv('/Users/test/Desktop/Яндекс Практикум/Проекты/Ферма/ferma_main.csv', sep=';')
cow_buy = pd.read_csv('/Users/test/Desktop/Яндекс Практикум/Проекты/Ферма/ferma_main.csv', sep=';')

ferma_main.to_excel(
    '/Users/test/Desktop/Яндекс Практикум/Проекты/Ферма/ferma_main.xlsx',
    engine='openpyxl',
    index=False  # Не сохранять индексы (рекомендуется)
)

ferma_dad.to_excel(
    '/Users/test/Desktop/Яндекс Практикум/Проекты/Ферма/ferma_dad.xlsx',
    engine='openpyxl',
    index=False  # Не сохранять индексы (рекомендуется)
)

cow_buy.to_excel(
    '/Users/test/Desktop/Яндекс Практикум/Проекты/Ферма/cow_buy.xlsx',
    engine='openpyxl',
    index=False  # Не сохранять индексы (рекомендуется)
)