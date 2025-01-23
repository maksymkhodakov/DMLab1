import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# ----------------------------------------------------------
# 1. Завантаження та попередня обробка даних
# ----------------------------------------------------------
df = pd.read_csv('Groceries_dataset.csv')

# Видаляємо порожні рядки (якщо є)
df.dropna(how='all', inplace=True)

print("Перші рядки DataFrame:")
print(df.head())
print("\nІнформація про DataFrame:")
print(df.info())

# ----------------------------------------------------------
# 2. Формування транзакцій
# ----------------------------------------------------------
# Групування за (Member_number, Date), щоб зібрати всі itemDescription у списки
grouped_df = df.groupby(["Member_number", "Date"])["itemDescription"].apply(list).reset_index()
print(f"\nКількість транзакцій (унікальних (Member_number, Date)): {len(grouped_df)}")

transactions = grouped_df["itemDescription"].tolist()

# ----------------------------------------------------------
# 3. One-Hot кодування за допомогою TransactionEncoder
# ----------------------------------------------------------
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
encoded_df = pd.DataFrame(te_array, columns=te.columns_)

print("\nПриклад закодованих даних (перші 5 рядків):")
print(encoded_df.head())

# ----------------------------------------------------------
# 4. Exploratory Data Analysis (EDA)
# ----------------------------------------------------------
item_counts = encoded_df.sum().sort_values(ascending=False)
print("\nТоп-10 товарів за частотою зустрічань:")
print(item_counts.head(10))

# Візуалізація топ-10 товарів
top_n = 10
top_items = item_counts.head(top_n)

plt.figure(figsize=(8, 5))
# Використовуємо один колір ('blue'), щоб уникнути Deprecated Warning у Seaborn
sns.barplot(x=top_items.values, y=top_items.index, orient='h', color='blue')
plt.title(f'Топ-{top_n} товарів за частотою')
plt.xlabel('Частота')
plt.ylabel('Товар')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 5. Пошук частих наборів (Apriori)
# ----------------------------------------------------------
min_support_threshold = 0.001

frequent_itemsets = apriori(encoded_df, min_support=min_support_threshold, use_colnames=True)
print("\nПриклади частих наборів (Frequent Itemsets):")
print(frequent_itemsets.head())

# ----------------------------------------------------------
# 6. Генерація правил асоціацій з нижчою confidence
# ----------------------------------------------------------
min_confidence_threshold = 0.1

num_itemsets = len(encoded_df)

rules = association_rules(
    frequent_itemsets,
    num_itemsets,
    metric="confidence",
    min_threshold=min_confidence_threshold
)

print("\nУсі згенеровані правила:")
print(rules)

# ----------------------------------------------------------
# 7. Аналіз та фільтрація правил
# ----------------------------------------------------------
if rules.empty:
    print("\n--- Жодне правило не знайдено за поточними параметрами ---")
else:
    # Сортуємо за lift та confidence
    rules.sort_values(["lift", "confidence"], ascending=False, inplace=True)
    print("\nПриклади правил (sorted by lift & confidence):")
    print(rules.head(10))

    # Додатково фільтруємо за lift >= 1
    filtered_rules = rules[rules["lift"] >= 1].copy()

    # Примусово робимо lift/confidence числовими, якщо потрібно
    filtered_rules["lift"] = pd.to_numeric(filtered_rules["lift"], errors="coerce")
    filtered_rules["confidence"] = pd.to_numeric(filtered_rules["confidence"], errors="coerce")

    if filtered_rules.empty:
        print("\nПісля фільтрації lift >= 1, жодних правил не лишилось.")
        top_rules = pd.DataFrame()
    else:
        # Обираємо 10 найкращих правил за lift
        top_rules = filtered_rules.nlargest(10, "lift")
        print("\nТоп-10 правил за lift >= 1:")
        print(top_rules)

    # ----------------------------------------------------------
    # 8. ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ
    # ----------------------------------------------------------
    # (A) Бар-плот топ-10 частих наборів
    top_10_frequent_itemsets = frequent_itemsets.nlargest(10, "support").copy()
    top_10_frequent_itemsets["itemsets_str"] = top_10_frequent_itemsets["itemsets"].apply(lambda x: ", ".join(x))

    plt.figure(figsize=(8, 5))
    sns.barplot(x="support", y="itemsets_str", data=top_10_frequent_itemsets, orient='h', color='green')
    plt.title(f"Топ-10 частих наборів (min_support={min_support_threshold})")
    plt.xlabel("Підтримка")
    plt.ylabel("Набір товарів")
    plt.tight_layout()
    plt.show()

    # (B) Побудова графу правил (Top 10 за lift), якщо вони існують
    if not top_rules.empty:
        G = nx.DiGraph()
        for _, row in top_rules.iterrows():
            antecedents = list(row["antecedents"])
            consequents = list(row["consequents"])
            for ant in antecedents:
                for con in consequents:
                    G.add_node(ant, color='skyblue')
                    G.add_node(con, color='lightgreen')
                    label_str = f"conf: {row['confidence']:.2f}, lift: {row['lift']:.2f}"
                    G.add_edge(ant, con, weight=row["lift"], label=label_str)

        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(G, k=2, seed=42)
        colors = [G.nodes[node]['color'] for node in G.nodes()]

        nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=2000)
        nx.draw_networkx_labels(G, pos, font_size=10)
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20)

        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        plt.title("Граф правил (Top 10 за Lift ≥ 1)")
        plt.axis("off")
        plt.show()
    else:
        print("\nНемає правил із lift >= 1 для візуалізації.")

print("\n--- Аналіз ринкового кошика завершено! ---")
