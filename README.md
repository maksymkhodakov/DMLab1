# Лабораторна робота №1  
## з дисципліни "Основи Data Mining"  
## студента групи ТТП-42  
## Ходакова Максима Олеговича

---

## 1. Мета роботи

Дослідити процес **Market Basket Analysis** (MBA) – аналізу ринкового кошика – за допомогою алгоритму **Apriori**. Знайти закономірності в тому, які товари часто купуються разом, і сформувати **правила асоціацій**.

## 2. Опис проєкту

У межах лабораторної роботи реалізовано **5 головних етапів**:

1. **Data Preprocessing**  
   - **Loading Data**: зчитування CSV-файлу, що містить транзакції (стовпці `Member_number`, `Date`, `itemDescription`).  
   - **Data Cleaning**: видалення порожніх рядків, перевірка на наявність пропущених значень.  
   - **Data Transformation**: групування товарів у транзакції (за `Member_number`, `Date`) і **one-hot кодування** (використано `TransactionEncoder`).

2. **Exploratory Data Analysis (EDA)**  
   - Визначення **найбільш поширених товарів** (топ-10).  
   - Візуалізація частоти товарів у вигляді **bar-плоту**.

3. **Implementing Apriori Algorithm**  
   - **Parameter Setting**: задання `min_support`, `min_confidence` та ін.  
   - **Frequent Itemset Generation**: пошук частих наборів товарів (Frequent Itemsets) за допомогою функції `apriori`.  
   - **Rule Generation**: генерація правил асоціацій з використанням метрики `confidence`.

4. **Analysis of Results**  
   - **Interpretation**: аналіз отриманих правил (значення lift, confidence тощо).  
   - **Filtering Rules**: відсікання нерелевантних правил, наприклад, з `lift < 1`.

5. **Visualization**  
   - **Bar plots**: частота товарів та підтримка (support) для топ-10 частих наборів.  
   - **Network Graph**: візуалізація правил асоціацій (топ-10 за `lift`).

---

## 3. Середовище та залежності

Для роботи необхідні:
- **Python 3.9+** (або інша версія Python 3)  
- **pip** для встановлення додаткових бібліотек  
- **Встановлені пакети**:
  - `pandas`
  - `mlxtend`
  - `seaborn`
  - `matplotlib`
  - `networkx`
