import matplotlib.pyplot as plt
from collections import Counter

# =========================
# 1. 讀取交易資料
# =========================
def load_transactions(filename):
    transactions = []
    with open(filename, 'r') as f:
        for line in f:
            items = list(map(int, line.strip().split()))
            if items:
                transactions.append(items)
    return transactions


# =========================
# 主程式
# =========================
if __name__ == "__main__":
    filename = r"retail.txt"
    output_file = "statistics_result.txt"

    transactions = load_transactions(filename)

    # 2. Number of transactions
    num_transactions = len(transactions)

    # 3. Number of unique items
    all_items = [item for trans in transactions for item in trans]
    unique_items = set(all_items)
    num_unique_items = len(unique_items)

    # 4. Average transaction length
    avg_transaction_length = sum(len(t) for t in transactions) / num_transactions

    # 5. Occurrence count of each item
    item_counts = Counter(all_items)

    # Top 10 most frequent items
    top10_items = item_counts.most_common(10)

    # =========================
    # 將結果呈現至終端機
    # =========================
    print("Number of transactions:", num_transactions)
    print("Number of unique items:", num_unique_items)
    print("Average transaction length:", round(avg_transaction_length, 2))



    # =========================
    # 6. 找出前十個發生次數最多的商品，以及繪製長條圖 (English)
    # =========================
    print("\nTop 10 most frequent items:")
    for item, count in top10_items:
        print(f"Item {item}: {count}")
    
    items = [str(item) for item, _ in top10_items]
    counts = [count for _, count in top10_items]

    plt.figure()
    plt.bar(items, counts)
    plt.xlabel("Item ID")
    plt.ylabel("Frequency")
    plt.title("Top 10 Most Frequent Items")
    plt.show()

    # =========================
    # 7. 計算某一個商品組合的發生次數
    # =========================
    target_items = [6, 8, 11]  # 選擇任意一個組合
    target_set = set(target_items)

    occurrence = 0
    for trans in transactions:
        if target_set.issubset(set(trans)):
            occurrence += 1

    print(f"\nItemset {target_items} occurs {occurrence} times in the database.")

    # =========================
    # 將結果寫入.txt檔案之中
    # =========================
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Retail Transaction Statistics\n")
        f.write("=============================\n")
        f.write(f"Number of transactions: {num_transactions}\n")
        f.write(f"Number of unique items: {num_unique_items}\n")
        f.write(f"Average transaction length: {avg_transaction_length:.2f}\n\n")

        f.write("Item occurrence counts:\n")
        for item, count in sorted(item_counts.items()):
            f.write(f"Item {item}: {count}\n")

        f.write("\n")
        f.write(f"Itemset {target_items} occurrence count: {occurrence}\n")

    print(f"\nResults have been saved to '{output_file}'")
