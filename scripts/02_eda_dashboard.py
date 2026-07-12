import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
df = pd.read_csv('/home/claude/project/data/cleaned_transactions.csv', parse_dates=['Transaction_Date'])

fig, axes = plt.subplots(2, 3, figsize=(19, 10))
fig.suptitle('Financial Transactions Dashboard', fontsize=18, fontweight='bold')

# 1. Revenue by Product
rev_by_prod = df.groupby('Product_Name')['Revenue'].sum().sort_values(ascending=False)
axes[0,0].bar(rev_by_prod.index, rev_by_prod.values, color='#4C72B0', edgecolor='black')
axes[0,0].set_title('Total Revenue by Product')
axes[0,0].set_ylabel('Revenue ($)')
axes[0,0].tick_params(axis='x', rotation=30)

# 2. Transaction Status breakdown
status_counts = df['Transaction_Status'].value_counts()
axes[0,1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
              colors=sns.color_palette('Set2'))
axes[0,1].set_title('Transaction Status Breakdown')

# 3. Payment Method breakdown
pay_counts = df['Payment_Method'].value_counts()
axes[0,2].bar(pay_counts.index, pay_counts.values, color='#55A868', edgecolor='black')
axes[0,2].set_title('Transactions by Payment Method')
axes[0,2].set_ylabel('Count')

# 4. Revenue trend over time (only rows w/ valid date)
ts = df.dropna(subset=['Transaction_Date']).copy()
ts['YearMonth'] = ts['Transaction_Date'].dt.to_period('M').dt.to_timestamp()
monthly_rev = ts.groupby('YearMonth')['Revenue'].sum().sort_index()
axes[1,0].plot(monthly_rev.index, monthly_rev.values, color='#C44E52', linewidth=1.5)
axes[1,0].set_title('Monthly Revenue Trend (rows with valid dates)')
axes[1,0].set_ylabel('Revenue ($)')
axes[1,0].tick_params(axis='x', rotation=45)

# 5. Price distribution
axes[1,1].hist(df['Price'].dropna(), bins=40, color='#8172B2', edgecolor='black')
axes[1,1].set_title('Price Distribution')
axes[1,1].set_xlabel('Price ($)')
axes[1,1].set_ylabel('Frequency')

# 6. Avg Revenue by Product & Status (heatmap-ish grouped bar)
pivot = df.pivot_table(index='Product_Name', columns='Transaction_Status', values='Revenue', aggfunc='mean')
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='Blues', ax=axes[1,2], cbar=False)
axes[1,2].set_title('Avg Revenue: Product x Status')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/home/claude/project/dashboard/dashboard.png', dpi=150)
plt.close()
print("Saved dashboard.png")

# Save summary stats for the report
summary = {
    'total_rows': len(df),
    'total_revenue': round(df['Revenue'].sum(), 2),
    'avg_transaction_value': round(df['Revenue'].mean(), 2),
    'top_product_by_revenue': rev_by_prod.index[0],
    'top_product_revenue': round(rev_by_prod.iloc[0], 2),
    'completed_pct': round((df['Transaction_Status']=='Completed').mean()*100, 1),
    'pending_pct': round((df['Transaction_Status']=='Pending').mean()*100, 1),
    'failed_pct': round((df['Transaction_Status']=='Failed').mean()*100, 1),
    'top_payment_method': pay_counts.index[0],
    'date_coverage_pct': round(df['Transaction_Date'].notna().mean()*100, 1),
}
import json
with open('/home/claude/project/data/eda_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(summary)
