import pandas as pd
import numpy as np

df = pd.read_csv('/mnt/user-data/uploads/dirty_financial_transactions.csv')
raw_rows = len(df)

df = df.drop_duplicates()

canonical_products = ['Headphones', 'Coffee Machine', 'Coffee', 'Tablet', 'Smartphone', 'Laptop']
canonical_sorted = sorted(canonical_products, key=len, reverse=True)

def fix_product(name):
    if pd.isna(name):
        return np.nan
    n = str(name).strip()
    if n == '':
        return np.nan
    # exact/prefix containment match, longest canonical first avoids 'Coffee' swallowing 'Coffee Machine'
    for c in canonical_sorted:
        if n == c or c.startswith(n):
            return c
    return np.nan

df['Product_Name'] = df['Product_Name'].apply(fix_product)

def fix_payment(pm):
    if pd.isna(pm):
        return np.nan
    p = str(pm).strip().lower().replace(' ', '')
    if p == 'paypal':
        return 'PayPal'
    if p == 'creditcard':
        return 'Credit Card'
    if p == 'cash':
        return 'Cash'
    return np.nan

df['Payment_Method'] = df['Payment_Method'].apply(fix_payment)

def fix_status(s):
    if pd.isna(s):
        return np.nan
    s2 = str(s).strip().lower()
    if s2 in ('completed', 'complete'):
        return 'Completed'
    if s2 == 'pending':
        return 'Pending'
    if s2 == 'failed':
        return 'Failed'
    return np.nan

df['Transaction_Status'] = df['Transaction_Status'].apply(fix_status)

def fix_price(p):
    if pd.isna(p):
        return np.nan
    p2 = str(p).replace('$', '').replace(',', '').strip()
    try:
        val = float(p2)
    except ValueError:
        return np.nan
    return round(abs(val), 2)

df['Price'] = df['Price'].apply(fix_price)

df['Quantity'] = df['Quantity'].abs()
df.loc[df['Quantity'] > 1000, 'Quantity'] = np.nan

# Keep invalid dates as NaT rather than dropping the row
df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce', format='%Y-%m-%d')
df['Year'] = df['Transaction_Date'].dt.year
df['Month'] = df['Transaction_Date'].dt.month

# Drop rows missing essential identifying/categorical fields (not date -- that's handled above)
before = len(df)
df = df.dropna(subset=['Transaction_ID', 'Customer_ID', 'Product_Name', 'Payment_Method'])
dropped_essential = before - len(df)

# Drop duplicate Transaction_IDs, keep first occurrence
before = len(df)
df = df.drop_duplicates(subset=['Transaction_ID'], keep='first')
dropped_dupe_ids = before - len(df)

df['Revenue'] = (df['Price'] * df['Quantity']).round(2)
df = df.reset_index(drop=True)

df.to_csv('/home/claude/project/data/cleaned_transactions.csv', index=False)

with open('/home/claude/project/data/cleaning_log.txt', 'w') as f:
    f.write(f"Raw rows: {raw_rows}\n")
    f.write(f"After removing exact duplicate rows: {raw_rows - (raw_rows - (raw_rows))}\n")
    f.write(f"Rows dropped (missing Transaction_ID/Customer_ID/Product_Name/Payment_Method): {dropped_essential}\n")
    f.write(f"Rows dropped (duplicate Transaction_ID): {dropped_dupe_ids}\n")
    f.write(f"Final cleaned row count: {len(df)}\n")
    f.write(f"Invalid/missing dates retained as NaT: {df['Transaction_Date'].isna().sum()} ({df['Transaction_Date'].isna().mean():.1%})\n")
    f.write(f"Missing Price (kept, excluded from revenue/price stats): {df['Price'].isna().sum()}\n")
    f.write(f"Missing Quantity: {df['Quantity'].isna().sum()}\n")
    f.write(f"Missing Transaction_Status: {df['Transaction_Status'].isna().sum()}\n")

print(open('/home/claude/project/data/cleaning_log.txt').read())
print(df.head())
print(df.dtypes)
