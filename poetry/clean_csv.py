import pandas as pd

# 假设原始csv文件名为original.csv，且已正确加载为DataFrame
df = pd.read_csv('./tang-poet.csv')

# 1. 清洗出生年份、去世年份和年龄列
def clean_dates(row):
    birth_year = ''.join(filter(str.isdigit, row['出生年份'])) # 取出生年份的数字部分
    death_year = ''.join(filter(str.isdigit, row['去世年份'])) # 取去世年份的数字部分
    if not birth_year or not death_year:
        return None  # 如果任意一项为空，则返回None，后续会删除这些行
    age = int(death_year) - int(birth_year) if death_year else None
    row['年龄'] = age
    if '、' in row['出生年份']:  # 处理出生年份有多重可能的情况
        row['备注'] += f"出生年份可能为{'、'.join(row['出生年份'].split('、')[:-1])}。"
    if '、' in row['去世年份']:
        row['备注'] += f"去世年份可能为{'、'.join(row['去世年份'].split('、')[:-1])}。"
    return row

df_cleaned = df.apply(clean_dates, axis=1).dropna(subset=['出生年份', '去世年份'])  # 删除出生和去世年份均不详的记录

# 2. 数据去重，按照姓名去重并保留第一条记录
df_unique = df_cleaned.drop_duplicates(subset='姓名', keep='first')

# 3. 计算年龄并移除非数字字符
df_unique['年龄'] = df_unique['年龄'].apply(lambda x: str(int(x)) if pd.notnull(x) else '')

# 4. 排序
df_sorted = df_unique.sort_values(by='出生年份')

# 5. 输出清洗后的csv文件
df_sorted.to_csv('tang-poet-cleaned.csv', index=False)