from collections import defaultdict
import os
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('../iwildcam_v2.0/metadata.csv')

data1 = data[data['location_remapped'] < 53]
data2 = data[data['location_remapped'] >= 53]
data3 = data2[data2['location_remapped'] >= 53 + 54]
data2 = data2[data2['location_remapped'] < 53 + 54]
data4 = data3[data3['location_remapped'] >= 53 + 54 + 54]
data3 = data3[data3['location_remapped'] < 53 + 54 + 54]
data5 = data4[data4['location_remapped'] >= 53 + 54 + 54 + 54]
data4 = data4[data4['location_remapped'] < 53 + 54 + 54 + 54]
data6 = data5[data5['location_remapped'] >= 53 + 54 + 54 + 54 + 54]
data5 = data5[data5['location_remapped'] < 53 + 54 + 54 + 54 + 54]

domains = [data1, data2, data3, data4, data5, data6]
domain_names = ['data1', 'data2', 'data3', 'data4', 'data5', 'data6']

# 获取每个域的类别集合
c1 = set(data1['y'])
c2 = set(data2['y'])
c3 = set(data3['y'])
c4 = set(data4['y'])
c5 = set(data5['y'])
c6 = set(data6['y'])

# 取所有域共有的类别
common_classes = c1 & c2 & c3 & c4 & c5 & c6

# 排除在任一域中样本数少于5的类别
b = set()
for cls in common_classes:
    for domain in domains:
        if domain[domain['y'] == cls].shape[0] < 5:
            b.add(cls)
            break

# 最终可用的类别集合
c = common_classes - b
print(f"共有 {len(c)} 个有效类别。")

# 创建类别到新标签的映射，从0开始并连续
sorted_classes = sorted(c)  # 先排序，确保映射的一致性
class_to_label = {cls: idx for idx, cls in enumerate(sorted_classes)}
print("类别映射如下：")
for cls, label in class_to_label.items():
    print(f"类别 {cls} 映射到标签 {label}")

# 过滤数据，保留有效类别，并应用类别映射
filtered_domains = []
for domain in domains:
    filtered = domain[domain['y'].isin(c)].copy()
    filtered['new_y'] = filtered['y'].map(class_to_label)
    filtered_domains.append(filtered.reset_index(drop=True))

# 创建输出目录
output_dir = '../iwildcam_v2.0/dataset2'
domains_output_dir = output_dir
os.makedirs(domains_output_dir, exist_ok=True)

# 创建每个域的文件夹
for domain_name in domain_names:
    domain_folder = os.path.join(domains_output_dir, domain_name)
    os.makedirs(domain_folder, exist_ok=True)

# 为每个域作为训练域，生成 train, val 和五个 test 标注文件
for i, domain in enumerate(filtered_domains):
    domain_name = domain_names[i]
    print(f"\nProcessing {domain_name} as training domain...")

    # 按类别分层抽样，确保每个类别在 train 和 val 中的比例一致
    train, val = train_test_split(
        domain,
        test_size=0.2,
        random_state=3,
        stratify=domain['new_y']
    )

    # 构建图像路径
    train['image_path'] = 'train/' + train['filename']
    val['image_path'] = 'train/' + val['filename']

    # 保存 train 和 val 标注文件，命名为 dataX_train.txt 和 dataX_val.txt
    domain_train_val_dir = os.path.join(domains_output_dir, domain_name)
    train_txt_path = os.path.join(domain_train_val_dir, f'{domain_name}_train.txt')
    val_txt_path = os.path.join(domain_train_val_dir, f'{domain_name}_val.txt')

    train[['image_path', 'new_y']].to_csv(train_txt_path, sep=' ', index=False, header=False)
    val[['image_path', 'new_y']].to_csv(val_txt_path, sep=' ', index=False, header=False)

    print(f"保存训练标注文件到 {train_txt_path}，共 {train.shape[0]} 条记录。")
    print(f"保存验证标注文件到 {val_txt_path}，共 {val.shape[0]} 条记录。")

    # 生成五个测试标注文件，分别对应其他五个域
    for j, test_domain in enumerate(filtered_domains):
        if j != i:
            test_domain_name = domain_names[j]
            test_txt_filename = f'{test_domain_name}_test.txt'
            test_txt_path = os.path.join(domain_train_val_dir, test_txt_filename)

            # 构建图像路径
            test_data = test_domain.copy()
            test_data['image_path'] = 'train/' + test_data['filename']

            # 保存测试标注文件
            test_data[['image_path', 'new_y']].to_csv(test_txt_path, sep=' ', index=False, header=False)
            print(f"保存测试标注文件到 {test_txt_path}，共 {test_data.shape[0]} 条记录。")

# （可选）保存类别映射到文件
mapping_df = pd.DataFrame(list(class_to_label.items()), columns=['original_class', 'new_label'])
mapping_csv_path = os.path.join(output_dir, 'class_mapping.csv')
mapping_df.to_csv(mapping_csv_path, index=False)
print(f"保存类别映射文件到 {mapping_csv_path}。")

print("所有标注文件生成完成。")

# ===== 生成以 dataX 组织的 loc domain 文件夹结构 =====
# 准备路径
base_dir = os.path.join(output_dir, 'locations_by_train_domain')
os.makedirs(base_dir, exist_ok=True)

# 构建一个 dict: 每个 domain 下有哪些 location dataframe
domain_to_loc_dfs = defaultdict(list)
for domain_name, df in zip(domain_names, filtered_domains):
    for loc_id, loc_df in df.groupby('location_remapped'):
        domain_to_loc_dfs[domain_name].append((loc_id, loc_df))

# 遍历每个 train_domain，测试文件夹 = 其余所有 domains 的 loc
for train_domain in domain_names:
    test_dir = os.path.join(base_dir, train_domain)
    os.makedirs(test_dir, exist_ok=True)

    for test_domain in domain_names:
        if test_domain == train_domain:
            continue  # 排除自身

        for loc_id, loc_df in domain_to_loc_dfs[test_domain]:
            loc_df = loc_df.copy()
            loc_df['image_path'] = 'train/' + loc_df['filename']
            txt_path = os.path.join(test_dir, f'loc{loc_id:03d}_test.txt')
            loc_df[['image_path', 'new_y']].to_csv(
                txt_path,
                sep=' ',
                index=False,
                header=False
            )

print(f'所有以 train domain 为键的细粒度测试 loc 文件夹已生成到: {base_dir}')


