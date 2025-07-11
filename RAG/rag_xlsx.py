import pandas as pd
from pypinyin import lazy_pinyin
import json
from pathlib import Path

def preprocess_data(excel_path):
    # 明确指定日期列的解析格式
    date_columns = ["检查时间", "创建时间", "审核时间", "登记时间"]
    
    dtype_dict = {
        "住院号": str,
        "门诊号": str,
        "上机技师": str
    }
    
    # 读取Excel时明确指定日期列
    df = pd.read_excel(
        excel_path,
        dtype=dtype_dict,
        parse_dates=date_columns
    )
    
    id_columns = ["住院号", "门诊号", "上机技师"]
    for col in id_columns:
        df[col] = df[col].astype(str).replace(["nan", "None", ""], "UNKNOWN")
    
    # 姓名转拼音
    df["病人姓名"] = df["病人姓名"].apply(lambda x: "_".join(lazy_pinyin(x)))
    
    # 将日期时间列格式化为字符串
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df

def save_as_nested_json(df, output_path, name_key="病人姓名"):
    """
    将DataFrame保存为嵌套字典结构的JSON文件
    格式: { "姓名1": {所有其他字段}, "姓名2": {所有其他字段} }
    
    参数:
        df: 要保存的DataFrame
        output_path: 输出文件路径
        name_key: 作为键名的列名
    """
    # 创建目录（如果不存在）
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 转换为嵌套字典结构
    nested_dict = {}
    for _, row in df.iterrows():
        name = row[name_key]
        # 创建当前记录的副本并移除姓名键
        record = row.to_dict()
        # 如果需要保留姓名在值中，可以注释下面这行
        del record[name_key]
        nested_dict[name] = record
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nested_dict, f, ensure_ascii=False, indent=2)
    
    print(f"数据已保存为嵌套JSON格式: {output_path}")

if __name__ == "__main__":
    # 预处理数据
    result_df = preprocess_data("zju_hospital_merged_file.xlsx")
    
    # 保存为CSV（可选）
    result_df.to_csv(
        "preprocessed_data_for_zju_hospital.csv",
        index=False,
        encoding='utf-8-sig'
    )
    
    # 保存为嵌套JSON格式
    save_as_nested_json(result_df, "preprocessed_data_for_zju_hospital.json")
    
    print("\n嵌套JSON结构示例:")
    # 打印第一个样本的格式示例
    sample_name = result_df.iloc[0]["病人姓名"]
    print(f'"{sample_name}": {{...}}')