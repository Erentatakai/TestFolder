import pandas as pd
from pypinyin import lazy_pinyin
import json
from rag_new import ImagingAnalysisRAG  # 假设您已有RAG模块
from tqdm import tqdm
from pathlib import Path

# 1. 数据预处理
def preprocess_data(excel_path):
    dtype_dict = {
        "住院号": str,  # 强制字符串类型
        "门诊号": str   # 强制字符串类型
    }
    df = pd.read_excel(excel_path, dtype=dtype_dict)
    id_columns = ["住院号", "门诊号"]  # 需要处理空值的列
    for col in id_columns:
        df[col] = df[col].astype(str).replace(["nan", "None", ""], "UNKNOWN")
    
    
    
    # 姓名转拼音
    #df["pinyin_name"] = df["病人姓名"].apply(lambda x: "_".join(lazy_pinyin(x)))
    df["病人姓名"]=df["病人姓名"].apply(lambda x:"_".join(lazy_pinyin(x)))
    
    return df



# 运行
if __name__ == "__main__":
    result_df=preprocess_data("zju_hospital_merged_file.xlsx")
    result_df.to_csv("preprocessed_data_for_zju_hospital_merged_file.csv", index=False, encoding='utf-8-sig')
    print(result_df.head()) 