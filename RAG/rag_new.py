import os
import tqdm
import json
import logging
import requests
import concurrent.futures
from openai import OpenAI
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
from langchain_community.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
ds_api_key="sk-6d2bed3b523649b5999218c03081accd"
##设置DEEPSEEK_API_KEY
os.environ["DEEPSEEK_API_KEY"] = ds_api_key
os.environ["QWEN_API_KEY"] = "sk-3faa3a1236214dc2a97e4cbf2b5dd993"
os.environ["ZHIPU_API_KEY"] = "52b4bc1df09c44948a250d6558fae6e2.pb7OhC7TGNj2ExeH"
##获取本地文件的目录
local_path = '/data/yunzhixu/Data/hydrocephalus'
# 配置参数
CONFIG = {
    "theory_path": "/data/yunzhixu/Data/hydrocephalus/theory/",
    "patient_files": {
        "imaging": local_path+"/json/zju_chd_info.json",
        "progress": local_path+"/json/progress_notes_pinyin.json",
        "operative": local_path+"/json/operative_notes.json"
    },
    "vector_db_path": "/data/yunzhixu/Data/hydrocephalus/json/embedding/faiss_index",
    "concurrency": 4,
    "chunk_size": 1000,
    "chunk_overlap": 200
}

# 智谱AI Embedding实现类（需放在HydrocephalusRAG类之前）
class ZhipuAIEmbeddings:
    """智谱AI Embedding接口实现"""
    def __init__(self, model: str, api_key: str, embedding_type: str = "query"):
        self.model = model
        self.api_key = api_key or os.getenv("ZHIPU_API_KEY")  # 优先使用环境变量
        self.embedding_type = embedding_type
        self.api_url = "https://open.bigmodel.cn/api/paas/v4/embeddings"

    def __call__(self, text: str) -> List[float]:
        """使实例可调用"""
        return self.embed_query(text)

    def _call_api(self, text: str) -> List[float]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "input": text,
            "encoding_type": self.embedding_type
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            json_response = response.json()
            if "data" not in json_response:
                logging.error(f"Unexpected API response format: {json_response}")
                raise ValueError("API response missing 'data' field")
            if not json_response["data"]:
                raise ValueError("Empty embeddings data in response")
            return json_response["data"][0]["embedding"]
        except Exception as e:
            logging.error(f"Embedding API调用失败: {str(e)}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """文档嵌入"""
        return [self._call_api(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """查询嵌入"""
        return self._call_api(text)

class HydrocephalusRAG:
    def __init__(self):
        # 初始化日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # 多模型嵌入支持
        # self.openai_emb = OpenAIEmbeddings(
        #     model="text-embedding-3-small",
        #     openai_api_key=os.getenv("OPENAI_API_KEY")
        # )
        # 智谱AI Embedding API配置
        self.zhipu_emb = ZhipuAIEmbeddings(
            model="embedding-3",
            api_key=os.getenv("ZHIPU_API_KEY"),  # 从环境变量获取API密钥
            embedding_type="query"  # query/doc
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"]
        )
        self.vector_db = None
        self.client = OpenAI(api_key="sk-3faa3a1236214dc2a97e4cbf2b5dd993", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    def init_vector_db(self):
        """初始化向量数据库，支持中英文混合文档"""
        # 处理医学理论文档(markdown格式)
        md_files = [str(p) for p in Path(CONFIG["theory_path"]).glob("*.md")]
        docs = []
        
        # 预处理文档，增强关键医学概念
        for md_path in md_files:
            with open(md_path, "r", encoding="utf-8") as f:
                text = f.read()
                # 对中文文档添加英文关键词注释
                if any(cn_keyword in md_path for cn_keyword in ["中国", "专家共识"]):  # 中文文档
                    text = self._enhance_chinese_text(text)
                # 对英文文档添加中文关键词注释
                else:  # 其他文档视为英文
                    text = self._enhance_english_text(text)
                docs.extend(self.text_splitter.split_text(text))
        ##save到json
        with open("/data/yunzhixu/Data/hydrocephalus/json/docs1.json", "w") as f:
            json.dump(docs, f, ensure_ascii=False, indent=4)
        
        # 创建向量存储（使用OpenAI嵌入）
        self.vector_db = FAISS.from_texts(
            texts=docs,
            embedding=self.zhipu_emb
        )
        self.vector_db.save_local(CONFIG["vector_db_path"])

    def load_patient_data(self, file_type: str) -> List[Dict]:
        """加载病人数据"""
        with open(Path(CONFIG["patient_files"][file_type]), "r") as f:
            return json.load(f)

    def _enhance_chinese_text(self, text: str) -> str:
        """增强中文文本，添加关键医学概念的英文对应"""
        # 关键医学概念映射
        concept_map = {
            "脑脊液压力": "CSF pressure",
            "分流管压力": "shunt pressure setting",
            "脑室腹腔分流术": "ventriculoperitoneal shunt",
            "交通性脑积水": "communicating hydrocephalus",
            "梗阻性脑积水": "obstructive hydrocephalus",
            "正常压力": "normal pressure",
            "颅内压": "intracranial pressure"
        }
        
        # 在中文术语后添加英文注释
        for cn, en in concept_map.items():
            text = text.replace(cn, f"{cn}({en})")
        return text

    def _enhance_english_text(self, text: str) -> str:
        """增强英文文本，添加关键医学概念的中文对应"""
        # 关键医学概念映射
        concept_map = {
            "CSF pressure": "脑脊液压力",
            "shunt pressure": "分流管压力",
            "ventriculoperitoneal shunt": "脑室腹腔分流术",
            "communicating hydrocephalus": "交通性脑积水",
            "obstructive hydrocephalus": "梗阻性脑积水",
            "normal pressure": "正常压力",
            "intracranial pressure": "颅内压"
        }
        
        # 在英文术语后添加中文注释
        for en, cn in concept_map.items():
            text = text.replace(en, f"{en}({cn})")
        return text


    ##单独测试检索函数的
    def test_search(self):
        """信息提取核心逻辑"""
        # 构建针对性的检索问题
        questions = [
            "脑积水成因或病因是什么? What is the etiology of hydrocephalus?",
            "脑脊液压力正常值范围是多少? What is the normal range of CSF pressure?",
            "分流管压力设置标准是什么? What are the standard shunt pressure settings?",
            "术后需要监测哪些关键指标? What are the key postoperative monitoring indicators?"
        ]
        
        # 多问题检索增强
        contexts = []
        for q in questions:
            docs = self.vector_db.similarity_search(q, k=2)
            contexts.extend([doc.page_content for doc in docs])
        context = "\n".join(list(set(contexts)))  # 去重
        return context

    def extract_info(self,context, img_report: Dict,process_report: Dict,opera_report: Dict) -> Dict:
        """信息提取核心逻辑"""
        # 构建针对性的检索问题


        ##单独提取影像报告的部分：
        # pn_reserved2disease ={}
        # pn_conclusion = {}
        # pn_squence = {}
        # for p_id in img_report.keys():
        #     ##结论：
        #     conclusion = img_report[p_id]['conclusion']
        #     age = img_report[p_id]['age']
        #     #检查
        #     squence = img_report[p_id]['squence']
        #     reserved2disease = img_report[p_id]['RESERVED2DISEASE']
        #     pn_
        # new_img_report = {}
        # new_img_report['conclusion'] = conclusion
        # new_img_report['RESERVED2DISEASE'] = reserved2disease
        # new_img_report['squence'] = squence



        
        prompt_template = """[医疗报告信息提取 - 中英双语]
        请基于前文的医学知识和报告内容提取信息：
        # 报告内容:
        病程报告:
        {process_report}
        手术报告:
        {operative_report}

        # 根结构化数据提取要求，对于病因和类型请结合前文医学知识得到结果。注意对于数值数据如CSF压力，请勿随意猜测，没有则直接填写为"UNKNOWN"：
        1. 病因分析(Etiology Analysis):
        - 类型: ['tumor', 'infection-', 'hemorrhage-related',
               'congenital', 'post-traumatic', 'idiopathic']

        2. 临床分型(Clinical Classification):
        - 类型: ['交通性', '梗阻性']

        3. 压力数据(Pressure Data):
        - 初次手术CSF压力(Initial CSF pressure): [数值(value): float, 单位(unit): "mmH2O"]
        - 分流管压力设置档位(level): [1.0/2.0/3.0],
        - 对应压力范围(pressure_range): "40-60mmH2O/60-80mmH2O",
    

        4. 术后分流管调整记录(Postoperative Adjustments):
        - 时间(date): "YYYY-MM-DD HH:MM"
        - 调整前压力(previous_level): float
        - 调整后压力(new_level): float
        - 调整原因(reason): string

        5. 二次导流手术信息(Secondary Shunt Surgery):
        - 是否实施(performed): boolean
        - 手术时间(surgery_date): "YYYY-MM-DD" (如未实施则标记"N/A")
        - 手术类型(type): ["脑室腹腔分流术(VP shunt)", "腰大池腹腔分流术(LP shunt)", "其他(other)"]

        6. 备注(Remarks):
        - 额外补充信息(additional_info): string,  请使用中文。

        # 处理规则：
        1. 压力数据处理：
        - 自动转换所有压力值为mmH2O单位
        - 档位转换规则：
            1.0 → 40-60 mmH2O (取中间值50mmH2O)
            2.0 → 60-80 mmH2O (取中间值70mmH2O)
        - 混合单位处理：同时保留原始值和转换值
        示例："80mmHg → 1088mmH2O (80×13.6)"

        2. 时间格式：
        - 精确到分钟，格式：YYYY-MM-DD HH:MM
        - 缺失时间戳时使用"UNKNOWN_TIMESTAMP"

        3. 结构化输出要求：
        - 布尔值字段严格使用true/false
        - 数值字段保留两位小数
        - 枚举字段必须从指定选项中选择

        请以严格的JSON格式返回结果，结构如下：
        {{
          "etiology": {{
            "type": "选项中的值",
            "secondary_cause": "string (仅当type=继发性时)"
          }},
          "classification": "选项中的值",
          "initial_csf_pressure": {{
            "value": float,
            "unit": "mmH2O",
            "original_value": "string"
          }},
          "shunt_settings": {{
            "initial_level": float,
            "pressure_range": "string",
          }},
          "postoperative_adjustments": [
            {{
              "date": "datetime",
              "previous_level": float,
              "new_level": float,
              "reason": "string"
            }}
          ],
          "secondary_shunt_surgery": {{
            "performed": boolean,
            "surgery_date": "datetime",
            "type": "string"
          }},
          "remarks": {{
            "additional_info": "string"
          }}
        }}"""
        
        # 调用LLM API（示例）
        formatted_prompt = prompt_template.format(
            context=context,
            process_report=json.dumps(process_report, ensure_ascii=False, indent=2),
            operative_report=json.dumps(opera_report, ensure_ascii=False, indent=2)
        )
        response = self.call_llm_api(formatted_prompt)
        print('response:', response)
        return self.parse_response(response)

    def process_reports(self):


        questions = [
            "脑积水成因或病因是什么? What is the etiology of hydrocephalus?",
            "脑脊液压力正常值范围是多少? What is the normal range of CSF pressure?",
            "分流管压力设置标准是什么? What are the standard shunt pressure settings?",
            "术后需要监测哪些关键指标? What are the key postoperative monitoring indicators?"
        ]
        
        # 多问题检索增强
        contexts = []
        for q in questions:
            docs = self.vector_db.similarity_search(q, k=2)
            contexts.extend([doc.page_content for doc in docs])
        context = "\n".join(list(set(contexts)))  # 去重
        """单独处理每个病人的报告"""
        # 加载所有病人数据
        imaging_data = self.load_patient_data("imaging")
        progress_data = self.load_patient_data("progress")
        operative_data = self.load_patient_data("operative")
        
        # 获取所有病人姓名（假设所有字典有相同的keys）
        patient_names = list(operative_data.keys())
        #设置tqdm进度条
        epoch = tqdm.tqdm(total=len(patient_names), desc="Processing Reports", unit="patient")

        results = {}
        for name in patient_names:
            # 处理当前病人的所有报告
            patient_results = {
                "case": self.extract_info(contexts,imaging_data[name],progress_data[name],operative_data[name]),
            }
            results[name] = patient_results
            # 更新进度条
            epoch.update(1)
            ##save dict
            with open(local_path+"/json/rag_results.json", "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
        


        
            
        return results

    def call_llm_api(self, prompt: str):
        """统一调用LLM API"""
        # try:
        #     client = DeepSeek(api_key=os.getenv('DEEPSEEK_API_KEY'))
        #     response = client.chat_completions.create(
        #         model="deepseek-chat",
        #         messages=[{
        #             "role": "system",
        #             "content": "你是一名专业的神经外科医疗数据分析助手"
        #         }, {
        #             "role": "user",
        #             "content": prompt
        #         }],
        #         temperature=0.2,
        #         max_tokens=1024
        #     )
        #     return response.choices[0].message.content



        ##使用qwen大模型
        try:
            response = self.client.chat.completions.create(
                model="qwen-plus-latest",
                messages=[{
                    "role": "system",
                    "content": "你是一名专业的神经外科医疗数据分析助手"
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"API调用失败: {str(e)}")
            return {}

    def parse_response(self, response: str) -> Dict:
        """解析API响应"""
        data = response #json.load(response)


        return data 
        # try:
        #     data = json.loads(response)
        #     # 压力值标准化处理
        #     if "pressure_measurements" in data:
        #         self.normalize_pressure_values(data["pressure_measurements"])
        #     # 档位转换验证
        #     self.validate_shunt_settings(data["shunt_settings"])
        #     return data
        # except (json.JSONDecodeError, KeyError) as e:
            
        #     return {
        #         "error": f"解析错误: {str(e)}",
        #         "raw_response": response
        #     }

    def validate_shunt_settings(self, settings: Dict):
        """验证分流管设置"""
        if "setting" in settings:
            level = settings.get("setting")
            if level not in [1.0, 2.0]:
                logger.warning(f"无效的分流管档位设置: {level}")
            settings["pressure_range"] = f"{40+20*(level-1)}-{60+20*(level-1)} mmH2O"

    def normalize_pressure_values(self, pressure_data: Dict):
        """压力值标准化处理"""
        if "original_value" in pressure_data:
            value = pressure_data["original_value"]
            try:
                if "mmHg" in value:
                    mmhg = float(value.replace("mmHg", ""))
                    pressure_data["value_mmh2o"] = round(mmhg * 13.6, 1)
                    pressure_data["unit"] = "mmH2O"
                elif "档" in value:
                    level = float(value.replace("档", ""))
                    pressure_data["value_mmh2o"] = f"{40+20*(level-1)}-{60+20*(level-1)}"
                    pressure_data["unit"] = "mmH2O"
                else:  # 处理纯数字输入
                    pressure_data["value_mmh2o"] = float(value)
                    pressure_data["unit"] = "mmH2O"
            except ValueError as e:
                logger.error(f"压力值转换错误: {str(e)}")
                pressure_data["error"] = str(e)




class ImagingAnalysisRAG(HydrocephalusRAG):
    def __init__(self):
        super().__init__()
        self.vector_db = FAISS.load_local(
            CONFIG["vector_db_path"], 
            self.zhipu_emb,
            allow_dangerous_deserialization=True
        )

    def has_reports(process_report: Dict, opera_report: Dict) -> bool:
            """判断是否存在Process_report和Opera_report"""
            return bool(process_report) and bool(opera_report)

    def extract_info(self, context, img_report: Dict) -> Dict:
        """信息提取核心逻辑"""
        # prompt_template = """[影像分析信息提取 - 中英双语]
        # 请基于前文的医学知识和影像报告内容提取信息：
        # # 报告内容:
        # 影像报告:
        # {img_report}

        # # 结构化数据提取要求：
        # 1. 病因分析(Etiology Analysis):
        # - 类型: ['tumor', 'infection', 'hemorrhage-related',
        #         'congenital', 'post-traumatic', 'idiopathic']

        # 2. 临床分型(Clinical Classification):
        # - 类型: ['交通性', '梗阻性']

        # 3. 影像判断的脑积水严重程度(Severity Assessment):
        # - 程度: ['轻度', '中度', '重度']
        # - 判断依据: string (简要说明判断依据)，使用中文。

        # 请以严格的JSON格式返回结果，结构如下：
        # {{
        #     "etiology": {{
        #         "type": "选项中的值",
        #         "secondary_cause": "string (仅当type=继发性时)"
        #     }},
        #     "classification": "选项中的值",
        #     "severity": {{
        #         "level": "选项中的值",
        #         "basis": "string"
        #     }}
        # }}"""


        prompt_template = """[脑积水影像报告结构化提取 - 知识检索增强]


        请基于以下影像报告内容和医学知识提取结构化信息，对于basic要求部分请给出报告的原文语句，不得进行任何编造
        # 报告内容:
        {img_report}

        # 结构化数据提取要求：
        1. 病因分析(Etiology Analysis)
        - 类型: [tumor', 'infection', 'hemorrhage-related',
                'congenital', 'post-traumatic', 'idiopathic']
        - 判断依据: string (判断的原词句)

        2. 临床分型(Clinical Classification)，请必须给出:
        - 类型: ['交通性', '梗阻性']
        - 梗阻位置(如梗阻性): string
        - 判断依据: string (判断的报告的词句)

        3. 影像特征:给出脑部结构关键词，如蛛网膜下腔，第三脑室等等。
        - 脑部结构特征: string
        - 其他显著特征: string
        - 判断依据: string (判断的报告的词句)

        4. 脑室大小变化(如有术前术后数据):
        - 术前测量: string (如包含具体数值，请给出，没有则无，请勿编造）
        - 术后测量: string (包含具体数值，请给出，没有则无，请勿编造）
        - 变化趋势: ['增大', '减小', '稳定']
        - 判断依据: string (判断的原词句)

        5、影像判断的脑积水的病情程度(Severity Assessment):
        - 程度: ['减轻', '稳定', '加重']
        - 判断依据: string (基于原报告内容,请勿编造）

        6、是否进行了手术和手术类型:
        - 是否手术: ['Yes', 'No', 'None'] (如无信息则为None,请勿瞎猜）
        - 手术类型: ['无信息‘，'VP分流', 'VA分流', 'ETV']
        - 判断依据: string (判断的原词句)

        请根据影像报告内容和医学知识尽可能完整地提取信息，以严格的JSON格式返回结果：
        {{
            "etiology": {{
                "type": "选项中的值",
                "basis": "string (判断的原词句)"
            }},
            "classification": {{
                "type": "选项中的值",
                "obstruction_location": "string (如梗阻性)",
                "basis": "string (判断的原词句)"
            }},
            "imaging_features": {{
                "brain_structure_features": "string",
                "other_features": "string",
                "basis": "string (判断的原词句)"
            }},
            "ventricular_changes": {{
                "preop_measurement": "string",
                "postop_measurement": "string",
                "trend": "选项中的值",
                "basis": "string (判断的原词句)"
            }}
            "severity": {{
                "level": "选项中的值",
                "basis": "string (基于原报告内容,请勿编造）"
            }},
            "surgery_info": {{
                "surgical_intervention": "Yes/No/None",
                "surgery_type": "选项中的值",
                "basis": "string (判断的原词句)"
            }}
        }}"""








        
        # 调用LLM API
        formatted_prompt = prompt_template.format(
            context=context,
            img_report=json.dumps(img_report, ensure_ascii=False, indent=2)
        )
        response = self.call_llm_api(formatted_prompt)
        return self.parse_response(response)

    def process_reports(self,patient_names: str = None) -> Dict:
        """单独处理影像分析数据"""
        # 加载影像数据


        questions = [
            "脑积水成因或病因是什么? What is the etiology of hydrocephalus?",
            "脑脊液压力正常值范围是多少? What is the normal range of CSF pressure?",
            "分流管压力设置标准是什么? What are the standard shunt pressure settings?",
            "术后需要监测哪些关键指标? What are the key postoperative monitoring indicators?"
        ]
        
        # 多问题检索增强
        contexts = []
        for q in questions:
            docs = self.vector_db.similarity_search(q, k=2)
            contexts.extend([doc.page_content for doc in docs])
        contexts = "\n".join(list(set(contexts)))  # 去重
        imaging_data = self.load_patient_data("imaging")
        
        
        # 获取所有病人姓名
        if patient_names is   None:
            patient_names = list(imaging_data.keys())



        
        
        # 设置tqdm进度条
        epoch = tqdm.tqdm(total=len(patient_names), desc="Processing Imaging Reports", unit="patient")

        results = {}
        for name in patient_names:
            # 处理当前病人的影像报告
            patient_results = {
                "imaging_analysis": self.extract_info(contexts, imaging_data[name]),
            }
            results[name] = patient_results
            # 更新进度条
            epoch.update(1)
            # 保存中间结果
            with open(local_path + "/json/imaging_results.json", "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
        
        return results




def response_to_json(response: str) -> Dict:
    """将API响应的str转换为JSON格式"""
    ##先判断是否为str还是dict
    if isinstance(response, dict):
        return response
    else:
        try:
            # 尝试解析为JSON
            start = response.find('```json\n') + len('```json\n')
            end = response.find('```', start)
            
            if start == -1 or end == -1:
                # 如果没有标记，尝试直接解析整个响应
                json_str = response
            else:
                # 提取标记之间的内容
                json_str = response[start:end].strip()
                
            # 解析JSON
            data =  json.loads(json_str)
            return data
        except json.JSONDecodeError:
            # 如果解析失败，返回原始字符串
            raise ValueError("无法解析为JSON格式")
            return response

def response_to_json2(raw_output: str) -> Dict:
    """将API响应的str转换为JSON格式"""

    json_string_with_markdown = raw_output

    # 找到JSON内容的开始和结束位置，这是最稳健的方法
    start_index = json_string_with_markdown.find('{')
    end_index = json_string_with_markdown.rfind('}')
    clean_json_string = json_string_with_markdown[start_index : end_index + 1]

    # 3. 解析为Python字典
    try:
        data_dict = json.loads(clean_json_string)
        
        # 4. 验证并使用数据
        print("✅ JSON解析成功！")
        print("数据类型:", type(data_dict))
        
        # 像操作普通字典一样访问数据
        # print("\n提取一些关键信息:")
        # print("病因类型 (Etiology Type):", data_dict['etiology']['type'])
        # print("初始脑脊液压力 (Initial CSF Pressure):", data_dict['initial_csf_pressure']['original_value'])
        return data_dict

    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {e}")
        return {}


class JsonDict_info_extractor:
    def __init__(self, data: Dict[str, Any]):
        ''''
        用于对病例的json数据进行处理
        '''
        self.data = data

    def _get_nested_value(self, data: Dict[str, Any], key: str) -> List[Any]:
        """获取嵌套字典中指定key的值"""
        keys = key.split('.')
        for k in keys:
            if isinstance(data, dict) and k in data:
                data = data[k]
            else:
                return []
        return data 
    def extract_info(self, keys: List[str],map_dict = None) -> Dict[str, Any]:
        result_dict = {}
        # map_dict = {'tumor':'tumor'
        #             ,'infection':'infection'
        #             ,'congenital':'congenital'
        #             ,'post-traumatic':'post-traumatic'
        #             ,'hemorrhage':'hemorrhage'
        #             ,'idiopathic':'idiopathic'
        #             ,'secondary':'secondary'
        # }
        patients = self.data.keys()
        counter_dict = {}
        for key in keys:
            new_dict = {}
            counter_dict[key] = []
            for patient in patients:
                info = self.data[patient]
                tmp = self._get_nested_value(info, key)
                if map_dict is not None:
                    tmp = map_terms(tmp, map_dict)
                counter_dict[key].append(tmp)
                new_dict[patient] = tmp
            result_dict[key] = new_dict
        return counter_dict, result_dict
    
    ##合并更改信息
    # def merge_info(self, keys: List[str]) -> Dict[str, Any]:


##词映射    
def map_terms(origin_name, map_dict) -> str:
    '''
    用于对模糊词映射
    '''
    new_name = []
    for key in map_dict.keys():
        value = map_dict[key]
        if key in origin_name:
            new_name = value
    if new_name == []:
        new_name = origin_name
    return new_name


def test_to_json():
    '''
    将输出字符都转变回json格式

    '''
    #读取results_qwen.json文件
    with open("/data/yunzhixu/Data/hydrocephalus/json/miccai/imaging_analysis_results.json", "r") as f:
        data = json.load(f)
    # print('data:', data)
    # data = response_to_json2(data)
    for key in data.keys():
        info = data[key]['imaging_analysis']
        info = response_to_json2(info)
        data[key]['imaging_analysis'] = info
    
    #save:
    with open("/data/yunzhixu/Data/hydrocephalus/json/miccai/imaging_analysis_results_precessed.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
       
def extract_info_from_json():
    """
    从JSON文件中提取指定键的信息
    :param json_file: JSON文件路径
    :param keys: 要提取的键列表
    :return: 提取的信息字典
    """
    json_file = "/data/yunzhixu/Data/hydrocephalus/json/miccai/imaging_analysis_results_precessed.json"
    with open(json_file, "r") as f:
        data = json.load(f)
    
    extractor = JsonDict_info_extractor(data)
    counter_dict, result_dict = extractor.extract_info(['imaging_analysis.etiology.type',
                                                        'imaging_analysis.classification',
                                                        'imaging_analysis.severity.level'])
    print('result_dict',result_dict)
    
    return counter_dict, result_dict

if __name__ == "__main__":

    #度svr_ids:
    with open('/data/yunzhixu/Data/hydrocephalus/json/data_info/SVR_sample_50_ids.json', 'r') as f:
        sample_ids = json.load(f)
    
    with open('/data/yunzhixu/Data/hydrocephalus/json/cls/p_id_name_dict.json', 'r') as f:
        p_id_name_dict = json.load(f)

    p_name = [p_id_name_dict[str(i)] for i in sample_ids]
    print(p_name)


    ##测试读取markdown文件：
    # zhipu_emb = ZhipuAIEmbeddings(
    #         model="embedding-3",
    #         api_key=os.getenv("ZHIPU_API_KEY"),  # 从环境变量获取API密钥
    #         embedding_type="query"  # query/doc
    #     )
    # vector_db = FAISS.load_local(CONFIG["vector_db_path"], zhipu_emb, allow_dangerous_deserialization=True)
    # question = "查询脑脊液压力为多少时不正常"
    # docs = vector_db.similarity_search(question, k=3)
    # context = "\n".join([doc.page_content for doc in docs])
    # print('context:', context)





    os.environ["ZHIPU_API_KEY"] = "52b4bc1df09c44948a250d6558fae6e2.pb7OhC7TGNj2ExeH"  # 请替换为实际密钥
    
    # rag = HydrocephalusRAG()
    # rag.init_vector_db()
    # results = rag.process_reports()
    
    # # 保存结果
    # output_path = Path("/data/yunzhixu/Data/hydrocephalus/json/results_qwen.json")
    # with open(output_path, "w") as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)


    # 使用 ImagingAnalysisRAG 处理影像报告

    # os.environ["ZHIPU_API_KEY"] = "52b4bc1df09c44948a250d6558fae6e2.pb7OhC7TGNj2ExeH"  # 请替换为实际密钥

    imaging_rag = ImagingAnalysisRAG()
    # p_name = ['zhou_zi_yue', 'lu_yi_bing', 'wang_jia_wei', 'li_jia_yi', 'li_xu_wei', 'shen_ke_hao', 'lu_rui', 'wang_jia_shuo', 'hu_ruo_ke', 'chen_chang_yi', 'pan_jing', 'liu_yi_ming', 'chen_bu_fan', 'zhang_wen_xin', 'yu_yu_xiang', 'wu_zi_qi', 'luo_cheng_xiang', 'shang_guan_long_tao', 'chen_wei_ting', 'fan_wen_xuan']
    imaging_results = imaging_rag.process_reports(p_name[:10])
    output_path = Path("/data/yunzhixu/Data/hydrocephalus/json/miccai/imaging_analysis_results_samples_addcontext.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(imaging_results, f, ensure_ascii=False, indent=2)

    new = {}
    for key in imaging_results.keys():
        info = imaging_results[key]['imaging_analysis']
        info = response_to_json2(info)
        new[key] = info
    with open(local_path + "/json/LLM_result/hydrocephalus_analysis_samples_addcontext_process2.json", "w") as f:
        json.dump(new, f, ensure_ascii=False, indent=4)


    # # 保存影像分析结果
    # output_path = Path("/data/yunzhixu/Data/hydrocephalus/json/miccai/imaging_analysis_results_all.json")
    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(imaging_results, f, ensure_ascii=False, indent=2)


    # test_to_json()
    # extract_info_from_json()
