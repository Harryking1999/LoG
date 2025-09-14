import openai
import json
import re
import time
import os
from typing import List, Dict, Any
from tqdm import tqdm

class DeepSeekAPIClient:
    def __init__(self, api_key: str, model_name: str = "deepseek-reasoner", 
                 api_base: str = "https://api.deepseek.com/beta",
                 max_new_tokens: int = 24000, stop_words: List[str] = None):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words or []
        
        # 设置OpenAI客户端（0.28.0版本的配置方式）
        openai.api_key = api_key
        openai.api_base = api_base
    
    def chat_completions_with_backoff(self, model: str, messages: List[Dict], 
                                    max_completion_tokens: int, stop: List[str], 
                                    temperature: float = 0.7, max_retries: int = 3):
        """带重试机制的API调用"""
        for attempt in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_completion_tokens,
                    stop=stop,
                    temperature=temperature
                )
                return response
            except Exception as e:
                print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    raise e
    
    def get_response(self, input_string, temperature: float = 0.7):
        """获取模型响应"""
        # 按照您提供的风格处理输入
        if isinstance(input_string, list) and self.model_name in ['deepseek-reasoner', 'deepseek-r1', 'Pro/deepseek-ai/DeepSeek-R1']:
            messages = [
                {"role": "user", "content": input_string[0]},
            ]
        else:
            messages = [{"role": "user", "content": input_string}]
                
        response = self.chat_completions_with_backoff(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=self.max_new_tokens,
            stop=self.stop_words,
            temperature=temperature
        )
        print('response: ', response)
        return response
    
    def extract_boxed_answer(self, text: str) -> str:
        """提取\boxed{}中的答案，并处理\text{}格式"""
        pattern = r'\\boxed\{([^}]*)\}'
        matches = re.findall(pattern, text)
        if matches:
            answer = matches[-1].strip()  # 获取最后一个匹配项
            
            # 处理\text{...}格式，提取其中的内容
            text_pattern = r'\\text\{([^}]*)\}'
            text_match = re.search(text_pattern, answer)
            if text_match:
                return text_match.group(1).strip()
            
            return answer
        return None
    
    def extract_question_content(self, question_text: str) -> str:
        """提取question中的'**Question**: Is it true or false or unkown: xxxx?'部分"""
        pattern = r'\*\*Question\*\*:\s*Is it true or false or unkown:\s*(.+?)\?'
        match = re.search(pattern, question_text, re.IGNORECASE)
        if match:
            return f"**Question**: Is it true or false or unkown: {match.group(1).strip()}?"
        return question_text  # 如果没有找到匹配的格式，返回原始问题
    
    def normalize_boolean_answer(self, answer: str) -> str:
        """标准化布尔答案，将包含true/false的答案统一格式"""
        if answer is None:
            return None
        
        answer_lower = answer.lower().strip()
        
        # 检查是否包含true或false
        if 'true' in answer_lower:
            return 'true'
        elif 'false' in answer_lower:
            return 'false'
        else:
            # 如果不包含true/false，返回原答案（用于数值等其他类型的答案）
            return answer.strip()
    
    def compare_answers(self, predicted: str, expected: str) -> bool:
        """比较预测答案和期望答案"""
        if predicted is None or expected is None:
            return False
        
        # 对于布尔类型答案，先标准化再比较
        predicted_norm = self.normalize_boolean_answer(predicted)
        expected_norm = self.normalize_boolean_answer(expected)
        
        return predicted_norm.lower() == expected_norm.lower()
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """加载JSONL文件"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        except FileNotFoundError:
            print(f"file not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return []
        return data
    
    def generate_output_filename(self, input_path: str, model_name: str) -> str:
        """根据输入文件路径和模型名生成输出文件名"""
        # 提取文件名（不包括路径和扩展名）
        filename = os.path.splitext(os.path.basename(input_path))[0]  # 例如: LoG_14
        return f"evaluation_results.{filename}.{model_name}.json"
    
    def evaluate_questions(self, jsonl_path: str, model_name: str = None, 
                          temperature: float = 0.7) -> Dict[str, Any]:
        """评估问题并统计结果"""
        if model_name:
            self.model_name = model_name
            
        data = self.load_jsonl(jsonl_path)
        if not data:
            return {"error": "cannot load files"}
        
        results = {
            "total": len(data),
            "correct": 0,
            "incorrect": 0,
            "no_boxed": 0,
            "api_errors": 0,
            "model_name": self.model_name,
            "input_file": jsonl_path,
            "details": []
        }
        
        # 使用tqdm添加进度条
        with tqdm(total=len(data), desc="process", unit="question") as pbar:
            for i, item in enumerate(data):
                question = item.get("question", "")
                expected_answer = item.get("answer", "")
                
                # 提取特定格式的question内容
                extracted_question = self.extract_question_content(question)
                
                try:
                    # 获取模型响应
                    response = self.get_response(question, temperature)
                    response_text = response['choices'][0]['message']['content']
                    
                    # 提取boxed答案
                    predicted_answer = self.extract_boxed_answer(response_text)
                    
                    # 判断结果
                    if predicted_answer is None:
                        result_status = "no_boxed"
                        results["no_boxed"] += 1
                        status_msg = "未找到\\boxed{}"
                    elif self.compare_answers(predicted_answer, expected_answer):
                        result_status = "correct"
                        results["correct"] += 1
                        # 显示标准化后的答案用于调试
                        norm_pred = self.normalize_boolean_answer(predicted_answer)
                        status_msg = f"correct: {norm_pred}"
                    else:
                        result_status = "incorrect"
                        results["incorrect"] += 1
                        norm_pred = self.normalize_boolean_answer(predicted_answer)
                        norm_exp = self.normalize_boolean_answer(expected_answer)
                        status_msg = f"false: predict={norm_pred}, expectation={norm_exp}"
                    
                    # 记录详细结果
                    results["details"].append({
                        "index": i,
                        "original_question": question,
                        "extracted_question": extracted_question,  # 新增：提取的特定格式问题
                        "expected": expected_answer,
                        "predicted": predicted_answer,
                        "predicted_normalized": self.normalize_boolean_answer(predicted_answer),
                        "expected_normalized": self.normalize_boolean_answer(expected_answer),
                        "status": result_status,
                        "full_response": response_text
                    })
                    
                except Exception as e:
                    result_status = "api_error"
                    status_msg = f"API error: {str(e)[:50]}..."
                    results["api_errors"] += 1
                    results["details"].append({
                        "index": i,
                        "original_question": question,
                        "extracted_question": extracted_question,
                        "expected": expected_answer,
                        "predicted": None,
                        "status": "api_error",
                        "error": str(e)
                    })
                
                # 更新进度条
                current_accuracy = results["correct"] / (i + 1) if (i + 1) > 0 else 0
                pbar.set_postfix({
                    'Correct': results["correct"],
                    'Acc': f"{current_accuracy:.1%}",
                    'Status': status_msg[:30]
                })
                pbar.update(1)
        
        # 计算最终准确率
        if results["total"] > 0:
            results["accuracy"] = results["correct"] / results["total"]
        else:
            results["accuracy"] = 0.0
            
        return results
    
    def save_results(self, results: Dict[str, Any], input_path: str, model_name: str = None):
        """保存评估结果，文件名格式：evaluation_results.LoG_x.model_name.json"""
        if model_name is None:
            model_name = self.model_name
            
        output_path = self.generate_output_filename(input_path, model_name)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"results preserved in: {output_path}")
        return output_path

def main(jsonl_path: str = "./generated_data/LoG_14.jsonl", 
         model_name: str = "deepseek-reasoner", 
         api_key: str = "sk-b56f448069294b79967b8c897aebcec3"):
    """
    主函数，支持参数化输入
    
    Args:
        jsonl_path: JSONL文件路径
        model_name: 模型名称
        api_key: API密钥
    """
    
    # 创建客户端
    client = DeepSeekAPIClient(
        api_key=api_key,
        model_name=model_name,
        api_base="https://api.deepseek.com/beta",
        max_new_tokens=24000
    )
    
    # 评估问题
    print(f"start evaluation...")
    print(f"file path: {jsonl_path}")
    print(f"model name: {model_name}")
    
    results = client.evaluate_questions(jsonl_path, model_name)
    
    # 检查是否有错误
    if "error" in results:
        print(f"error: {results['error']}")
        return
    
    # 打印统计结果
    print(f"\n=== Evaluation Results ===")
    print(f"question count: {results['total']}")
    print(f"correct: {results['correct']}")
    print(f"incorrect: {results['incorrect']}")
    print(f"format error(no boxed): {results['no_boxed']}")
    print(f"API error: {results['api_errors']}")
    print(f"Acc: {results['accuracy']:.2%}")
    print(f"model: {results['model_name']}")
    print(f"input file: {results['input_file']}")
    
    # 保存详细结果
    output_file = client.save_results(results, jsonl_path, model_name)
    print(f"detailed results preserved in: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepSeek API evaluation tool")
    parser.add_argument("--jsonl_path", type=str, default="./generated_data/LoG_5.jsonl",
                       help="JSONL file path")
    parser.add_argument("--model_name", type=str, default="deepseek-reasoner",
                       help="model name")
    parser.add_argument("--api_key", type=str, default="sk-b56f448069294b79967b8c897aebcec3",
                       help="API key")
    
    args = parser.parse_args()
    
    main(args.jsonl_path, args.model_name, args.api_key)