import json
import re
import argparse
import os
from typing import List, Dict, Any, Tuple


class StepByStepEvaluator:
    def __init__(self, api_key: str, model_name: str = "deepseek-reasoner", 
                 api_base: str = "https://api.deepseek.com/beta", debug_mode: bool = False,
                 context_window_size: int = 2):
        """
        初始化逐步评估器
        
        Args:
            api_key: API密钥
            model_name: 模型名称
            api_base: API基础URL
            debug_mode: 调试模式
            context_window_size: 上下文窗口大小
        """
        self.debug_mode = debug_mode
        self.model_name = model_name
        self.context_window_size = context_window_size
        
        if not debug_mode:
            # 只在非调试模式下导入和初始化API客户端
            from apply_llm import DeepSeekAPIClient
            self.client = DeepSeekAPIClient(
                api_key=api_key,
                model_name=model_name,
                api_base=api_base,
                max_new_tokens=10000  # 对于分析任务，不需要太长的响应
            )
        else:
            self.client = None
            
        self.extract_prompt_template = self.load_extract_prompt()
    
    def load_extract_prompt(self) -> str:
        """加载提取提示模板"""
        try:
            with open('extract_prompt.txt', 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError("找不到 extract_prompt.txt 文件")
    
    def load_evaluation_log(self, log_path: str) -> Dict[str, Any]:
        """加载评估日志文件"""
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到日志文件: {log_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析错误: {e}")
    
    def split_reasoning_text(self, text: str) -> tuple[List[str], List[str]]:
        """
        将推理文本按句子分割，同时保存分隔符信息
        
        Args:
            text: 推理文本
            
        Returns:
            (句子列表, 分隔符列表)
        """
        # 使用多个分隔符进行分割：句号、问号、感叹号、连续换行符
        # 将连续的换行符(\n\n+)作为一个分隔符处理
        parts = re.split(r'([.!?。？！]|\n\n+)', text)
        
        # 分离句子和分隔符
        sentences = []
        separators = []
        
        i = 0
        while i < len(parts):
            if i + 1 < len(parts):
                content = parts[i].strip()
                separator = parts[i + 1]
                
                if content:  # 只处理有内容的句子
                    if re.match(r'\n\n+', separator):
                        # 连续换行符
                        sentences.append(content)
                        separators.append('\n\n')  # 标准化为\n\n
                    elif separator in '.!?。？！':
                        # 标点符号需要添加到句子末尾
                        sentences.append(content + separator)
                        separators.append(' ')  # 标点符号后用空格分隔
                i += 2
            else:
                sentence = parts[i].strip()
                if sentence:
                    sentences.append(sentence)
                    separators.append('')  # 最后一个句子没有分隔符
                i += 1
        
        return sentences, separators
    
    def create_context_with_sentence(self, sentences: List[str], separators: List[str], current_index: int) -> str:
        """
        创建带有上下文的句子字符串，标记当前句子，保留原始分隔符
        
        Args:
            sentences: 所有句子列表
            separators: 分隔符列表
            current_index: 当前句子的索引
            
        Returns:
            格式化的上下文字符串
        """
        context_parts = []
        
        # 计算上下文范围
        start_idx = max(0, current_index - self.context_window_size)
        end_idx = min(len(sentences), current_index + self.context_window_size + 1)
        
        for i in range(start_idx, end_idx):
            if i == current_index:
                # 标记当前句子
                context_parts.append(f"\\current{{{sentences[i]}}}")
            else:
                # 上下文句子
                context_parts.append(sentences[i])
            
            # 添加分隔符（除了最后一个句子）
            if i < end_idx - 1 and i < len(separators):
                separator = separators[i]
                if separator:  # 只有非空分隔符才添加
                    context_parts.append(separator)
        
        return "".join(context_parts)
    
    def create_analysis_prompt(self, sentences: List[str], separators: List[str], current_index: int) -> str:
        """
        创建分析提示，将带上下文的句子插入到提取提示模板中
        
        Args:
            sentences: 所有句子列表
            separators: 分隔符列表
            current_index: 当前句子的索引
            
        Returns:
            完整的分析提示
        """
        # 生成带上下文的句子字符串
        context_with_sentence = self.create_context_with_sentence(sentences, separators, current_index)
        
        # 使用模板替换
        template = self.extract_prompt_template
        prompt = template.replace("{context_with_sentence}", context_with_sentence)
        
        return prompt
    
    def extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """
        从响应中提取JSON结果
        
        Args:
            response_text: 模型响应文本
            
        Returns:
            提取的JSON对象，如果提取失败返回错误信息
        """
        try:
            # 尝试直接解析整个响应为JSON
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试从响应中提取JSON部分
            json_pattern = r'\{[^}]*\}'
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            
            # 如果都失败了，返回错误信息
            return {
                "error": "无法从响应中提取有效的JSON",
                "raw_response": response_text
            }
    
    def analyze_sentence(self, sentences: List[str], separators: List[str], current_index: int) -> Dict[str, Any]:
        """
        分析单个句子（带上下文）
        
        Args:
            sentences: 所有句子列表
            current_index: 当前句子的索引
            
        Returns:
            分析结果，包含原句子、提示、响应和提取的JSON
        """
        current_sentence = sentences[current_index]
        context_with_sentence = self.create_context_with_sentence(sentences, separators, current_index)
        print(f"输入: {context_with_sentence}")
        
        # 创建分析提示（包含上下文）
        prompt = self.create_analysis_prompt(sentences, separators, current_index)
        
        try:
            # 调用API
            response = self.client.get_response(prompt, temperature=0.0)
            
            # 提取响应文本
            if self.client.model_name in ['deepseek-reasoner']:
                response_text = response['choices'][0]['message']['content']
                thinking = response['choices'][0]['message'].get('reasoning_content', '')
            else:
                response_text = response['choices'][0]['text'].strip()
                thinking = ''
            
            # 提取JSON结果
            json_result = self.extract_json_from_response(response_text)
            
            # 打印分析结果
            if "error" not in json_result:
                print(f"结果: {json_result}")
            else:
                print(f"分析失败: {json_result.get('error', '未知错误')}")
            
            return {
                "sentence": current_sentence,
                "sentence_index": current_index,
                "context": context_with_sentence,
                "prompt": prompt,
                "response_text": response_text,
                "thinking": thinking,
                "json_result": json_result,
                "success": "error" not in json_result
            }
            
        except Exception as e:
            print(f"分析句子时发生错误: {e}")
            return {
                "sentence": current_sentence,
                "sentence_index": current_index,
                "context": context_with_sentence,
                "prompt": prompt,
                "error": str(e),
                "success": False
            }
    
    def evaluate_reasoning_process(self, log_path: str, output_path: str = None, debug_mode: bool = False) -> Dict[str, Any]:
        """
        评估推理过程
        
        Args:
            log_path: 日志文件路径
            output_path: 输出文件路径，如果为None则自动生成
            
        Returns:
            评估结果
        """
        print(f"开始评估推理过程...")
        print(f"日志文件: {log_path}")
        
        # 加载日志
        log_data = self.load_evaluation_log(log_path)
        details = log_data.get('details', [])
        
        if not details:
            raise ValueError("日志文件中没有找到details数据")
        
        print(f"找到 {len(details)} 条记录")
        
        # 评估结果
        evaluation_results = {
            "log_path": log_path,
            "total_records": len(details),
            "model_name": self.model_name,
            "processed_records": [],
            "summary": {
                "total_sentences": 0,
                "successful_analyses": 0,
                "failed_analyses": 0,
                "reasoning_sentences": 0,
                "statement_sentences": 0,
                "other_sentences": 0
            }
        }
        
        # 为了测试方便，只处理第一条记录
        print("\n处理第一条记录（测试模式）...")
        
        record = details[0]
        print(f"记录索引: {record.get('index', 'N/A')}")
        print(f"问题状态: {record.get('status', 'N/A')}")
        
        # 获取推理过程文本
        reasoning_text = record.get('full_response', '')
        thinking_text = record.get('thinking', '')

        if(len(thinking_text) > 0):
            reasoning_text = thinking_text
        
        if not reasoning_text:
            print("警告: 没有找到推理过程文本")
            return evaluation_results
        
        print(f"推理文本长度: {len(reasoning_text)} 字符")
        
        # 分割句子
        sentences, separators = self.split_reasoning_text(reasoning_text)
        print(f"分割得到 {len(sentences)} 个句子")
        
        # 分析每个句子
        sentence_analyses = []
        for i in range(len(sentences)):
            print(f"\n[{i+1}/{len(sentences)}]", end=" ")
            
            if debug_mode:
                # 调试模式：不调用API，只显示句子和提示
                current_sentence = sentences[i]
                context = self.create_context_with_sentence(sentences, separators, i)
                prompt = self.create_analysis_prompt(sentences, separators, i)
                analysis = {
                    "sentence": current_sentence,
                    "sentence_index": i,
                    "context": context,
                    "prompt": prompt,
                    "response_text": "[DEBUG MODE - NO API CALL]",
                    "thinking": "",
                    "json_result": {"type": "debug", "message": "调试模式，未调用API"},
                    "success": True
                }
                print(f"输入: {context}")
                print(f"结果: 调试模式")
            else:
                analysis = self.analyze_sentence(sentences, separators, i)
            
            sentence_analyses.append(analysis)
            
            # 更新统计
            evaluation_results["summary"]["total_sentences"] += 1
            if analysis["success"]:
                evaluation_results["summary"]["successful_analyses"] += 1
                json_result = analysis["json_result"]
                if isinstance(json_result, dict) and "type" in json_result:
                    if json_result["type"] == "reasoning":
                        evaluation_results["summary"]["reasoning_sentences"] += 1
                    elif json_result["type"] == "statement":
                        evaluation_results["summary"]["statement_sentences"] += 1
                    elif json_result["type"] == "other":
                        evaluation_results["summary"]["other_sentences"] += 1
            else:
                evaluation_results["summary"]["failed_analyses"] += 1
            
            # 不再重复打印分析结果，因为在analyze_sentence中已经打印了
        
        # 保存处理的记录
        processed_record = {
            "original_record": {
                "index": record.get('index'),
                "status": record.get('status'),
                "question": record.get('extracted_question', ''),
                "expected": record.get('expected'),
                "predicted": record.get('predicted')
            },
            "reasoning_text": reasoning_text,
            "thinking_text": thinking_text,
            "sentences": sentences,
            "sentence_analyses": sentence_analyses
        }
        
        evaluation_results["processed_records"].append(processed_record)
        
        # 保存结果
        if output_path is None:
            # 自动生成输出文件名
            base_name = os.path.splitext(os.path.basename(log_path))[0]
            output_path = f"step_by_step_evaluation_{base_name}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== 评估完成 ===")
        print(f"总句子数: {evaluation_results['summary']['total_sentences']}")
        print(f"成功分析: {evaluation_results['summary']['successful_analyses']}")
        print(f"失败分析: {evaluation_results['summary']['failed_analyses']}")
        print(f"推理句子: {evaluation_results['summary']['reasoning_sentences']}")
        print(f"陈述句子: {evaluation_results['summary']['statement_sentences']}")
        print(f"其他句子: {evaluation_results['summary']['other_sentences']}")
        print(f"结果已保存到: {output_path}")
        
        return evaluation_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="逐步评估推理过程")
    parser.add_argument("--log_path", type=str, required=True,
                       help="评估日志文件路径")
    parser.add_argument("--output_path", type=str, default=None,
                       help="输出文件路径（可选，默认自动生成）")
    parser.add_argument("--api_key", type=str, default="sk-b56f448069294b79967b8c897aebcec3",
                       help="API密钥")
    parser.add_argument("--model_name", type=str, default="deepseek-reasoner",
                       help="模型名称")
    parser.add_argument("--api_base", type=str, default="https://api.deepseek.com/beta",
                       help="API基础URL")
    parser.add_argument("--debug_mode", action="store_true",
                       help="调试模式，不调用API，只测试文本处理功能")
    parser.add_argument("--context_window_size", type=int, default=2,
                       help="上下文窗口大小（默认为2）")
    
    args = parser.parse_args()
    
    try:
        # 创建评估器
        evaluator = StepByStepEvaluator(
            api_key=args.api_key,
            model_name=args.model_name,
            api_base=args.api_base,
            debug_mode=args.debug_mode,
            context_window_size=args.context_window_size
        )
        
        # 执行评估
        results = evaluator.evaluate_reasoning_process(
            log_path=args.log_path,
            output_path=args.output_path,
            debug_mode=args.debug_mode
        )
        
        print("\n评估成功完成！")
        
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
