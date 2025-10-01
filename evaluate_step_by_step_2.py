import json
import re
import argparse
import os
import time
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class NodeStatus(Enum):
    """节点状态枚举"""
    CORRECT = "correct"           # 正确节点
    SELF_ERROR = "self_error"     # 节点本身错误
    PARENT_ERROR = "parent_error" # 父节点错误导致的错误


@dataclass
class NodeRecord:
    """节点记录数据类"""
    statement: str
    positions: List[int]          # 出现的位置列表
    status_history: List[NodeStatus]  # 每次出现时的状态
    occurrence_count: int = 0

@dataclass
class LogNode:
    """LoG节点数据类"""
    output: str                    # 节点输出 (如 "x is babcpus")
    depth: int                     # 节点深度/hop
    deduction_rule: str           # 推理规则
    input_nodes: List[str]        # 输入节点列表
    required_premises: Set[str]   # 依赖的前提条件集合
    node_index: int               # 节点在图中的索引


@dataclass
class StatementNode:
    """Statement节点类 - 用于后处理"""
    original_statement: str       # 原始语句
    input_entity: str            # 输入实体
    output_entity: str           # 输出实体
    output_parsed: Dict[str, Any]  # 解析后的输出结构
    occurrence_count: int = 0    # 出现次数
    is_correct: bool = False     # 节点本身是否正确（可以从前提推导出来）
    is_premise: bool = False     # 是否为前提条件
    node_type: str = "unknown"   # 节点类型 (premise/derived/hallucination)
    first_occurrence_index: int = -1  # 首次出现的索引
    
    # 新增：推理路径完整性追踪
    path_is_valid: bool = False  # 推理路径是否完全正确
    dependency_nodes: List[str] = None  # 依赖的前置节点列表
    invalid_dependencies: List[str] = None  # 无效的依赖节点
    reasoning_quality: str = "unknown"  # 推理质量 (perfect/partial/invalid)
    
    def __post_init__(self):
        if self.dependency_nodes is None:
            self.dependency_nodes = []
        if self.invalid_dependencies is None:
            self.invalid_dependencies = []
    
    def __str__(self):
        return f"StatementNode('{self.original_statement}', correct={self.is_correct}, path_valid={self.path_is_valid}, quality={self.reasoning_quality})"


class StatementProcessor:
    """Statement处理器"""
    
    def __init__(self):
        self.condition_list: List[str] = []      # 条件列表
        self.declared_list: List[str] = []       # 声明列表
        self.node_records: Dict[str, NodeRecord] = {}  # 节点记录
    
    def parse_output_entities(self, output_part: str) -> Dict[str, Any]:
        """解析output部分的实体结构"""
        output_part = output_part.strip()
        
        if " and " in output_part:
            entities = [e.strip() for e in output_part.split(" and ")]
            return {"type": "and", "entities": entities}
        elif " or " in output_part:
            entities = [e.strip() for e in output_part.split(" or ")]
            return {"type": "or", "entities": entities}
        else:
            return {"type": "single", "entities": [output_part]}
    
    def extract_initial_conditions(self, original_question: str) -> List[str]:
        """
        从原始问题中提取初始条件
        
        Args:
            original_question: 原始问题文本
            
        Returns:
            初始条件列表
        """
        conditions = []
        
        # 提取Given Information部分
        given_match = re.search(r'\*\*Given Information\*\*:\s*(.*?)\s*\*\*', 
                               original_question, re.DOTALL)
        
        if given_match:
            given_text = given_match.group(1).strip()
            
            # 按句号分割条件
            raw_conditions = re.split(r'\.\s*', given_text)
            
            for condition in raw_conditions:
                condition = condition.strip()
                if condition and ' is ' in condition:
                    conditions.append(condition)
        
        return conditions
    
    def clean_statement(self, statement: str) -> str:
        """
        清洗statement格式，将不标准的表达转换为标准的"A is B"格式
        
        Args:
            statement: 原始statement
            
        Returns:
            清洗后的statement
        """
        if not statement or not isinstance(statement, str):
            return statement
        
        # 去除首尾空格
        cleaned = statement.strip()
        
        # 替换各种等价表达为标准的"is"
        replacements = [
            (" is connected to ", " is "),
            (" is in ", " is "),
            (" belongs to ", " is "),
            (" is contained in ", " is "),
            (" is part of ", " is "),
            (" is under ", " is "),
            (" is also under ", " is "),
            (" leads to ", " is "),
            (" connects to ", " is "),
            (" is from ", " is "),  # 新增
            (" from ", ""),  # 移除from前缀
            (" is not in ", " is not "),  # 处理否定
            (" is not ", " is not "),  # 处理否定
        ]
        
        for old_phrase, new_phrase in replacements:
            cleaned = cleaned.replace(old_phrase, new_phrase)
        
        # 处理 "from A is B" 这种格式 → "A is B"
        if cleaned.startswith("from "):
            cleaned = cleaned[5:].strip()
        
        # 过滤包含"not"的语句，因为我们的推理系统不处理否定
        if " not " in cleaned:
            return ""
        
        # 处理"A includes B" → "B is A"的情况
        if " includes " in cleaned:
            parts = cleaned.split(" includes ", 1)
            if len(parts) == 2:
                a, b = parts[0].strip(), parts[1].strip()
                cleaned = f"{b} is {a}"
        
        # 去除代词和不合格的表达
        pronouns_to_remove = ["it ", "this ", "that "]
        for pronoun in pronouns_to_remove:
            if cleaned.startswith(pronoun):
                # 如果以代词开头，标记为无效
                return ""
        
        # 检查是否为自指语句
        if " is " in cleaned:
            parts = cleaned.split(" is ", 1)
            if len(parts) == 2:
                a, b = parts[0].strip(), parts[1].strip()
                if a == b:
                    # 自指语句，返回空字符串表示应该被过滤
                    return ""
                
                # 检查是否为有效的实体格式
                if not self.is_valid_entity(a):
                    return ""
        
        # 检查是否包含重复主语的复合语句（如 "x is A and x is B"）
        if " and " in cleaned and " is " in cleaned:
            # 检查是否存在重复的主语模式
            if self._has_repeated_subject_pattern(cleaned):
                return ""
        
        # 最终格式验证：必须是 "A is B" 格式且A和B都是有效实体
        if " is " not in cleaned:
            return ""
        
        return cleaned
    
    def _has_repeated_subject_pattern(self, statement: str) -> bool:
        """
        检查语句是否包含重复主语的模式，如 "x is A and x is B"
        
        Args:
            statement: 要检查的语句
            
        Returns:
            是否包含重复主语模式
        """
        # 使用正则表达式匹配重复主语模式
        # 匹配类似 "x is A and x is B" 的模式
        import re
        pattern = r'^(\w+)\s+is\s+\w+\s+and\s+\1\s+is\s+\w+'
        
        if re.match(pattern, statement):
            return True
        
        # 也检查更复杂的情况，如多个重复
        # 分割 "and" 连接的部分
        parts = statement.split(' and ')
        if len(parts) <= 1:
            return False
        
        # 提取每部分的主语
        subjects = []
        for part in parts:
            part = part.strip()
            if ' is ' in part:
                subject = part.split(' is ')[0].strip()
                subjects.append(subject)
        
        # 检查是否有重复的主语
        if len(subjects) > len(set(subjects)):
            return True
        
        return False
    
    def is_valid_entity(self, entity: str) -> bool:
        """
        检查实体是否符合格式要求
        
        Args:
            entity: 实体名称
            
        Returns:
            是否为有效实体
        """
        if not entity or not isinstance(entity, str):
            return False
        
        entity = entity.strip()
        
        # 过滤掉包含"everything"的复合表达
        if "everything" in entity.lower():
            return False
        
        # 过滤掉包含"in"连接词的复合表达（除了简单的以pus结尾的实体）
        if " in " in entity:
            return False
        
        # 检查是否为x或以pus结尾的简单实体
        if entity == "x" or (entity.endswith("pus") and " " not in entity):
            return True
        
        return False
    
    def expand_chain_statement(self, statement: str) -> List[str]:
        """
        将链式语句展开为多个"A is B"格式的语句
        
        Args:
            statement: 链式语句，如"x→A→B→C"
            
        Returns:
            展开后的语句列表，如["x is A", "A is B", "B is C", "x is C"]
        """
        if "→" not in statement:
            return [statement]
        
        # 分割链式节点
        nodes = [node.strip() for node in statement.split("→")]
        
        if len(nodes) < 2:
            return [statement]
        
        # 生成"A is B"格式的语句
        expanded_statements = []
        
        # 1. 生成相邻节点的连接关系
        for i in range(len(nodes) - 1):
            input_node = nodes[i]
            output_node = nodes[i + 1]
            
            # 检查实体有效性
            if self.is_valid_entity(input_node) and self.is_valid_entity(output_node):
                expanded_statements.append(f"{input_node} is {output_node}")
        
        # 2. 添加首尾连接关系（如果链长度大于2）
        if len(nodes) > 2:
            first_node = nodes[0]
            last_node = nodes[-1]
            
            # 检查首尾节点的有效性
            if self.is_valid_entity(first_node) and self.is_valid_entity(last_node):
                # 避免重复添加（如果链长度为2，首尾连接已经在上面添加过了）
                first_to_last = f"{first_node} is {last_node}"
                if first_to_last not in expanded_statements:
                    expanded_statements.append(first_to_last)
        
        return expanded_statements
    
    def parse_statement_to_node(self, statement: str, stmt_type: str) -> Dict[str, Any]:
        """
        将statement解析为节点格式
        
        Args:
            statement: statement字符串
            stmt_type: statement类型
            
        Returns:
            节点字典，包含original, input, output, output_parsed字段
        """
        if stmt_type == "planning" and " is " not in statement:
            # 定义规划格式："A" → 查找A的定义
            if self.is_valid_entity(statement):
                return {
                    "original": statement,
                    "input": statement,
                    "output": "?",  # 表示待查找
                    "output_parsed": {"type": "unknown", "entities": []},
                    "type": stmt_type
                }
            else:
                return None  # 无效实体，过滤掉
        
        if " is " not in statement:
            return None  # 不符合"A is B"格式
        
        # 分割"A is B"
        parts = statement.split(" is ", 1)
        if len(parts) != 2:
            return None
        
        input_part = parts[0].strip()
        output_part = parts[1].strip()
        
        # 检查input部分的有效性
        if not self.is_valid_entity(input_part):
            return None  # input无效，过滤掉
        
        # 解析output部分的结构
        output_parsed = self.parse_output_entities(output_part)
        
        # 检查output部分的有效性（使用解析后的实体列表）
        valid_entities = [e for e in output_parsed["entities"] if self.is_valid_entity(e)]
        if len(valid_entities) != len(output_parsed["entities"]):
            return None  # 有无效实体，过滤掉
        
        return {
            "original": statement,
            "input": input_part,
            "output": output_part,
            "output_parsed": output_parsed,  # 新增：解析后的output结构
            "type": stmt_type
        }
    
    def normalize_and_parse_statements(self, statements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        标准化并解析statements为节点格式
        
        Args:
            statements: 清洗后的statements列表
            
        Returns:
            标准化的节点列表
        """
        normalized_nodes = []
        
        for stmt_dict in statements:
            stmt_type = stmt_dict.get('type', 'unknown')
            statement = stmt_dict.get('statement', '')
            
            if not statement:
                continue
            
            # 1. 处理链式语句
            if "→" in statement:
                expanded_statements = self.expand_chain_statement(statement)
                for expanded_stmt in expanded_statements:
                    node = self.parse_statement_to_node(expanded_stmt, stmt_type)
                    if node:
                        normalized_nodes.append(node)
            else:
                # 2. 处理普通语句
                node = self.parse_statement_to_node(statement, stmt_type)
                if node:
                    normalized_nodes.append(node)
        
        return normalized_nodes
    
    def clean_statements_list(self, statements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        清洗statements列表，标准化格式并过滤无效statements
        
        Args:
            statements: 原始statements列表
            
        Returns:
            清洗后的statements列表
        """
        cleaned_statements = []
        
        for stmt_dict in statements:
            if not isinstance(stmt_dict, dict):
                # 兼容旧格式
                continue
            
            stmt_type = stmt_dict.get('type', 'unknown')
            statement = stmt_dict.get('statement', '')
            
            if not statement:
                continue
            
            # 清洗statement
            cleaned_statement = self.clean_statement(statement)
            
            # 如果清洗后为空，跳过
            if not cleaned_statement:
                continue
            
            # 对于planning类型，进行额外的格式检查
            if stmt_type == 'planning':
                # 检查是否符合两种合法格式：
                # 1. "A is B" (目标/连接规划)
                # 2. "A" (定义规划)
                
                if " is " in cleaned_statement:
                    # 格式1：目标/连接规划
                    parts = cleaned_statement.split(" is ", 1)
                    if len(parts) == 2:
                        a, b = parts[0].strip(), parts[1].strip()
                        if a and b:
                            cleaned_statements.append({
                                "type": stmt_type,
                                "statement": cleaned_statement
                            })
                elif cleaned_statement and " " not in cleaned_statement:
                    # 格式2：定义规划（单个概念）
                    cleaned_statements.append({
                        "type": stmt_type,
                        "statement": cleaned_statement
                    })
                # 其他格式的planning语句被过滤掉
            else:
                # actual类型保持清洗后的结果
                cleaned_statements.append({
                    "type": stmt_type,
                    "statement": cleaned_statement
                })
        
        return cleaned_statements
    
    def can_derive_from_conditions(self, declared_conditions: List[str], current_statement: str) -> bool:
        """
        判断当前陈述是否可以从已声明的条件中推导出来
        
        Args:
            declared_conditions: 已声明的条件列表
            current_statement: 当前陈述
            
        Returns:
            是否可以推导出来
        """
        # TODO: 这个函数需要用户实现具体的推导逻辑
        # 目前返回False作为占位符
        return False
    
    def process_statements(self, statements: List[str], debug_mode: bool = False) -> Dict[str, Any]:
        """
        处理statements列表，执行后处理逻辑
        
        Args:
            statements: statement列表
            debug_mode: 调试模式
            
        Returns:
            处理结果
        """
        if debug_mode:
            # 调试模式：只记录，不执行实际逻辑
            return {
                "debug_mode": True,
                "total_statements": len(statements),
                "statements": statements,
                "message": "调试模式：跳过后处理逻辑"
            }
        
        # TODO: 实现完整的后处理逻辑
        results = {
            "condition_list": self.condition_list,
            "declared_list": self.declared_list,
            "node_records": {},
            "processing_complete": False
        }
        
        # 转换node_records为可序列化的格式
        for statement, record in self.node_records.items():
            results["node_records"][statement] = {
                "statement": record.statement,
                "positions": record.positions,
                "status_history": [status.value for status in record.status_history],
                "occurrence_count": record.occurrence_count
            }
        
        return results


class LogicalReasoningEngine:
    """逻辑推理引擎 - 集成自debug_test.py的推理算法"""
    
    def __init__(self, max_depth: int = 2000, timeout: int = 120):
        """
        初始化推理引擎
        
        Args:
            max_depth: 最大推理深度
            timeout: 超时时间（秒）
        """
        self.max_depth = max_depth
        self.timeout = timeout
    
    def parse_output_entities(self, output_part: str) -> Dict[str, Any]:
        """解析output部分的实体结构"""
        output_part = output_part.strip()
        
        if " and " in output_part:
            entities = [e.strip() for e in output_part.split(" and ")]
            return {"type": "and", "entities": entities}
        elif " or " in output_part:
            entities = [e.strip() for e in output_part.split(" or ")]
            return {"type": "or", "entities": entities}
        else:
            return {"type": "single", "entities": [output_part]}
    
    def statements_equal(self, stmt1: Dict[str, Any], stmt2: Dict[str, Any]) -> bool:
        """判断两个语句是否相等"""
        return (stmt1["input"] == stmt2["input"] and 
                stmt1["output"] == stmt2["output"])
    
    def is_provable(self, target: Dict[str, Any], premises: List[Dict[str, Any]], 
                   visited: set = None, depth: int = 0, start_time: float = None, 
                   debug: bool = False, return_proof_trace: bool = False) -> tuple[bool, Dict[str, Any]] | bool:
        """
        反向推理：判断目标是否可以从前提中推导出来
        
        Args:
            target: 目标结论，格式为节点字典
            premises: 前提条件列表，每个元素为节点字典
            visited: 已访问的目标集合，防止循环
            depth: 递归深度
            start_time: 开始时间
            debug: 是否打印调试信息
            return_proof_trace: 是否返回推理轨迹（包含使用的前提条件）
            
        Returns:
            如果return_proof_trace=True: (是否可推导, 推理轨迹字典)
            如果return_proof_trace=False: 是否可推导
        """
        if visited is None:
            visited = set()
        
        if start_time is None:
            start_time = time.time()
        
        # 初始化推理轨迹
        proof_trace = {
            "target": target,
            "is_provable": False,
            "proof_method": None,
            "used_premises": [],  # 直接使用的前提条件
            "intermediate_steps": [],  # 中间推理步骤
            "depth": depth,
            "reasoning_path": None
        }
        
        # 超时检查
        if time.time() - start_time > self.timeout:
            if debug:
                print(f"推理超时({self.timeout}s)，终止")
            proof_trace["proof_method"] = "timeout"
            return (False, proof_trace) if return_proof_trace else False
        
        indent = "  " * depth
        if debug:
            print(f"{indent}尝试证明: {target['input']} is {target['output']}")
        
        # 改进的循环检测：防止无限递归
        target_key = f"{target['input']}→{target['output']}"
        
        # 如果在当前路径中已经访问过这个目标，直接跳过
        if target_key in visited:
            if debug:
                print(f"{indent}检测到循环依赖，跳过")
            proof_trace["proof_method"] = "circular_dependency"
            return (False, proof_trace) if return_proof_trace else False
        
        if depth > self.max_depth:
            if debug:
                print(f"{indent}超过最大深度({self.max_depth})，跳过")
            proof_trace["proof_method"] = "max_depth_exceeded"
            return (False, proof_trace) if return_proof_trace else False
        
        # 临时添加到visited
        visited.add(target_key)
        
        try:
            # 基础情况：目标已经在前提中
            for premise in premises:
                if self.statements_equal(target, premise):
                    if debug:
                        print(f"{indent}✓ 在前提中找到: {premise.get('original', premise['input'] + ' is ' + premise['output'])}")
                    
                    proof_trace["is_provable"] = True
                    proof_trace["proof_method"] = "direct_premise"
                    proof_trace["used_premises"] = [premise]
                    proof_trace["reasoning_path"] = f"直接前提: {premise.get('original', premise['input'] + ' is ' + premise['output'])}"
                    
                    return (True, proof_trace) if return_proof_trace else True
            
            # 寻找可能的推理路径
            possible_paths = self.find_reasoning_paths(target, premises, debug and depth < 5)
            
            if debug:
                print(f"{indent}找到 {len(possible_paths)} 种推理路径")
            
            # 按优先级排序路径：优先尝试简单的路径
            def path_priority(path):
                rule_priority = {
                    'CE': 1, 'DI_EXPAND': 2, 'DI': 3, 'MP': 3, 'CI': 5, 'MP+CE': 6
                }
                return (len(path['intermediates']), rule_priority.get(path['rule'], 10))
            
            possible_paths.sort(key=path_priority)
            
            # 尝试所有推理路径，只受超时限制
            for i, path in enumerate(possible_paths):
                # 超时检查
                if time.time() - start_time > self.timeout:
                    if debug:
                        print(f"{indent}推理超时，停止尝试更多路径")
                    break
                    
                if debug:
                    print(f"{indent}尝试路径 {i+1}: {path['rule']}")
                    for j, intermediate in enumerate(path['intermediates']):
                        print(f"{indent}  需要: {intermediate['input']} is {intermediate['output']}")
                
                # 检查这条路径的所有中间步骤是否都可以证明
                all_provable = True
                path_used_premises = []
                path_intermediate_steps = []
                
                for intermediate in path['intermediates']:
                    # 检查是否是直接启用前提
                    if intermediate.get("type") == "enabling_premise":
                        # 这是一个直接前提，直接添加到使用的前提列表中
                        if return_proof_trace:
                            path_used_premises.append(intermediate)
                        continue
                    
                    # 使用当前visited的副本，避免影响其他路径
                    if return_proof_trace:
                        intermediate_provable, intermediate_trace = self.is_provable(
                            intermediate, premises, visited.copy(), 
                            depth + 1, start_time, debug, return_proof_trace=True
                        )
                        
                        if intermediate_provable:
                            # 收集中间步骤的前提条件
                            intermediate_premises = intermediate_trace.get("used_premises", [])
                            path_used_premises.extend(intermediate_premises)
                            path_intermediate_steps.append(intermediate_trace)
                        else:
                            all_provable = False
                            break
                    else:
                        if not self.is_provable(intermediate, premises, visited.copy(), 
                                              depth + 1, start_time, debug):
                            all_provable = False
                            break
                
                if all_provable:
                    if debug:
                        print(f"{indent}✓ 路径 {i+1} 成功")
                    
                    proof_trace["is_provable"] = True
                    proof_trace["proof_method"] = path['rule']
                    intermediate_descriptions = []
                    for inter in path['intermediates']:
                        desc = inter.get('original', f"{inter['input']} is {inter['output']}")
                        intermediate_descriptions.append(desc)
                    proof_trace["reasoning_path"] = f"{path['rule']}规则: {' → '.join(intermediate_descriptions)}"
                    
                    if return_proof_trace:
                        # 去重前提条件
                        unique_premises = []
                        seen_premises = set()
                        for premise in path_used_premises:
                            premise_key = f"{premise['input']}→{premise['output']}"
                            if premise_key not in seen_premises:
                                unique_premises.append(premise)
                                seen_premises.add(premise_key)
                        
                        proof_trace["used_premises"] = unique_premises
                        proof_trace["intermediate_steps"] = path_intermediate_steps
                    
                    return (True, proof_trace) if return_proof_trace else True
                elif debug:
                    print(f"{indent}✗ 路径 {i+1} 失败")
            
            if debug:
                print(f"{indent}✗ 所有路径都失败")
            
            proof_trace["proof_method"] = "no_valid_path"
            return (False, proof_trace) if return_proof_trace else False
            
        finally:
            # 移除当前目标的访问记录，允许其他路径访问
            visited.discard(target_key)
    
    def find_reasoning_paths(self, target: Dict[str, Any], premises: List[Dict[str, Any]], 
                           debug: bool = False) -> List[Dict[str, Any]]:
        """从已有前提中寻找可以推导出目标的推理路径"""
        paths = []
        target_input = target["input"]
        target_output = target["output"]
        target_output_parsed = target["output_parsed"]
        
        # 规则1: MP (Modus Ponens)
        mp_paths = self.find_mp_paths(target, premises, debug)
        paths.extend(mp_paths)
        
        # 规则2: CE (Conjunction Elimination) 
        ce_paths = self.find_ce_paths(target, premises, debug)
        paths.extend(ce_paths)
        
        # 规则3: CI (Conjunction Introduction) - 只对and类型的目标应用
        if target_output_parsed["type"] == "and":
            ci_paths = self.find_ci_paths(target, premises, debug)
            paths.extend(ci_paths)
        
        # 规则4: DI (Disjunction Introduction) - 只对or类型的目标应用
        if target_output_parsed["type"] == "or":
            di_paths = self.find_di_paths(target, premises, debug)
            paths.extend(di_paths)
        
        return paths
    
    def find_mp_paths(self, target: Dict[str, Any], premises: List[Dict[str, Any]], 
                     debug: bool = False) -> List[Dict[str, Any]]:
        """寻找MP规则的推理路径"""
        paths = []
        target_input = target["input"]
        target_output = target["output"]
        target_output_parsed = target["output_parsed"]
        
        # 寻找形如 target_input is X 的前提
        for premise in premises:
            if premise["input"] == target_input:
                # 找到了 target_input is X
                x_value = premise["output"]
                x_parsed = premise["output_parsed"]
                
                # 情况1: 如果X是单个值或or组合，寻找 X is target_output
                if x_parsed["type"] == "single" or x_parsed["type"] == "or":
                    # 添加启用前提（target_input is X）
                    enabling_premise = {
                        "input": premise["input"],
                        "output": premise["output"],
                        "output_parsed": premise["output_parsed"],
                        "original": premise.get("original", f"{premise['input']} is {premise['output']}"),
                        "type": "enabling_premise"
                    }
                    
                    intermediate_target = {
                        "input": x_value,
                        "output": target_output,
                        "output_parsed": target_output_parsed,
                        "original": f"{x_value} is {target_output}",
                        "type": "intermediate"
                    }
                    paths.append({
                        "rule": "MP",
                        "intermediates": [enabling_premise, intermediate_target]
                    })
                    
                    if debug:
                        print(f"    MP路径: {target_input} is {x_value} → 需要证明 {x_value} is {target_output}")
                
                # 情况2: 如果X是复合值(如A and B)，可以通过CE提取单个部分，然后继续MP
                elif x_parsed["type"] == "and":
                    # TODO: 优化MP规则 - 支持直接使用复合值中的实体进行MP
                    # 当前问题：对于 x is kirypus and poxgpus + kirypus is xizrpus and robspus
                    # 应该能直接推导 x is xizrpus and robspus，而不需要通过CI规则
                    # 
                    # 优化方案：
                    # 方式2a: 直接使用复合值中的单个实体进行MP（更直接）
                    # for entity in x_parsed["entities"]:
                    #     # 检查是否存在以这个实体开头的前提
                    #     entity_premise_exists = any(p["input"] == entity for p in premises)
                    #     if entity_premise_exists:
                    #         # 添加启用前提（target_input is X）
                    #         enabling_premise = {
                    #             "input": premise["input"],
                    #             "output": premise["output"],
                    #             "output_parsed": premise["output_parsed"],
                    #             "original": premise.get("original", f"{premise['input']} is {premise['output']}"),
                    #             "type": "enabling_premise"
                    #         }
                    #         
                    #         intermediate_target = {
                    #             "input": entity,
                    #             "output": target_output,
                    #             "output_parsed": target_output_parsed,
                    #             "original": f"{entity} is {target_output}",
                    #             "type": "intermediate"
                    #         }
                    #         
                    #         paths.append({
                    #             "rule": "MP",
                    #             "intermediates": [enabling_premise, intermediate_target]
                    #         })
                    
                    # 当前实现：通过CE提取单个部分，然后继续MP
                    for entity in x_parsed["entities"]:
                        ce_intermediate = {
                            "input": target_input,
                            "output": entity,
                            "output_parsed": {"type": "single", "entities": [entity]},
                            "original": f"{target_input} is {entity}",
                            "type": "intermediate"
                        }
                        mp_intermediate = {
                            "input": entity,
                            "output": target_output,
                            "output_parsed": target_output_parsed,
                            "original": f"{entity} is {target_output}",
                            "type": "intermediate"
                        }
                        
                        paths.append({
                            "rule": "MP+CE",
                            "intermediates": [ce_intermediate, mp_intermediate]
                        })
                        
                        if debug:
                            print(f"    MP+CE路径: {target_input} is {x_value} → CE得到 {target_input} is {entity} → MP需要 {entity} is {target_output}")
        
        return paths
    
    def find_ce_paths(self, target: Dict[str, Any], premises: List[Dict[str, Any]], 
                     debug: bool = False) -> List[Dict[str, Any]]:
        """寻找CE规则的推理路径"""
        paths = []
        target_input = target["input"]
        target_output = target["output"]
        target_output_parsed = target["output_parsed"]
        
        # 如果目标是单个实体，寻找包含它的复合条件
        if target_output_parsed["type"] == "single":
            target_entity = target_output_parsed["entities"][0]
            
            # 寻找形如 target_input is ... and target_entity ... 的条件
            for premise in premises:
                if (premise["input"] == target_input and 
                    premise["output_parsed"]["type"] == "and" and
                    target_entity in premise["output_parsed"]["entities"]):
                    
                    # 找到了可以通过CE得到目标的复合条件
                    intermediate = {
                        "input": target_input,
                        "output": premise["output"],
                        "output_parsed": premise["output_parsed"],
                        "original": premise.get("original", f"{target_input} is {premise['output']}"),
                        "type": "intermediate"
                    }
                    paths.append({
                        "rule": "CE",
                        "intermediates": [intermediate]
                    })
                    
                    if debug:
                        print(f"    CE路径: {premise['output']} contains {target_output}")
        
        return paths
    
    def find_ci_paths(self, target: Dict[str, Any], premises: List[Dict[str, Any]], 
                     debug: bool = False) -> List[Dict[str, Any]]:
        """寻找CI规则的推理路径"""
        paths = []
        target_input = target["input"]
        target_output_parsed = target["output_parsed"]
        
        # 只对and类型的目标应用CI规则
        if target_output_parsed["type"] == "and":
            entities = target_output_parsed["entities"]
            
            # 需要为每个实体找到单独的条件
            intermediates = []
            for entity in entities:
                intermediate = {
                    "input": target_input,
                    "output": entity,
                    "output_parsed": {"type": "single", "entities": [entity]},
                    "original": f"{target_input} is {entity}",
                    "type": "intermediate"
                }
                intermediates.append(intermediate)
            
            paths.append({
                "rule": "CI",
                "intermediates": intermediates
            })
            
            if debug:
                print(f"    CI路径: 需要 {len(entities)} 个单独的条件")
        
        return paths
    
    def find_di_paths(self, target: Dict[str, Any], premises: List[Dict[str, Any]], 
                     debug: bool = False) -> List[Dict[str, Any]]:
        """寻找DI规则的推理路径"""
        paths = []
        target_input = target["input"]
        target_output_parsed = target["output_parsed"]
        
        # 只对or类型的目标应用DI规则
        if target_output_parsed["type"] == "or":
            target_entities = set(target_output_parsed["entities"])
            
            # 方法1: 为每个实体创建一个单独的路径（只需要其中一个成功）
            for entity in target_entities:
                intermediate = {
                    "input": target_input,
                    "output": entity,
                    "output_parsed": {"type": "single", "entities": [entity]},
                    "original": f"{target_input} is {entity}",
                    "type": "intermediate"
                }
                
                paths.append({
                    "rule": "DI",
                    "intermediates": [intermediate]
                })
                
                if debug:
                    print(f"    DI路径: 只需要证明 {target_input} is {entity}")
            
            # 方法2: 寻找包含目标实体子集的or条件（DI扩展）
            for premise in premises:
                if (premise["input"] == target_input and 
                    premise["output_parsed"]["type"] == "or"):
                    premise_entities = set(premise["output_parsed"]["entities"])
                    
                    # 如果前提的实体是目标实体的子集，可以通过DI扩展
                    if premise_entities.issubset(target_entities) and premise_entities != target_entities:
                        intermediate = {
                            "input": target_input,
                            "output": premise["output"],
                            "output_parsed": premise["output_parsed"],
                            "original": premise.get("original", f"{target_input} is {premise['output']}"),
                            "type": "intermediate"
                        }
                        
                        paths.append({
                            "rule": "DI_EXPAND",
                            "intermediates": [intermediate]
                        })
                        
                        if debug:
                            print(f"    DI扩展路径: 从 {target_input} is {premise['output']} 扩展到 {target['output']}")
        
        return paths


class PostProcessor:
    """后处理器 - 处理Statement列表和LoG图验证"""
    
    def __init__(self, reasoning_engine: LogicalReasoningEngine, verbose_premise: bool = False):
        """
        初始化后处理器
        
        Args:
            reasoning_engine: 推理引擎实例
            verbose_premise: 是否启用详细前提输出模式
        """
        self.reasoning_engine = reasoning_engine
        self.verbose_premise = verbose_premise
        self.statement_list: List[StatementNode] = []  # Statement节点列表
        self.log_graph: List[Dict[str, Any]] = []      # LoG标准答案图
        self.illuminated_nodes: set = set()            # 已点亮的LoG节点
        
        # 新增：LoG依赖分析相关
        self.log_nodes: List[LogNode] = []             # LoG节点列表（带依赖信息）
        self.premise_statements: Set[str] = set()      # 初始前提条件集合
        self.depth_stats: Dict[int, int] = {}          # 深度统计
        
    def load_log_graph(self, graph_data: List[Dict[str, Any]]):
        """
        加载LoG标准答案图并进行依赖分析
        
        Args:
            graph_data: 图数据列表
        """
        self.log_graph = graph_data
        self.illuminated_nodes = set()
        print(f"[后处理] 加载LoG图，包含 {len(self.log_graph)} 个节点")
        
        # 解析LoG图并构建依赖关系
        self.parse_log_graph(graph_data)
        
        # 打印图结构以便调试
        for i, node in enumerate(self.log_graph):
            print(f"  LoG节点 {i}: {node.get('output', 'N/A')} (规则: {node.get('deduction_rule', 'N/A')}, 深度: {node.get('depth', 'N/A')})")
    
    def parse_log_graph(self, graph_data: List[Dict[str, Any]]) -> None:
        """
        解析LoG图数据，构建节点结构并分析依赖关系
        
        Args:
            graph_data: LoG图数据列表
        """
        self.log_nodes = []
        self.depth_stats = {}
        
        # 构建LogNode对象
        for i, node_data in enumerate(graph_data):
            output = node_data.get('output', '')
            depth = node_data.get('depth', 0)
            deduction_rule = node_data.get('deduction_rule', '')
            input_nodes = node_data.get('input', [])
            
            # 确保input_nodes是列表
            if not isinstance(input_nodes, list):
                input_nodes = []
            
            log_node = LogNode(
                output=output,
                depth=depth,
                deduction_rule=deduction_rule,
                input_nodes=input_nodes,
                required_premises=set(),  # 稍后计算
                node_index=i
            )
            
            self.log_nodes.append(log_node)
            
            # 统计深度分布
            if depth not in self.depth_stats:
                self.depth_stats[depth] = 0
            self.depth_stats[depth] += 1
        
        # 分析依赖关系（如果已经设置了前提条件）
        if self.premise_statements:
            self.analyze_log_dependencies()
    
    def set_initial_premises(self, initial_conditions: List[str]):
        """
        设置初始前提条件并触发依赖分析
        
        Args:
            initial_conditions: 初始条件列表
        """
        self.premise_statements = set(initial_conditions)
        
        # 如果LoG图已经加载，进行依赖分析
        if self.log_nodes:
            self.analyze_log_dependencies()
    
    def analyze_log_dependencies(self) -> None:
        """
        分析所有LoG节点的依赖关系
        """
        # 静默分析所有节点的依赖关系
        for node in self.log_nodes:
            dependencies = self.find_node_dependencies_recursive(node)
            node.required_premises = dependencies
    
    def find_node_dependencies_recursive(self, node: LogNode, visited: Set[int] = None) -> Set[str]:
        """
        递归查找节点的所有依赖前提
        
        Args:
            node: 目标节点
            visited: 已访问的节点索引集合（防止循环）
            
        Returns:
            节点依赖的所有前提条件集合
        """
        if visited is None:
            visited = set()
        
        # 防止循环依赖
        if node.node_index in visited:
            return set()
        
        visited.add(node.node_index)
        
        dependencies = set()
        
        # 如果是深度0的节点，它就是前提条件
        if node.depth == 0:
            dependencies.add(node.output)
            return dependencies
        
        # 对于非前提节点，查找其输入节点的依赖
        for input_statement in node.input_nodes:
            # 在LoG图中查找对应的输入节点
            input_node = self.find_log_node_by_output(input_statement)
            if input_node:
                # 递归获取输入节点的依赖
                input_dependencies = self.find_node_dependencies_recursive(input_node, visited.copy())
                dependencies.update(input_dependencies)
            else:
                # 如果在LoG图中找不到，可能是外部前提
                if input_statement in self.premise_statements:
                    dependencies.add(input_statement)
        
        return dependencies
    
    def find_log_node_by_output(self, output: str) -> Optional[LogNode]:
        """
        根据输出查找LoG节点
        
        Args:
            output: 节点输出字符串
            
        Returns:
            找到的节点，如果不存在返回None
        """
        for node in self.log_nodes:
            if node.output == output:
                return node
        return None
    
    def print_log_dependency_summary(self):
        """打印LoG依赖关系摘要（简化版）"""
        # 统计依赖复杂度分布
        complexity_dist = {}
        for node in self.log_nodes:
            complexity = len(node.required_premises)
            if complexity not in complexity_dist:
                complexity_dist[complexity] = 0
            complexity_dist[complexity] += 1
        
        print(f"[LoG分析] LoG图: {len(self.log_nodes)}节点, {len(self.premise_statements)}前提, 复杂度1-{max(complexity_dist.keys())}")
    
    def illuminate_nodes_by_advanced_matching(self, correct_statements: List[StatementNode]):
        """
        高级节点点亮算法：基于premise匹配来点亮LoG节点
        
        这是核心的第二步和第三步实现：
        1. 对于LoG中没有但is_provable的节点，找到推出该节点所需的已点亮premise
        2. 基于premise匹配来点亮LoG中相应的节点和其子节点
        
        Args:
            correct_statements: 正确的Statement节点列表
        """
        if not self.log_nodes or not self.premise_statements:
            print("[高级点亮] 缺少LoG图或前提条件，跳过高级点亮")
            return
        
        # 第二步：对于不在LoG中但is_provable的节点，获取其推理轨迹
        external_proofs = self.collect_external_statement_proofs(correct_statements)
        
        # 第三步：基于推理轨迹中的premise来点亮LoG节点
        newly_illuminated = self.illuminate_log_nodes_by_premises(external_proofs)
        
        if newly_illuminated > 0:
            print(f"[高级点亮] 新点亮 {newly_illuminated} 个LoG节点")
    
    def collect_external_statement_proofs(self, correct_statements: List[StatementNode]) -> List[Dict[str, Any]]:
        """
        收集不在LoG图中但正确的statement的推理轨迹
        
        Args:
            correct_statements: 正确的Statement节点列表
            
        Returns:
            推理轨迹列表，每个包含statement和其使用的premises
        """
        external_proofs = []
        log_outputs = {node.output for node in self.log_nodes}
        
        # 获取已点亮的premises（初始前提条件）
        illuminated_premises = []
        for stmt in correct_statements:
            if stmt.is_premise:
                premise = {
                    "original": stmt.original_statement,
                    "input": stmt.input_entity,
                    "output": stmt.output_entity,
                    "output_parsed": stmt.output_parsed,
                    "type": "illuminated_premise"
                }
                illuminated_premises.append(premise)
        
        # 检查每个正确但不在LoG中的statement
        external_count = 0
        for stmt in correct_statements:
            if not stmt.is_premise and stmt.original_statement not in log_outputs:
                external_count += 1
                
                # 构造目标节点
                target = {
                    "input": stmt.input_entity,
                    "output": stmt.output_entity,
                    "output_parsed": stmt.output_parsed,
                    "original": stmt.original_statement,
                    "type": "external_statement"
                }
                
                # 使用推理引擎获取推理轨迹（只使用已点亮的前提）
                try:
                    is_provable, proof_trace = self.reasoning_engine.is_provable(
                        target, illuminated_premises, debug=False, return_proof_trace=True
                    )
                    
                    if is_provable and proof_trace:
                        external_proofs.append({
                            "statement": stmt.original_statement,
                            "target": target,
                            "proof_trace": proof_trace,
                            "used_premises": proof_trace.get("used_premises", [])
                        })
                    
                except Exception:
                    pass  # 静默处理错误
        
        return external_proofs
    
    def illuminate_log_nodes_by_premises(self, external_proofs: List[Dict[str, Any]]) -> int:
        """
        基于外部statement的premise来点亮LoG节点
        
        Args:
            external_proofs: 外部statement的推理轨迹列表
            
        Returns:
            新点亮的节点数量
        """
        if not external_proofs:
            return 0
        
        # 收集所有外部推理中使用的premises
        all_used_premises = set()
        for proof in external_proofs:
            used_premises = proof.get("used_premises", [])
            for premise in used_premises:
                premise_statement = premise.get('original', f"{premise['input']} is {premise['output']}")
                all_used_premises.add(premise_statement)
        
        # 遍历LoG节点，检查哪些可以被点亮
        newly_illuminated = 0
        
        # 从最深的节点开始检查（自底向上）
        sorted_nodes = sorted(self.log_nodes, key=lambda x: x.depth, reverse=True)
        
        for log_node in sorted_nodes:
            node_output = log_node.output
            
            # 跳过已经点亮的节点
            if node_output in self.illuminated_nodes:
                continue
            
            # 检查该节点所需的所有前提是否都在外部推理使用的前提中
            required_premises = log_node.required_premises
            
            if required_premises and required_premises.issubset(all_used_premises):
                # 该节点的所有前提都被外部推理使用了，可以点亮
                self.illuminated_nodes.add(node_output)
                newly_illuminated += 1
                
                # 点亮整棵依赖子树（所有前置节点）
                subtree_count = self._illuminate_node_with_subtree(node_output)
                newly_illuminated += subtree_count
        
        return newly_illuminated
    
    def auto_illuminate_dependency_subtree(self, illuminated_node: LogNode) -> int:
        """
        点亮节点的整棵依赖子树
        
        当一个节点被点亮时，应该点亮以它为根的整棵依赖子树，
        包括它依赖的所有前置节点（更深层的节点）以及它们的依赖节点
        
        Args:
            illuminated_node: 被点亮的节点
            
        Returns:
            新点亮的依赖节点数量
        """
        newly_illuminated = 0
        
        # 使用队列进行广度优先遍历，确保所有依赖节点都被访问
        queue = [illuminated_node]
        visited = set()
        
        while queue:
            current_node = queue.pop(0)
            
            # 防止重复访问
            if current_node.node_index in visited:
                continue
            visited.add(current_node.node_index)
            
            # 遍历当前节点的所有输入节点（依赖节点）
            for input_statement in current_node.input_nodes:
                # 在LoG图中找到对应的依赖节点
                dependency_node = self.find_log_node_by_output(input_statement)
                
                if dependency_node and dependency_node.output not in self.illuminated_nodes:
                    # 点亮这个依赖节点（不调用子树点亮，避免无限递归）
                    self.illuminated_nodes.add(dependency_node.output)
                    newly_illuminated += 1
                    print(f"[子树点亮]     └─ 点亮依赖节点: {dependency_node.output} (深度{dependency_node.depth})")
                    
                    # 将这个依赖节点加入队列，继续处理它的依赖
                    queue.append(dependency_node)
        
        return newly_illuminated
    
    def find_statement_node(self, target_statement: str) -> Optional[StatementNode]:
        """
        在statement列表中查找节点
        
        Args:
            target_statement: 目标语句
            
        Returns:
            找到的节点，如果不存在返回None
        """
        for node in self.statement_list:
            if node.original_statement == target_statement:
                return node
        return None
    
    def is_in_premises(self, statement: str, initial_conditions: List[str]) -> bool:
        """
        检查语句是否在初始前提条件中
        
        Args:
            statement: 语句
            initial_conditions: 初始条件列表
            
        Returns:
            是否在前提中
        """
        return statement in initial_conditions
    
    def create_statement_node(self, original_statement: str, input_entity: str, 
                            output_entity: str, output_parsed: Dict[str, Any], 
                            occurrence_index: int) -> StatementNode:
        """
        创建新的Statement节点
        
        Args:
            original_statement: 原始语句
            input_entity: 输入实体
            output_entity: 输出实体
            output_parsed: 解析后的输出结构
            occurrence_index: 出现索引
            
        Returns:
            新创建的Statement节点
        """
        return StatementNode(
            original_statement=original_statement,
            input_entity=input_entity,
            output_entity=output_entity,
            output_parsed=output_parsed,
            occurrence_count=1,
            is_correct=False,
            is_premise=False,
            node_type="unknown",
            first_occurrence_index=occurrence_index
        )
    
    def find_corresponding_log_node(self, statement: str) -> Optional[Dict[str, Any]]:
        """
        在LoG图中找到对应的节点
        
        Args:
            statement: 完整的语句 (如 "x is B")
            
        Returns:
            对应的LoG节点，如果不存在返回None
        """
        for node in self.log_graph:
            if node.get('output', '') == statement:
                return node
        return None
    
    def illuminate_log_node(self, log_node: Dict[str, Any]):
        """
        点亮LoG节点，并自动点亮其整棵依赖子树
        
        Args:
            log_node: LoG节点
        """
        node_id = log_node.get('output', '')
        if node_id not in self.illuminated_nodes:
            self.illuminated_nodes.add(node_id)
            print(f"[后处理] 点亮LoG节点: {node_id}")
            
            # 找到对应的LogNode对象并点亮整棵依赖子树
            self._illuminate_node_with_subtree(node_id)
    
    def _illuminate_node_with_subtree(self, node_id: str) -> int:
        """
        点亮指定节点的整棵依赖子树（不包括节点本身）
        
        Args:
            node_id: 节点输出ID
            
        Returns:
            新点亮的依赖节点数量
        """
        # 找到对应的LogNode对象
        log_node_obj = self.find_log_node_by_output(node_id)
        if log_node_obj:
            # 点亮整棵依赖子树
            subtree_count = self.auto_illuminate_dependency_subtree(log_node_obj)
            if subtree_count > 0:
                print(f"[后处理]   └─ 自动点亮依赖子树: {subtree_count} 个节点")
            return subtree_count
        return 0
    
    def illuminate_nodes_by_proof_trace(self, proof_trace: Dict[str, Any]):
        """
        基于推理轨迹点亮LoG节点
        
        Args:
            proof_trace: 推理轨迹字典
        """
        if not proof_trace or not proof_trace.get("is_provable", False):
            return
        
        used_premises = proof_trace.get("used_premises", [])
        print(f"[后处理] 基于推理轨迹点亮节点，使用了 {len(used_premises)} 个前提条件:")
        
        # 收集所有使用的前提条件的语句
        premise_statements = set()
        for premise in used_premises:
            premise_statement = premise.get('original', f"{premise['input']} is {premise['output']}")
            premise_statements.add(premise_statement)
            print(f"  - 前提: {premise_statement}")
        
        # 遍历LoG图，找到所有可以通过这些前提条件推导出的节点
        illuminated_count = 0
        for log_node in self.log_graph:
            node_output = log_node.get('output', '')
            
            # 跳过已经点亮的节点
            if node_output in self.illuminated_nodes:
                continue
            
            # 检查这个LoG节点是否可以通过我们使用的前提条件推导出来
            # 构造目标节点
            if ' is ' in node_output:
                parts = node_output.split(' is ', 1)
                if len(parts) == 2:
                    input_part = parts[0].strip()
                    output_part = parts[1].strip()
                    output_parsed = self.reasoning_engine.parse_output_entities(output_part)
                    
                    target_node = {
                        "input": input_part,
                        "output": output_part,
                        "output_parsed": output_parsed,
                        "original": node_output,
                        "type": "log_node_check"
                    }
                    
                    # 使用相同的前提条件检查是否可以推导出这个LoG节点
                    try:
                        is_provable = self.reasoning_engine.is_provable(
                            target_node, used_premises, debug=False
                        )
                        
                        if is_provable:
                            self.illuminated_nodes.add(node_output)
                            illuminated_count += 1
                            print(f"  ✓ 点亮LoG节点: {node_output}")
                            
                            # 点亮整棵依赖子树
                            self._illuminate_node_with_subtree(node_output)
                            
                    except Exception as e:
                        print(f"  ⚠️  检查LoG节点时出错: {e}")
        
        if illuminated_count > 0:
            print(f"[后处理] 通过推理轨迹额外点亮了 {illuminated_count} 个LoG节点")
    
    def print_detailed_proof_trace(self, proof_trace: Dict[str, Any], indent: str = ""):
        """
        打印详细的推理轨迹信息
        
        Args:
            proof_trace: 推理轨迹字典
            indent: 缩进字符串
        """
        if not proof_trace or not proof_trace.get("is_provable", False):
            print(f"{indent}❌ 无法推导")
            return
        
        target = proof_trace.get("target", {})
        used_premises = proof_trace.get("used_premises", [])
        intermediate_steps = proof_trace.get("intermediate_steps", [])
        
        print(f"{indent}🎯 目标: {target.get('original', 'N/A')}")
        print(f"{indent}📊 推理深度: {proof_trace.get('depth', 0)}")
        print(f"{indent}🔧 推理方法: {proof_trace.get('proof_method', 'unknown')}")
        
        if used_premises:
            print(f"{indent}📋 使用的前提条件 ({len(used_premises)} 个):")
            for i, premise in enumerate(used_premises):
                premise_original = premise.get('original', f"{premise.get('input', '?')} is {premise.get('output', '?')}")
                print(f"{indent}  {i+1:2d}. {premise_original}")
        
        if intermediate_steps:
            print(f"{indent}🔗 中间推理步骤 ({len(intermediate_steps)} 个):")
            for i, step in enumerate(intermediate_steps):
                step_target = step.get('target', {})
                step_original = step_target.get('original', f"{step_target.get('input', '?')} is {step_target.get('output', '?')}")
                step_method = step.get('proof_method', 'unknown')
                step_depth = step.get('depth', 0)
                
                print(f"{indent}  步骤 {i+1}: {step_original}")
                print(f"{indent}    方法: {step_method}, 深度: {step_depth}")
                
                # 递归显示子步骤的前提条件
                step_premises = step.get('used_premises', [])
                if step_premises:
                    print(f"{indent}    前提 ({len(step_premises)} 个):")
                    for j, premise in enumerate(step_premises):
                        premise_original = premise.get('original', f"{premise.get('input', '?')} is {premise.get('output', '?')}")
                        print(f"{indent}      {j+1}. {premise_original}")
        
        print(f"{indent}✅ 推理完成")
    
    def auto_illuminate_children(self, parent_node: Dict[str, Any]):
        """
        自动点亮父节点的所有依赖节点（前置节点）
        
        Args:
            parent_node: 父节点
        """
        parent_depth = parent_node.get('depth', 0)
        parent_inputs = parent_node.get('input', [])
        
        if not isinstance(parent_inputs, list):
            return
        
        # 找到所有依赖节点（输出是当前节点输入的节点，深度应该更大）
        dependencies_illuminated = 0
        for input_statement in parent_inputs:
            for log_node in self.log_graph:
                if (log_node.get('output', '') == input_statement and 
                    log_node.get('depth', 0) > parent_depth):  # 修复：依赖节点深度更大
                    
                    dependency_id = log_node.get('output', '')
                    if dependency_id not in self.illuminated_nodes:
                        self.illuminated_nodes.add(dependency_id)
                        dependencies_illuminated += 1
                        print(f"[后处理]   └─ 自动点亮依赖节点: {dependency_id} (深度{log_node.get('depth', 0)})")
                        
                        # 递归点亮更深层的依赖节点
                        self.auto_illuminate_children(log_node)
        
        if dependencies_illuminated > 0:
            print(f"[后处理] 共自动点亮 {dependencies_illuminated} 个依赖节点")
    
    def get_correct_statements_as_premises(self) -> List[Dict[str, Any]]:
        """
        获取正确的Statement作为前提条件
        
        Returns:
            正确Statement的前提格式列表
        """
        premises = []
        for stmt_node in self.statement_list:
            if stmt_node.is_correct:
                premise = {
                    "original": stmt_node.original_statement,
                    "input": stmt_node.input_entity,
                    "output": stmt_node.output_entity,
                    "output_parsed": stmt_node.output_parsed,
                    "type": "correct_statement"
                }
                premises.append(premise)
        return premises
    
    def get_all_statements_as_premises(self) -> List[Dict[str, Any]]:
        """
        获取所有Statement作为前提条件（包括错误的）
        
        Returns:
            所有Statement的前提格式列表
        """
        premises = []
        for stmt_node in self.statement_list:
            premise = {
                "original": stmt_node.original_statement,
                "input": stmt_node.input_entity,
                "output": stmt_node.output_entity,
                "output_parsed": stmt_node.output_parsed,
                "type": "all_statement"
            }
            premises.append(premise)
        return premises
    
    def get_proof_trace_for_node(self, target_node: StatementNode, 
                                 premises_only: bool = True) -> Optional[Dict[str, Any]]:
        """
        获取节点的推理轨迹，使用纯前提条件（不包含其他statement）
        
        Args:
            target_node: 目标节点
            premises_only: 是否只使用前提条件（不包含其他推理出的statement）
            
        Returns:
            推理轨迹字典，如果无法推导则返回None
        """
        if target_node.is_premise:
            # 前提节点没有推理轨迹
            return None
        
        # 构造目标节点
        target = {
            "input": target_node.input_entity,
            "output": target_node.output_entity,
            "output_parsed": target_node.output_parsed,
            "original": target_node.original_statement,
            "type": "target_for_trace"
        }
        
        # 获取前提条件
        if premises_only:
            premises = [
                {
                    "original": stmt.original_statement,
                    "input": stmt.input_entity,
                    "output": stmt.output_entity,
                    "output_parsed": stmt.output_parsed,
                    "type": "premise_only"
                }
                for stmt in self.statement_list if stmt.is_premise
            ]
        else:
            premises = self.get_correct_statements_as_premises()
        
        # 使用推理引擎获取推理轨迹
        try:
            is_provable, proof_trace = self.reasoning_engine.is_provable(
                target, premises, debug=False, return_proof_trace=True
            )
            
            if is_provable:
                return proof_trace
            else:
                return None
                
        except Exception as e:
            print(f"获取推理轨迹时出错: {e}")
            return None
    
    def get_proof_trace_with_all_statements(self, target_node: StatementNode) -> Optional[Dict[str, Any]]:
        """
        使用所有statements（包括错误的）获取节点的推理轨迹
        
        Args:
            target_node: 目标节点
            
        Returns:
            推理轨迹字典，如果无法推导则返回None
        """
        if target_node.is_premise:
            # 前提节点没有推理轨迹
            return None
        
        # 构造目标节点
        target = {
            "input": target_node.input_entity,
            "output": target_node.output_entity,
            "output_parsed": target_node.output_parsed,
            "original": target_node.original_statement,
            "type": "target_for_trace_all"
        }
        
        # 获取所有statements作为前提条件
        all_premises = self.get_all_statements_as_premises()
        
        # 使用推理引擎获取推理轨迹
        try:
            is_provable, proof_trace = self.reasoning_engine.is_provable(
                target, all_premises, debug=False, return_proof_trace=True
            )
            
            if is_provable:
                return proof_trace
            else:
                return None
                
        except Exception as e:
            print(f"获取推理轨迹时出错: {e}")
            return None
    
    def analyze_reasoning_path(self, target_node: StatementNode) -> Dict[str, Any]:
        """
        分析节点的推理路径完整性
        
        Args:
            target_node: 目标节点
            
        Returns:
            路径分析结果
        """
        if target_node.is_premise:
            # 前提节点路径总是有效的
            target_node.path_is_valid = True
            target_node.reasoning_quality = "perfect"
            return {"status": "premise", "dependencies": []}
        
        # 找到推理路径中的所有依赖节点
        dependencies = []
        invalid_deps = []
        
        # 简化版本：检查直接依赖
        # 对于 "x is B"，检查是否存在 "x is A" 和 "A is B" 的路径
        input_entity = target_node.input_entity
        output_entity = target_node.output_entity
        
        # 查找所有以input_entity开头的语句
        input_statements = [s for s in self.statement_list if s.input_entity == input_entity and s != target_node]
        
        for stmt in input_statements:
            dependencies.append(stmt.original_statement)
            if not stmt.is_correct:
                invalid_deps.append(stmt.original_statement)
        
        # 查找连接到output_entity的语句
        connecting_statements = [s for s in self.statement_list 
                               if s.output_entity == output_entity and s.input_entity != input_entity]
        
        for stmt in connecting_statements:
            dependencies.append(stmt.original_statement)
            if not stmt.is_correct:
                invalid_deps.append(stmt.original_statement)
        
        # 更新节点信息
        target_node.dependency_nodes = dependencies
        target_node.invalid_dependencies = invalid_deps
        
        # 判断推理质量
        if len(invalid_deps) == 0:
            target_node.path_is_valid = True
            target_node.reasoning_quality = "perfect"
        elif len(invalid_deps) < len(dependencies):
            target_node.path_is_valid = False
            target_node.reasoning_quality = "partial"
        else:
            target_node.path_is_valid = False
            target_node.reasoning_quality = "invalid"
        
        return {
            "status": "analyzed",
            "dependencies": dependencies,
            "invalid_dependencies": invalid_deps,
            "path_valid": target_node.path_is_valid,
            "quality": target_node.reasoning_quality
        }
    
    def process_nodes(self, normalized_nodes: List[Dict[str, Any]], 
                     initial_conditions: List[str]) -> Dict[str, Any]:
        """
        处理标准化节点列表
        
        Args:
            normalized_nodes: 标准化节点列表
            initial_conditions: 初始条件列表
            
        Returns:
            处理结果
        """
        print(f"\n{'='*60}")
        print(f"[后处理] 开始处理节点")
        print(f"{'='*60}")
        
        print(f"📊 初始状态:")
        print(f"   - 标准化节点数: {len(normalized_nodes)}")
        print(f"   - 初始条件数: {len(initial_conditions)}")
        print(f"   - LoG图节点数: {len(self.log_graph)}")
        
        print(f"\n📋 初始条件列表:")
        for i, condition in enumerate(initial_conditions):
            print(f"   {i+1:2d}. {condition}")
        
        # 首先将初始条件加入statement列表
        print(f"\n🔧 构建初始Statement列表...")
        for i, condition in enumerate(initial_conditions):
            if ' is ' in condition:
                parts = condition.split(' is ', 1)
                if len(parts) == 2:
                    input_part = parts[0].strip()
                    output_part = parts[1].strip()
                    output_parsed = self.reasoning_engine.parse_output_entities(output_part)
                    
                    stmt_node = self.create_statement_node(
                        condition, input_part, output_part, output_parsed, -1
                    )
                    stmt_node.is_premise = True
                    stmt_node.is_correct = True
                    stmt_node.node_type = "premise"
                    
                    self.statement_list.append(stmt_node)
        
        print(f"✅ 初始Statement列表构建完成，包含 {len(self.statement_list)} 个前提节点")
        
        # 遍历每个actual节点
        print(f"\n🔍 开始处理actual节点...")
        actual_nodes_processed = 0
        new_nodes_added = 0
        existing_nodes_updated = 0
        
        for i, node in enumerate(normalized_nodes):
            if node.get("type") == "actual":
                actual_nodes_processed += 1
                current_statement = node.get("original", f"{node['input']} is {node['output']}")
                
                print(f"\n   节点 {actual_nodes_processed}: {current_statement}")
                
                # 步骤1: 检查是否已在statement列表中
                existing_node = self.find_statement_node(current_statement)
                if existing_node:
                    existing_node.occurrence_count += 1
                    existing_nodes_updated += 1
                    print(f"      ↻ 重复节点，计数: {existing_node.occurrence_count}")
                    continue
                
                # 步骤2: 检查是否在题目条件中
                if self.is_in_premises(current_statement, initial_conditions):
                    print(f"      📋 前提条件（已处理）")
                    continue
                
                # 步骤3: 这是一个新的推理节点，需要验证
                print(f"      🔍 验证新节点...")
                
                # 创建目标节点用于验证
                target_node = {
                    "input": node['input'],
                    "output": node['output'],
                    "output_parsed": node['output_parsed'],
                    "original": current_statement,
                    "type": "target"
                }
                
                # 先用正确的statement做条件验证
                correct_premises = self.get_correct_statements_as_premises()
                is_provable_with_correct = self.reasoning_engine.is_provable(
                    target_node, correct_premises, debug=False
                )
                
                is_provable = is_provable_with_correct
                used_correct_only = True  # 记录是否只用了正确的statements
                
                if not is_provable_with_correct:
                    # 如果用正确的推不出，再加上错误的statement试试
                    all_premises = self.get_all_statements_as_premises()
                    is_provable = self.reasoning_engine.is_provable(
                        target_node, all_premises, debug=False
                    )
                    used_correct_only = False  # 标记使用了所有statements
                
                # 创建新的Statement节点
                stmt_node = self.create_statement_node(
                    current_statement, node['input'], node['output'], 
                    node['output_parsed'], i
                )
                
                if is_provable:
                    stmt_node.is_correct = True
                    stmt_node.node_type = "derived"
                    
                    # 在LoG图中找到对应节点并点亮
                    log_node = self.find_corresponding_log_node(current_statement)
                    if log_node:
                        self.illuminate_log_node(log_node)
                        print(f"      ✅ 验证成功 + LoG匹配")
                    else:
                        print(f"      ✅ 验证成功 (LoG中无对应节点)")
                        
                        # 对于不在LoG中的节点，获取其推理轨迹并尝试点亮相关节点
                        print(f"      🔍 获取推理轨迹以点亮隐式节点...")
                        # 使用与验证时相同的前提集合来获取推理轨迹
                        proof_trace = None
                        if used_correct_only:
                            # 如果验证时只用了正确的statements，优先尝试这个
                            proof_trace = self.get_proof_trace_for_node(stmt_node, premises_only=False)
                            if not proof_trace:
                                # 如果找不到，尝试只使用原始前提
                                proof_trace = self.get_proof_trace_for_node(stmt_node, premises_only=True)
                        else:
                            # 如果验证时用了所有statements，需要用相同的集合获取推理轨迹
                            proof_trace = self.get_proof_trace_with_all_statements(stmt_node)
                        
                        if proof_trace:
                            print(f"      📋 推理方法: {proof_trace.get('proof_method', 'unknown')}")
                            print(f"      📋 推理路径: {proof_trace.get('reasoning_path', 'N/A')}")
                            
                            # 如果启用详细前提模式，显示详细的推理轨迹
                            if self.verbose_premise:
                                self.print_detailed_proof_trace(proof_trace, indent="        ")
                            
                            self.illuminate_nodes_by_proof_trace(proof_trace)
                        else:
                            print(f"      ⚠️  无法获取推理轨迹")
                else:
                    stmt_node.is_correct = False
                    stmt_node.node_type = "hallucination"
                    print(f"      ❌ 验证失败 - 幻觉节点")
                
                # 无论正确与否都加入Statement列表
                self.statement_list.append(stmt_node)
                new_nodes_added += 1
        
        print(f"\n🔍 分析推理路径完整性...")
        
        # 对所有非前提节点分析推理路径
        path_analysis_results = []
        perfect_reasoning_count = 0
        partial_reasoning_count = 0
        invalid_reasoning_count = 0
        
        for stmt_node in self.statement_list:
            if not stmt_node.is_premise:
                analysis_result = self.analyze_reasoning_path(stmt_node)
                path_analysis_results.append(analysis_result)
                
                if stmt_node.reasoning_quality == "perfect":
                    perfect_reasoning_count += 1
                elif stmt_node.reasoning_quality == "partial":
                    partial_reasoning_count += 1
                else:
                    invalid_reasoning_count += 1
        
        print(f"\n{'='*60}")
        print(f"[后处理] 处理完成")
        print(f"{'='*60}")
        
        print(f"📈 处理统计:")
        print(f"   - 处理的actual节点: {actual_nodes_processed}")
        print(f"   - 新增节点: {new_nodes_added}")
        print(f"   - 更新已存在节点: {existing_nodes_updated}")
        print(f"   - 点亮的LoG节点: {len(self.illuminated_nodes)}")
        
        print(f"\n🎯 推理质量统计:")
        print(f"   - 完美推理: {perfect_reasoning_count} (节点正确 + 路径正确)")
        print(f"   - 部分推理: {partial_reasoning_count} (节点正确 + 路径部分错误)")
        print(f"   - 无效推理: {invalid_reasoning_count} (节点错误或路径完全错误)")
        
        # 设置初始前提条件并触发LoG依赖分析
        self.set_initial_premises(initial_conditions)
        self.print_log_dependency_summary()
        
        # 执行高级节点点亮算法
        correct_statements = [s for s in self.statement_list if s.is_correct]
        self.illuminate_nodes_by_advanced_matching(correct_statements)
        
        # 打印最终的Statement列表
        self.print_statement_summary()
        
        # 计算所有评估指标
        metrics = self.calculate_all_metrics()
        
        return {
            "total_nodes_processed": actual_nodes_processed,
            "new_nodes_added": new_nodes_added,
            "existing_nodes_updated": existing_nodes_updated,
            "statement_list_size": len(self.statement_list),
            "illuminated_log_nodes": len(self.illuminated_nodes),
            "correct_statements": len([s for s in self.statement_list if s.is_correct]),
            "incorrect_statements": len([s for s in self.statement_list if not s.is_correct]),
            "reasoning_quality": {
                "perfect_reasoning": perfect_reasoning_count,
                "partial_reasoning": partial_reasoning_count,
                "invalid_reasoning": invalid_reasoning_count
            },
            "path_analysis_results": path_analysis_results,
            "evaluation_metrics": metrics
        }
    
    def print_statement_summary(self):
        """打印Statement列表摘要"""
        print(f"\n📊 === Statement列表分析 ===")
        
        premise_count = len([s for s in self.statement_list if s.is_premise])
        derived_count = len([s for s in self.statement_list if s.node_type == "derived"])
        hallucination_count = len([s for s in self.statement_list if s.node_type == "hallucination"])
        
        print(f"📈 节点统计:")
        print(f"   - 总节点数: {len(self.statement_list)}")
        print(f"   - 前提节点: {premise_count}")
        print(f"   - 推理节点: {derived_count}")
        print(f"   - 幻觉节点: {hallucination_count}")
        print(f"   - 点亮LoG节点: {len(self.illuminated_nodes)}")
        
        if hallucination_count > 0:
            print(f"\n❌ 幻觉节点详情:")
            for i, stmt_node in enumerate(self.statement_list):
                if stmt_node.node_type == "hallucination":
                    print(f"   - {stmt_node.original_statement} (出现{stmt_node.occurrence_count}次)")
        
        if derived_count > 0:
            print(f"\n✅ 推理节点详情:")
            for i, stmt_node in enumerate(self.statement_list):
                if stmt_node.node_type == "derived":
                    quality_emoji = "🟢" if stmt_node.reasoning_quality == "perfect" else "🟡" if stmt_node.reasoning_quality == "partial" else "🔴"
                    print(f"   {quality_emoji} {stmt_node.original_statement} (出现{stmt_node.occurrence_count}次, 质量:{stmt_node.reasoning_quality})")
                    
                    if stmt_node.invalid_dependencies:
                        print(f"      ⚠️  无效依赖: {stmt_node.invalid_dependencies}")
        
        if len(self.illuminated_nodes) > 0:
            print(f"\n🔥 点亮的LoG节点:")
            for node_id in self.illuminated_nodes:
                print(f"   - {node_id}")
        else:
            print(f"\n⚠️  未点亮任何LoG节点")
    
    def calculate_coverage_metrics(self) -> Dict[str, Any]:
        """
        计算Coverage指标（类似召回率）
        
        Returns:
            Coverage指标结果
        """
        print(f"\n🎯 计算Coverage指标...")
        
        # 计算LoG图的最大深度
        max_log_depth = max([node.get('depth', 0) for node in self.log_graph]) if self.log_graph else 0
        
        # 1.1 深度Coverage - 基于各推理层点亮比例计算最大达到的层级
        # 首先统计各深度的点亮情况
        depth_stats = {}
        for log_node in self.log_graph:
            depth = log_node.get('depth', 0)
            node_output = log_node.get('output', '')
            
            if depth not in depth_stats:
                depth_stats[depth] = {'total': 0, 'illuminated': 0}
            
            depth_stats[depth]['total'] += 1
            if node_output in self.illuminated_nodes:
                depth_stats[depth]['illuminated'] += 1
        
        # 找到最高的有节点点亮的层级（转换为层级编号）
        max_layer_reached = 0
        deepest_illuminated_node = None
        max_depth_reached = 0
        
        for depth in sorted(depth_stats.keys(), reverse=True):
            stats = depth_stats[depth]
            if stats['illuminated'] > 0:
                layer_number = max_log_depth - depth + 1
                if layer_number > max_layer_reached:
                    max_layer_reached = layer_number
                    max_depth_reached = depth
                    
                    # 找到该深度的一个点亮节点作为代表
                    for log_node in self.log_graph:
                        if (log_node.get('depth', 0) == depth and 
                            log_node.get('output', '') in self.illuminated_nodes):
                            deepest_illuminated_node = log_node.get('output', '')
                            break
        
        max_layer_total = max_log_depth if max_log_depth > 0 else 0  # 总层级数（不包括前提）
        depth_coverage_ratio = max_layer_reached / max_layer_total if max_layer_total > 0 else 0
        
        print(f"   深度Coverage: {max_layer_reached}/{max_layer_total} = {depth_coverage_ratio:.2%}")
        if deepest_illuminated_node:
            print(f"   最深点亮节点: {deepest_illuminated_node} (第{max_layer_reached}层)")
        
        # 1.2 节点Coverage - 标准LoG被点亮的节点比例
        total_log_nodes = len(self.log_graph)
        illuminated_count = len(self.illuminated_nodes)
        node_coverage_ratio = illuminated_count / total_log_nodes if total_log_nodes > 0 else 0
        
        print(f"   节点Coverage: {illuminated_count}/{total_log_nodes} = {node_coverage_ratio:.2%}")
        
        # 1.3 前提条件Coverage - 统计初始条件的覆盖情况
        # 前提条件的"点亮"指的是它们在推理过程中被用到了（出现次数>1）
        premise_statements = [stmt for stmt in self.statement_list if stmt.is_premise]
        illuminated_premise_count = 0
        
        # 统计出现次数大于1的前提节点（说明在推理中被引用了）
        for premise_stmt in premise_statements:
            if premise_stmt.occurrence_count > 1:
                illuminated_premise_count += 1
        
        premise_coverage_ratio = illuminated_premise_count / len(premise_statements) if premise_statements else 0
        print(f"   前提条件Coverage: {illuminated_premise_count}/{len(premise_statements)} = {premise_coverage_ratio:.2%}")
        
        # 1.4 各推理层点亮比例（按照倒序深度显示）
        print(f"   各推理层点亮比例:")
        # 按深度倒序排列，最大深度为第1层
        for depth in sorted(depth_stats.keys(), reverse=True):
            stats = depth_stats[depth]
            ratio = stats['illuminated'] / stats['total'] if stats['total'] > 0 else 0
            layer_number = max_log_depth - depth + 1
            print(f"     第{layer_number}层: {stats['illuminated']}/{stats['total']} = {ratio:.2%}")
        
        return {
            "depth_coverage": {
                "max_layer_reached": max_layer_reached,
                "max_layer_total": max_layer_total,
                "max_depth_reached": max_depth_reached,  # 保留原始深度信息
                "max_log_depth": max_log_depth,
                "ratio": depth_coverage_ratio,
                "deepest_node": deepest_illuminated_node
            },
            "node_coverage": {
                "illuminated_count": illuminated_count,
                "total_log_nodes": total_log_nodes,
                "ratio": node_coverage_ratio
            },
            "premise_coverage": {
                "illuminated_premise_count": illuminated_premise_count,
                "total_premise_statements": len(premise_statements),
                "ratio": premise_coverage_ratio
            },
            "depth_distribution": depth_stats,
            "layer_distribution": {
                f"layer_{max_log_depth - depth + 1}": {
                    "depth": depth,
                    "total": stats['total'],
                    "illuminated": stats['illuminated'],
                    "ratio": stats['illuminated'] / stats['total'] if stats['total'] > 0 else 0
                }
                for depth, stats in depth_stats.items()
            }
        }
    
    def calculate_precision_metrics(self) -> Dict[str, Any]:
        """
        计算Precision指标
        
        Returns:
            Precision指标结果
        """
        print(f"\n🎯 计算Precision指标...")
        
        # 只考虑非前提节点
        non_premise_nodes = [s for s in self.statement_list if not s.is_premise]
        total_derived_nodes = len(non_premise_nodes)
        
        if total_derived_nodes == 0:
            print(f"   没有推理节点可供分析")
            return {
                "error_rate": {"provable_count": 0, "total_count": 0, "ratio": 0},
                "strict_error_rate": {"valid_count": 0, "total_count": 0, "ratio": 0}
            }
        
        # 2.1 Error Rate - 推出节点有多少是is_provable的
        provable_count = len([s for s in non_premise_nodes if s.is_correct])
        error_rate = 1 - (provable_count / total_derived_nodes)
        
        print(f"   Error Rate: {total_derived_nodes - provable_count}/{total_derived_nodes} = {error_rate:.2%}")
        print(f"   可推导节点: {provable_count}/{total_derived_nodes}")
        
        # 2.2 Strict Error Rate - 节点is_provable且所有祖先都正确
        strict_valid_count = 0
        
        for stmt_node in non_premise_nodes:
            if stmt_node.is_correct:  # 节点本身可推导
                # 检查所有依赖节点是否都正确
                all_deps_valid = True
                for dep_statement in stmt_node.dependency_nodes:
                    dep_node = self.find_statement_node(dep_statement)
                    if dep_node and not dep_node.is_correct:
                        all_deps_valid = False
                        break
                
                if all_deps_valid:
                    strict_valid_count += 1
        
        strict_error_rate = 1 - (strict_valid_count / total_derived_nodes)
        
        print(f"   Strict Error Rate: {total_derived_nodes - strict_valid_count}/{total_derived_nodes} = {strict_error_rate:.2%}")
        print(f"   严格有效节点: {strict_valid_count}/{total_derived_nodes}")
        
        # 详细分析
        print(f"\n   详细分析:")
        perfect_count = len([s for s in non_premise_nodes if s.reasoning_quality == "perfect"])
        partial_count = len([s for s in non_premise_nodes if s.reasoning_quality == "partial"])
        invalid_count = len([s for s in non_premise_nodes if s.reasoning_quality == "invalid"])
        
        print(f"     完美推理: {perfect_count} ({perfect_count/total_derived_nodes:.2%})")
        print(f"     部分推理: {partial_count} ({partial_count/total_derived_nodes:.2%})")
        print(f"     无效推理: {invalid_count} ({invalid_count/total_derived_nodes:.2%})")
        
        return {
            "error_rate": {
                "provable_count": provable_count,
                "total_count": total_derived_nodes,
                "ratio": error_rate
            },
            "strict_error_rate": {
                "valid_count": strict_valid_count,
                "total_count": total_derived_nodes,
                "ratio": strict_error_rate
            },
            "quality_distribution": {
                "perfect": perfect_count,
                "partial": partial_count,
                "invalid": invalid_count,
                "total": total_derived_nodes
            }
        }
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        计算所有指标
        
        Returns:
            完整的指标结果
        """
        print(f"\n{'='*60}")
        print(f"📊 计算评估指标")
        print(f"{'='*60}")
        
        coverage_metrics = self.calculate_coverage_metrics()
        precision_metrics = self.calculate_precision_metrics()
        
        return {
            "coverage": coverage_metrics,
            "precision": precision_metrics,
            "summary": {
                "total_statements": len(self.statement_list),
                "premise_statements": len([s for s in self.statement_list if s.is_premise]),
                "derived_statements": len([s for s in self.statement_list if not s.is_premise]),
                "illuminated_log_nodes": len(self.illuminated_nodes),
                "total_log_nodes": len(self.log_graph)
            }
        }


class StepByStepEvaluator2:
    def __init__(self, api_key: str, model_name: str = "deepseek-reasoner", 
                 api_base: str = "https://api.deepseek.com/beta", debug_mode: bool = False,
                 llm_debug_mode: bool = False, api_mode: str = "commercial",
                 verbose_premise: bool = False, single_record_debug_mode: bool = False,
                 max_records: int = None, record_index: int = 0):
        """
        初始化逐步评估器2.0
        
        Args:
            api_key: API密钥
            model_name: 模型名称
            api_base: API基础URL
            debug_mode: 调试模式（跳过所有API调用）
            llm_debug_mode: LLM调试模式（只做提取和记录）
            api_mode: API模式，"commercial"或"vllm"
            verbose_premise: 详细输出模式，显示每个节点的前提信息
            single_record_debug_mode: 单条记录调试模式，只处理第一条数据
            max_records: 遍历模式下最大处理记录数，None表示处理所有记录
            record_index: 单条记录调试模式下指定处理哪一条记录（从0开始）
        """
        self.debug_mode = debug_mode
        self.llm_debug_mode = llm_debug_mode
        self.model_name = model_name
        self.api_mode = api_mode
        self.verbose_premise = verbose_premise
        self.single_record_debug_mode = single_record_debug_mode
        self.max_records = max_records
        self.record_index = record_index
        
        if not debug_mode:
            # 只在非调试模式下导入和初始化API客户端
            from apply_llm import DeepSeekAPIClient
            self.client = DeepSeekAPIClient(
                api_key=api_key,
                model_name=model_name,
                api_base=api_base,
                max_new_tokens=5000
            )
        else:
            self.client = None
            
        self.extract_prompt_template = self.load_extract_prompt()
        self.statement_processor = StatementProcessor()
        self.reasoning_engine = LogicalReasoningEngine(max_depth=1000, timeout=600)
        self.post_processor = PostProcessor(self.reasoning_engine, verbose_premise=verbose_premise)
        
        # 创建LLM提取结果缓存目录
        self.cache_dir = "./LLM_extract_node"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_extract_prompt(self) -> str:
        """加载提取提示模板"""
        try:
            with open('extract_prompt_2.txt', 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError("找不到 extract_prompt_2.txt 文件")
    
    def load_evaluation_log(self, log_path: str) -> Dict[str, Any]:
        """加载评估日志文件"""
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到日志文件: {log_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析错误: {e}")
    
    def load_log_graph_data(self, input_file_path: str) -> Dict[int, List[Dict[str, Any]]]:
        """
        从LoG数据文件中加载图数据
        
        Args:
            input_file_path: LoG数据文件路径 (如 ./generated_data/LoG_5.jsonl)
            
        Returns:
            字典，key为example的id，value为对应的graph数据
        """
        graph_data = {}
        try:
            with open(input_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        example_id = example.get('id', -1)
                        graph = example.get('graph', [])
                        graph_data[example_id] = graph
            
            print(f"从 {input_file_path} 加载了 {len(graph_data)} 个例子的图数据")
            return graph_data
            
        except FileNotFoundError:
            print(f"警告: 找不到LoG数据文件: {input_file_path}")
            return {}
        except Exception as e:
            print(f"加载LoG图数据时出错: {e}")
            return {}
    
    def get_cache_file_path(self, log_path: str, record_index: int) -> str:
        """
        获取缓存文件路径
        
        Args:
            log_path: 原始日志文件路径
            record_index: 记录索引
            
        Returns:
            缓存文件路径
        """
        import hashlib
        
        # 基于日志文件路径和记录索引生成缓存文件名
        log_file_name = os.path.basename(log_path).replace('.json', '')
        cache_file_name = f"{log_file_name}_record_{record_index}.json"
        return os.path.join(self.cache_dir, cache_file_name)
    
    def load_cached_extraction(self, cache_path: str) -> Optional[Dict[str, Any]]:
        """
        加载缓存的提取结果
        
        Args:
            cache_path: 缓存文件路径
            
        Returns:
            缓存的结果，如果不存在返回None
        """
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                print(f"📁 从缓存加载提取结果: {os.path.basename(cache_path)}")
                return cached_data
        except Exception as e:
            print(f"⚠️  加载缓存失败: {e}")
        
        return None
    
    def save_extraction_to_cache(self, cache_path: str, extraction_data: Dict[str, Any]):
        """
        保存提取结果到缓存
        
        Args:
            cache_path: 缓存文件路径
            extraction_data: 提取数据
        """
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(extraction_data, f, ensure_ascii=False, indent=2)
            print(f"💾 保存提取结果到缓存: {os.path.basename(cache_path)}")
        except Exception as e:
            print(f"⚠️  保存缓存失败: {e}")
    
    def split_reasoning_text(self, text: str) -> tuple[List[str], List[str]]:
        """
        将推理文本按句子分割，同时保存分隔符信息
        
        Args:
            text: 推理文本
            
        Returns:
            (句子列表, 分隔符列表)
        """
        # 使用多个分隔符进行分割：句号、问号、感叹号、连续换行符
        parts = re.split(r'([.!?。？！]|\n\n+)', text)
        
        sentences = []
        separators = []
        
        i = 0
        while i < len(parts):
            if i + 1 < len(parts):
                content = parts[i].strip()
                separator = parts[i + 1]
                
                if content:  # 只处理有内容的句子
                    if re.match(r'\n\n+', separator):
                        sentences.append(content)
                        separators.append('\n\n')
                    elif separator in '.!?。？！':
                        sentences.append(content + separator)
                        separators.append(' ')
                i += 2
            else:
                sentence = parts[i].strip()
                if sentence:
                    sentences.append(sentence)
                    separators.append('')
                i += 1
        
        return sentences, separators
    
    
    def create_analysis_prompt(self, current_sentence: str) -> str:
        """
        创建分析提示，将当前句子插入到提取提示模板中
        
        Args:
            current_sentence: 当前句子
            
        Returns:
            完整的分析提示
        """
        # 使用模板替换
        template = self.extract_prompt_template
        prompt = template.replace("{current_sentence}", current_sentence)
        
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
            # 尝试从代码块中提取JSON
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            match = re.search(json_pattern, response_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # 尝试提取任何JSON对象
            json_pattern = r'\{[^}]*"statements"[^}]*\}'
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
    
    def extract_statements_from_sentence(self, current_sentence: str, sentence_index: int, total_sentences: int) -> Dict[str, Any]:
        """
        从单个句子中提取statements
        
        Args:
            current_sentence: 当前句子
            sentence_index: 句子索引
            total_sentences: 总句子数
            
        Returns:
            提取结果，包含原句子、提示、响应和提取的statements
        """
        print(f"[{sentence_index+1}/{total_sentences}] 输入: {current_sentence[:100]}...")
        
        # 创建分析提示（只包含当前句子）
        prompt = self.create_analysis_prompt(current_sentence)
        
        try:
            if self.debug_mode:
                # 完全调试模式：不调用API
                response_text = "[DEBUG MODE - NO API CALL]"
                thinking = ""
                json_result = {"statements": ["debug statement 1", "debug statement 2"]}
                print(f"结果: 调试模式 - {len(json_result.get('statements', []))} 个statements")
            else:
                # 调用API
                response = self.client.get_response(prompt, temperature=0.0)
                
                # 提取响应文本
                if self.api_mode == "commercial":
                    response_text = response['choices'][0]['message']['content']
                    thinking = response['choices'][0]['message'].get('reasoning_content', '')
                elif self.api_mode == "vllm":
                    response_text = response['choices'][0]['text'].strip()
                    thinking = ''
                else:
                    # 默认使用commercial模式
                    response_text = response['choices'][0]['message']['content']
                    thinking = response['choices'][0]['message'].get('reasoning_content', '')
                
                # 提取JSON结果
                json_result = self.extract_json_from_response(response_text)
                
                # 打印分析结果
                if "error" not in json_result:
                    statements = json_result.get('statements', [])
                    print(f"结果: 提取到 {len(statements)} 个statements")
                    if statements:
                        for i, stmt in enumerate(statements):
                            if isinstance(stmt, dict):
                                stmt_type = stmt.get('type', 'unknown')
                                stmt_text = stmt.get('statement', str(stmt))
                                print(f"  {i+1}. [{stmt_type}] {stmt_text}")
                            else:
                                print(f"  {i+1}. [legacy] {stmt}")
                else:
                    print(f"提取失败: {json_result.get('error', '未知错误')}")
            
            return {
                "sentence": current_sentence,
                "sentence_index": sentence_index,
                "prompt": prompt,
                "response_text": response_text,
                "thinking": thinking,
                "json_result": json_result,
                "success": "error" not in json_result
            }
            
        except Exception as e:
            print(f"提取statements时发生错误: {e}")
            return {
                "sentence": current_sentence,
                "sentence_index": sentence_index,
                "prompt": prompt,
                "error": str(e),
                "success": False
            }
    
    def evaluate_reasoning_process(self, log_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        评估推理过程
        
        Args:
            log_path: 日志文件路径
            output_path: 输出文件路径，如果为None则自动生成
            
        Returns:
            评估结果
        """
        print(f"开始评估推理过程（版本2.0）...")
        print(f"日志文件: {log_path}")
        print(f"LLM调试模式: {self.llm_debug_mode}")
        
        # 加载日志
        log_data = self.load_evaluation_log(log_path)
        details = log_data.get('details', [])
        
        if not details:
            raise ValueError("日志文件中没有找到details数据")
        
        print(f"找到 {len(details)} 条记录")
        
        # 尝试从input_file字段获取对应的LoG数据文件路径
        input_file = log_data.get('input_file', '')
        graph_data_dict = {}
        
        if input_file:
            print(f"检测到input_file: {input_file}")
            graph_data_dict = self.load_log_graph_data(input_file)
        else:
            print("未找到input_file字段，尝试推断LoG数据文件路径")
            # 从log_path推断对应的LoG数据文件
            if 'LoG_5' in log_path:
                graph_data_dict = self.load_log_graph_data('./generated_data/LoG_5.jsonl')
            elif 'LoG_4' in log_path:
                graph_data_dict = self.load_log_graph_data('./generated_data/LoG_4.jsonl')
            elif 'LoG_6' in log_path:
                graph_data_dict = self.load_log_graph_data('./generated_data/LoG_6.jsonl')
            elif 'LoG_7' in log_path:
                graph_data_dict = self.load_log_graph_data('./generated_data/LoG_7.jsonl')
            elif 'LoG_8' in log_path:
                graph_data_dict = self.load_log_graph_data('./generated_data/LoG_8.jsonl')
            elif 'LoG_9' in log_path:
                graph_data_dict = self.load_log_graph_data('./generated_data/LoG_9.jsonl')
            elif 'LoG_10' in log_path:
                graph_data_dict = self.load_log_graph_data('./generated_data/LoG_10.jsonl')
            elif 'LoG_11' in log_path:
                graph_data_dict = self.load_log_graph_data('./generated_data/LoG_11.jsonl')
            elif 'LoG_12' in log_path:
                graph_data_dict = self.load_log_graph_data('./generated_data/LoG_12.jsonl')
            else:
                print(f"未知的LoG文件类型: {log_path}")
        
        # 评估结果
        evaluation_results = {
            "log_path": log_path,
            "total_records": len(details),
            "model_name": self.model_name,
            "llm_debug_mode": self.llm_debug_mode,
            "processed_records": [],
            "summary": {
                "total_sentences": 0,
                "successful_extractions": 0,
                "failed_extractions": 0,
                "total_statements": 0
            }
        }
        
        # 根据模式决定处理哪些记录
        if self.single_record_debug_mode:
            # 单条记录调试模式
            if self.record_index >= len(details):
                raise ValueError(f"指定的记录索引 {self.record_index} 超出范围，总记录数: {len(details)}")
            
            print(f"\n单条记录调试模式：处理第 {self.record_index + 1} 条记录（索引 {self.record_index}）...")
            records_to_process = [details[self.record_index]]
        else:
            # 遍历模式
            if self.max_records is not None:
                max_records = min(self.max_records, len(details))
                print(f"\n处理前 {max_records} 条记录（总共 {len(details)} 条）...")
                records_to_process = details[:max_records]
            else:
                print(f"\n处理所有 {len(details)} 条记录...")
                records_to_process = details
        
        # 用于收集所有记录的指标
        all_metrics = []
        
        # 处理每条记录
        for record_idx, record in enumerate(records_to_process):
            print(f"\n{'='*80}")
            print(f"处理记录 {record_idx + 1}/{len(records_to_process)}")
            print(f"{'='*80}")
            
            record_index = record.get('index', 0)
            print(f"记录索引: {record_index}")
            print(f"问题状态: {record.get('status', 'N/A')}")
            
            # 处理单条记录
            try:
                processed_record = self.process_single_record(
                    record, log_path, graph_data_dict, evaluation_results["summary"]
                )
                evaluation_results["processed_records"].append(processed_record)
                
                # 收集指标用于计算平均值
                if processed_record and "evaluation_metrics" in processed_record:
                    all_metrics.append(processed_record["evaluation_metrics"])
                    
            except Exception as e:
                print(f"处理记录 {record_index} 时出错: {e}")
                continue
        
        # 计算平均指标
        if all_metrics:
            average_metrics = self.calculate_average_metrics(all_metrics)
            evaluation_results["average_metrics"] = average_metrics
            self.print_average_metrics_summary(average_metrics, len(all_metrics))
        else:
            print("没有有效的指标数据用于计算平均值")
        
        # 保存结果
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(log_path))[0]
            output_path = f"step_by_step_evaluation_2_{base_name}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {output_path}")
        
        return evaluation_results
    
    def process_single_record(self, record: Dict[str, Any], log_path: str, 
                            graph_data_dict: Dict[int, List[Dict[str, Any]]], 
                            summary_stats: Dict[str, int]) -> Dict[str, Any]:
        """
        处理单条记录
        
        Args:
            record: 记录数据
            log_path: 日志文件路径
            graph_data_dict: 图数据字典
            summary_stats: 汇总统计信息
            
        Returns:
            处理后的记录数据
        """
        record_index = record.get('index', 0)
        
        # 检查是否有缓存的提取结果
        cache_path = self.get_cache_file_path(log_path, record_index)
        cached_result = self.load_cached_extraction(cache_path)
        
        if cached_result:
            # 使用缓存的结果
            sentence_extractions = cached_result.get('sentence_extractions', [])
            all_statements = cached_result.get('all_statements', [])
            sentences = cached_result.get('sentences', [])
            initial_conditions = cached_result.get('initial_conditions', [])
            reasoning_text = cached_result.get('reasoning_text', '')
            thinking_text = cached_result.get('thinking_text', '')
            
            print(f"✅ 使用缓存结果:")
            print(f"   - 句子数: {len(sentences)}")
            print(f"   - 提取的语句数: {len(all_statements)}")
            print(f"   - 初始条件数: {len(initial_conditions)}")
            
        else:
            # 执行LLM提取
            print(f"🔄 执行LLM提取（未找到缓存）...")
            
            # 提取初始条件
            original_question = record.get('original_question', '')
            initial_conditions = self.statement_processor.extract_initial_conditions(original_question)
            print(f"提取到 {len(initial_conditions)} 个初始条件:")
            for i, condition in enumerate(initial_conditions):
                print(f"  {i+1}. {condition}")
            
            # 获取推理过程文本
            reasoning_text = record.get('full_response', '')
            thinking_text = record.get('thinking', '')

            if len(thinking_text) > 0:
                reasoning_text = thinking_text
            
            if not reasoning_text:
                print("警告: 没有找到推理过程文本")
                return None
            
            print(f"推理文本长度: {len(reasoning_text)} 字符")
            
            # 分割句子
            sentences, separators = self.split_reasoning_text(reasoning_text)
            print(f"分割得到 {len(sentences)} 个句子")
            
            # 提取每个句子的statements
            sentence_extractions = []
            all_statements = []
            
            for i, sentence in enumerate(sentences):
                print(f"\n--- 处理句子 {i+1}/{len(sentences)} ---")
                
                extraction = self.extract_statements_from_sentence(sentence, i, len(sentences))
                sentence_extractions.append(extraction)
                
                # 收集所有statements
                if extraction["success"]:
                    statements = extraction["json_result"].get("statements", [])
                    # 处理新格式的statements（带type字段）
                    for stmt in statements:
                        if isinstance(stmt, dict):
                            all_statements.append(stmt)
                        else:
                            # 兼容旧格式
                            all_statements.append({"type": "legacy", "statement": stmt})
                    
                    # 更新统计
                    summary_stats["successful_extractions"] += 1
                    summary_stats["total_statements"] += len(statements)
                else:
                    summary_stats["failed_extractions"] += 1
                
                summary_stats["total_sentences"] += 1
            
            # 保存提取结果到缓存
            extraction_cache_data = {
                "record_index": record_index,
                "initial_conditions": initial_conditions,
                "reasoning_text": reasoning_text,
                "thinking_text": thinking_text,
                "sentences": sentences,
                "sentence_extractions": sentence_extractions,
                "all_statements": all_statements,
                "extraction_timestamp": time.time()
            }
            
            self.save_extraction_to_cache(cache_path, extraction_cache_data)
        
        print(f"\n=== Statement提取完成 ===")
        print(f"总句子数: {len(sentences)}")
        print(f"成功提取: {len([e for e in sentence_extractions if e.get('success', False)])}")
        print(f"失败提取: {len([e for e in sentence_extractions if not e.get('success', False)])}")
        print(f"总statements数: {len(all_statements)}")
        
        # 清洗statements格式
        print(f"\n=== 开始清洗Statement格式 ===")
        original_count = len(all_statements)
        cleaned_statements = self.statement_processor.clean_statements_list(all_statements)
        cleaned_count = len(cleaned_statements)
        
        print(f"原始statements数: {original_count}")
        print(f"清洗后statements数: {cleaned_count}")
        print(f"过滤掉的statements数: {original_count - cleaned_count}")
        
        if cleaned_count != original_count:
            print("清洗后的statements:")
            for i, stmt in enumerate(cleaned_statements):
                stmt_type = stmt.get('type', 'unknown')
                stmt_text = stmt.get('statement', '')
                print(f"  {i+1}. [{stmt_type}] {stmt_text}")
        
        # 更新all_statements为清洗后的版本
        all_statements = cleaned_statements
        
        # 标准化并解析为节点格式
        print(f"\n=== 开始标准化和解析节点 ===")
        normalized_nodes = self.statement_processor.normalize_and_parse_statements(all_statements)
        normalized_count = len(normalized_nodes)
        
        print(f"清洗后statements数: {cleaned_count}")
        print(f"标准化后节点数: {normalized_count}")
        print(f"过滤掉的无效实体数: {cleaned_count - normalized_count}")
        
        if normalized_count > 0:
            print("标准化后的节点:")
            for i, node in enumerate(normalized_nodes):
                node_type = node.get('type', 'unknown')
                original = node.get('original', '')
                input_part = node.get('input', '')
                output_part = node.get('output', '')
                print(f"  {i+1}. [{node_type}] {input_part} → {output_part} (原始: {original})")
        
        # 执行新的后处理逻辑
        print(f"\n开始新的后处理...")
        
        # 从记录中提取LoG图数据，如果没有则从graph_data_dict中获取
        graph_data = record.get('graph', [])
        if not graph_data and graph_data_dict:
            # 尝试从graph_data_dict中获取对应的图数据
            record_index = record.get('index', -1)
            if record_index in graph_data_dict:
                graph_data = graph_data_dict[record_index]
                print(f"[后处理] 从LoG数据文件中获取到图数据，节点数: {len(graph_data)}")
        
        # 为每条记录创建新的PostProcessor实例，避免状态混乱
        record_post_processor = PostProcessor(self.reasoning_engine, verbose_premise=self.verbose_premise)
        
        if graph_data:
            record_post_processor.load_log_graph(graph_data)
        else:
            print("[后处理] 警告: 未找到LoG图数据")
        
        # 执行后处理
        post_processing_result = record_post_processor.process_nodes(
            normalized_nodes, initial_conditions
        )
        
        # 为了向后兼容，也保留旧的后处理结果
        if self.llm_debug_mode:
            print(f"LLM调试模式：使用简化的旧后处理作为补充")
            legacy_result = self.statement_processor.process_statements(all_statements, debug_mode=True)
        else:
            self.statement_processor.condition_list = initial_conditions
            legacy_result = self.statement_processor.process_statements(all_statements)
        
        post_processing_result["legacy_result"] = legacy_result
        
        # 保存处理的记录
        processed_record = {
            "original_record": {
                "index": record.get('index'),
                "status": record.get('status'),
                "question": record.get('extracted_question', ''),
                "expected": record.get('expected'),
                "predicted": record.get('predicted')
            },
            "initial_conditions": initial_conditions,
            "reasoning_text": reasoning_text,
            "thinking_text": thinking_text,
            "sentences": sentences,
            "sentence_extractions": sentence_extractions,
            "all_statements": all_statements,
            "cleaned_statements": cleaned_statements,
            "normalized_nodes": normalized_nodes,
            "post_processing_result": post_processing_result,
            "statement_list": [
                {
                    "original_statement": stmt.original_statement,
                    "input_entity": stmt.input_entity,
                    "output_entity": stmt.output_entity,
                    "occurrence_count": stmt.occurrence_count,
                    "is_correct": stmt.is_correct,
                    "is_premise": stmt.is_premise,
                    "node_type": stmt.node_type,
                    "first_occurrence_index": stmt.first_occurrence_index,
                    "path_is_valid": stmt.path_is_valid,
                    "reasoning_quality": stmt.reasoning_quality,
                    "dependency_nodes": stmt.dependency_nodes,
                    "invalid_dependencies": stmt.invalid_dependencies
                }
                for stmt in record_post_processor.statement_list
            ],
            "illuminated_log_nodes": list(record_post_processor.illuminated_nodes),
            "evaluation_metrics": post_processing_result.get("evaluation_metrics", {})
        }
        
        return processed_record
    
    def calculate_average_metrics(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        计算所有记录的平均指标
        
        Args:
            all_metrics: 所有记录的指标列表
            
        Returns:
            平均指标结果
        """
        if not all_metrics:
            return {}
        
        # 初始化累加器
        coverage_totals = {
            "depth_coverage_ratio": 0,
            "node_coverage_ratio": 0,
            "premise_coverage_ratio": 0,
            "max_layer_reached": 0,
            "illuminated_count": 0,
            "total_log_nodes": 0
        }
        
        # 用于收集各层分布数据
        layer_distributions = []
        
        precision_totals = {
            "error_rate": 0,
            "strict_error_rate": 0,
            "provable_count": 0,
            "total_count": 0,
            "perfect_count": 0,
            "partial_count": 0,
            "invalid_count": 0
        }
        
        summary_totals = {
            "total_statements": 0,
            "premise_statements": 0,
            "derived_statements": 0
        }
        
        # 累加所有指标
        for metrics in all_metrics:
            coverage = metrics.get("coverage", {})
            precision = metrics.get("precision", {})
            summary = metrics.get("summary", {})
            
            # Coverage指标
            if "depth_coverage" in coverage:
                depth_cov = coverage["depth_coverage"]
                coverage_totals["depth_coverage_ratio"] += depth_cov.get("ratio", 0)
                coverage_totals["max_layer_reached"] += depth_cov.get("max_layer_reached", 0)
            
            if "node_coverage" in coverage:
                node_cov = coverage["node_coverage"]
                coverage_totals["node_coverage_ratio"] += node_cov.get("ratio", 0)
                coverage_totals["illuminated_count"] += node_cov.get("illuminated_count", 0)
                coverage_totals["total_log_nodes"] += node_cov.get("total_log_nodes", 0)
            
            if "premise_coverage" in coverage:
                premise_cov = coverage["premise_coverage"]
                coverage_totals["premise_coverage_ratio"] += premise_cov.get("ratio", 0)
            
            # 收集各层分布数据
            if "layer_distribution" in coverage:
                layer_distributions.append(coverage["layer_distribution"])
            
            # Precision指标
            if "error_rate" in precision:
                error_rate = precision["error_rate"]
                precision_totals["error_rate"] += error_rate.get("ratio", 0)
                precision_totals["provable_count"] += error_rate.get("provable_count", 0)
                precision_totals["total_count"] += error_rate.get("total_count", 0)
            
            if "strict_error_rate" in precision:
                strict_error = precision["strict_error_rate"]
                precision_totals["strict_error_rate"] += strict_error.get("ratio", 0)
            
            if "quality_distribution" in precision:
                quality = precision["quality_distribution"]
                precision_totals["perfect_count"] += quality.get("perfect", 0)
                precision_totals["partial_count"] += quality.get("partial", 0)
                precision_totals["invalid_count"] += quality.get("invalid", 0)
            
            # Summary指标
            summary_totals["total_statements"] += summary.get("total_statements", 0)
            summary_totals["premise_statements"] += summary.get("premise_statements", 0)
            summary_totals["derived_statements"] += summary.get("derived_statements", 0)
        
        # 计算平均值
        n = len(all_metrics)
        
        # 计算各层点亮比例的平均值
        average_layer_distribution = {}
        if layer_distributions:
            # 收集所有层的数据
            all_layers = set()
            for layer_dist in layer_distributions:
                all_layers.update(layer_dist.keys())
            
            # 为每一层计算平均比例
            for layer_key in all_layers:
                layer_totals = {"total": 0, "illuminated": 0, "ratio_sum": 0, "count": 0}
                
                for layer_dist in layer_distributions:
                    if layer_key in layer_dist:
                        layer_data = layer_dist[layer_key]
                        layer_totals["total"] += layer_data.get("total", 0)
                        layer_totals["illuminated"] += layer_data.get("illuminated", 0)
                        layer_totals["ratio_sum"] += layer_data.get("ratio", 0)
                        layer_totals["count"] += 1
                
                # 计算该层的平均比例
                average_ratio = layer_totals["ratio_sum"] / layer_totals["count"] if layer_totals["count"] > 0 else 0
                overall_ratio = layer_totals["illuminated"] / layer_totals["total"] if layer_totals["total"] > 0 else 0
                
                average_layer_distribution[layer_key] = {
                    "total_nodes": layer_totals["total"],
                    "total_illuminated": layer_totals["illuminated"],
                    "average_ratio": average_ratio,
                    "overall_ratio": overall_ratio,
                    "record_count": layer_totals["count"]
                }
        
        return {
            "record_count": n,
            "coverage": {
                "depth_coverage": {
                    "average_ratio": coverage_totals["depth_coverage_ratio"] / n,
                    "average_max_layer": coverage_totals["max_layer_reached"] / n
                },
                "node_coverage": {
                    "average_ratio": coverage_totals["node_coverage_ratio"] / n,
                    "total_illuminated": coverage_totals["illuminated_count"],
                    "total_log_nodes": coverage_totals["total_log_nodes"],
                    "overall_ratio": coverage_totals["illuminated_count"] / coverage_totals["total_log_nodes"] if coverage_totals["total_log_nodes"] > 0 else 0
                },
                "premise_coverage": {
                    "average_ratio": coverage_totals["premise_coverage_ratio"] / n
                },
                "layer_distribution": average_layer_distribution
            },
            "precision": {
                "error_rate": {
                    "average_ratio": precision_totals["error_rate"] / n,
                    "total_provable": precision_totals["provable_count"],
                    "total_derived": precision_totals["total_count"],
                    "overall_ratio": 1 - (precision_totals["provable_count"] / precision_totals["total_count"]) if precision_totals["total_count"] > 0 else 0
                },
                "strict_error_rate": {
                    "average_ratio": precision_totals["strict_error_rate"] / n
                },
                "quality_distribution": {
                    "total_perfect": precision_totals["perfect_count"],
                    "total_partial": precision_totals["partial_count"],
                    "total_invalid": precision_totals["invalid_count"],
                    "total_derived": precision_totals["perfect_count"] + precision_totals["partial_count"] + precision_totals["invalid_count"]
                }
            },
            "summary": {
                "total_statements": summary_totals["total_statements"],
                "total_premise_statements": summary_totals["premise_statements"],
                "total_derived_statements": summary_totals["derived_statements"],
                "average_statements_per_record": summary_totals["total_statements"] / n,
                "average_derived_per_record": summary_totals["derived_statements"] / n
            }
        }
    
    def print_average_metrics_summary(self, average_metrics: Dict[str, Any], record_count: int):
        """
        打印平均指标摘要
        
        Args:
            average_metrics: 平均指标数据
            record_count: 记录数量
        """
        print(f"\n{'='*80}")
        print(f"📊 平均指标摘要 (基于 {record_count} 条记录)")
        print(f"{'='*80}")
        
        coverage = average_metrics.get("coverage", {})
        precision = average_metrics.get("precision", {})
        summary = average_metrics.get("summary", {})
        
        # Coverage指标
        print(f"\n🎯 Coverage指标 (召回率):")
        if "depth_coverage" in coverage:
            depth = coverage["depth_coverage"]
            print(f"   深度Coverage: {depth.get('average_ratio', 0):.2%} (平均最深层级: {depth.get('average_max_layer', 0):.1f})")
        
        if "node_coverage" in coverage:
            node = coverage["node_coverage"]
            print(f"   节点Coverage: {node.get('average_ratio', 0):.2%} (总体: {node.get('overall_ratio', 0):.2%})")
            print(f"     - 总点亮节点: {node.get('total_illuminated', 0)}")
            print(f"     - 总LoG节点: {node.get('total_log_nodes', 0)}")
        
        if "premise_coverage" in coverage:
            premise = coverage["premise_coverage"]
            print(f"   前提Coverage: {premise.get('average_ratio', 0):.2%}")
        
        # 各层点亮比例平均值
        if "layer_distribution" in coverage:
            layer_dist = coverage["layer_distribution"]
            if layer_dist:
                print(f"   各推理层平均点亮比例:")
                # 按层级编号排序显示
                sorted_layers = sorted(layer_dist.items(), key=lambda x: int(x[0].split('_')[1]))
                for layer_key, layer_data in sorted_layers:
                    layer_num = layer_key.split('_')[1]
                    avg_ratio = layer_data.get('average_ratio', 0)
                    overall_ratio = layer_data.get('overall_ratio', 0)
                    total_nodes = layer_data.get('total_nodes', 0)
                    total_illuminated = layer_data.get('total_illuminated', 0)
                    record_count = layer_data.get('record_count', 0)
                    print(f"     第{layer_num}层: 平均{avg_ratio:.2%}, 总体{total_illuminated}/{total_nodes}={overall_ratio:.2%} ({record_count}条记录)")
        
        # Precision指标
        print(f"\n🎯 Precision指标 (精确率):")
        if "error_rate" in precision:
            error = precision["error_rate"]
            print(f"   Error Rate: {error.get('average_ratio', 0):.2%} (总体: {error.get('overall_ratio', 0):.2%})")
            print(f"     - 可推导节点: {error.get('total_provable', 0)}")
            print(f"     - 总推理节点: {error.get('total_derived', 0)}")
        
        if "strict_error_rate" in precision:
            strict = precision["strict_error_rate"]
            print(f"   Strict Error Rate: {strict.get('average_ratio', 0):.2%}")
        
        if "quality_distribution" in precision:
            quality = precision["quality_distribution"]
            total = quality.get("total_derived", 1)
            print(f"   推理质量分布:")
            print(f"     - 完美推理: {quality.get('total_perfect', 0)} ({quality.get('total_perfect', 0)/total:.2%})")
            print(f"     - 部分推理: {quality.get('total_partial', 0)} ({quality.get('total_partial', 0)/total:.2%})")
            print(f"     - 无效推理: {quality.get('total_invalid', 0)} ({quality.get('total_invalid', 0)/total:.2%})")
        
        # Summary指标
        print(f"\n📈 数据统计:")
        print(f"   总statements: {summary.get('total_statements', 0)} (平均: {summary.get('average_statements_per_record', 0):.1f}/记录)")
        print(f"   前提statements: {summary.get('total_premise_statements', 0)}")
        print(f"   推理statements: {summary.get('total_derived_statements', 0)} (平均: {summary.get('average_derived_per_record', 0):.1f}/记录)")
    
    def verify_reasoning_with_engine(self, initial_conditions: List[str], 
                                   normalized_nodes: List[Dict[str, Any]], 
                                   target_question: str = None, debug: bool = False) -> Dict[str, Any]:
        """
        使用推理引擎验证推理过程
        
        Args:
            initial_conditions: 初始条件列表
            normalized_nodes: 标准化的节点列表
            target_question: 目标问题（可选）
            debug: 是否开启调试模式
            
        Returns:
            验证结果
        """
        print(f"\n=== 开始推理引擎验证 ===")
        
        # 将初始条件转换为节点格式
        premise_nodes = []
        for condition in initial_conditions:
            if ' is ' in condition:
                parts = condition.split(' is ', 1)
                if len(parts) == 2:
                    input_part = parts[0].strip()
                    output_part = parts[1].strip()
                    output_parsed = self.reasoning_engine.parse_output_entities(output_part)
                    
                    premise_nodes.append({
                        "original": condition,
                        "input": input_part,
                        "output": output_part,
                        "output_parsed": output_parsed,
                        "type": "initial_condition"
                    })
        
        print(f"转换了 {len(premise_nodes)} 个初始条件为前提节点")
        
        # 验证每个推理步骤
        verification_results = []
        successful_verifications = 0
        
        for i, node in enumerate(normalized_nodes):
            if node.get("type") == "actual":  # 只验证actual类型的节点
                print(f"\n验证节点 {i+1}: {node['input']} is {node['output']}")
                
                # 构造目标节点
                target_node = {
                    "input": node["input"],
                    "output": node["output"],
                    "output_parsed": node["output_parsed"],
                    "original": node.get("original", f"{node['input']} is {node['output']}"),
                    "type": "target"
                }
                
                # 使用推理引擎验证
                start_time = time.time()
                is_provable = self.reasoning_engine.is_provable(
                    target_node, premise_nodes, debug=debug
                )
                verification_time = time.time() - start_time
                
                result = {
                    "node_index": i,
                    "node": node,
                    "target": target_node,
                    "is_provable": is_provable,
                    "verification_time": verification_time,
                    "status": "success" if is_provable else "failed"
                }
                
                verification_results.append(result)
                
                if is_provable:
                    successful_verifications += 1
                    print(f"  ✓ 验证成功 ({verification_time:.2f}s)")
                    
                    # 如果验证成功，将此节点添加到前提中，供后续验证使用
                    premise_nodes.append(target_node)
                else:
                    print(f"  ✗ 验证失败 ({verification_time:.2f}s)")
        
        # 如果有目标问题，也验证一下
        target_verification = None
        if target_question and ' is ' in target_question:
            print(f"\n验证最终目标: {target_question}")
            parts = target_question.split(' is ', 1)
            if len(parts) == 2:
                input_part = parts[0].strip()
                output_part = parts[1].strip()
                output_parsed = self.reasoning_engine.parse_output_entities(output_part)
                
                target_node = {
                    "input": input_part,
                    "output": output_part,
                    "output_parsed": output_parsed,
                    "original": target_question,
                    "type": "final_target"
                }
                
                start_time = time.time()
                is_provable = self.reasoning_engine.is_provable(
                    target_node, premise_nodes, debug=debug
                )
                verification_time = time.time() - start_time
                
                target_verification = {
                    "target_question": target_question,
                    "target": target_node,
                    "is_provable": is_provable,
                    "verification_time": verification_time,
                    "status": "success" if is_provable else "failed"
                }
                
                if is_provable:
                    print(f"  ✓ 最终目标验证成功 ({verification_time:.2f}s)")
                else:
                    print(f"  ✗ 最终目标验证失败 ({verification_time:.2f}s)")
        
        # 统计结果
        total_nodes = len([n for n in normalized_nodes if n.get("type") == "actual"])
        success_rate = successful_verifications / total_nodes * 100 if total_nodes > 0 else 0
        
        print(f"\n=== 验证结果统计 ===")
        print(f"总节点数: {total_nodes}")
        print(f"验证成功: {successful_verifications}")
        print(f"验证失败: {total_nodes - successful_verifications}")
        print(f"成功率: {success_rate:.1f}%")
        
        return {
            "premise_nodes_count": len(premise_nodes),
            "total_nodes": total_nodes,
            "successful_verifications": successful_verifications,
            "failed_verifications": total_nodes - successful_verifications,
            "success_rate": success_rate,
            "verification_results": verification_results,
            "target_verification": target_verification,
            "reasoning_engine_config": {
                "max_depth": self.reasoning_engine.max_depth,
                "timeout": self.reasoning_engine.timeout
            }
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="逐步评估推理过程 v2.0")
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
    parser.add_argument("--llm_debug_mode", action="store_true",
                       help="LLM调试模式，只做提取和记录，跳过后处理逻辑")
    parser.add_argument("--api_mode", type=str, default="commercial",
                       choices=["commercial", "vllm"],
                       help="API模式：commercial（商业API）或vllm（VLLM API）")
    parser.add_argument("--verbose_premise", action="store_true",
                       help="详细输出模式，显示每个节点的前提信息和推理轨迹")
    parser.add_argument("--single_record_debug", action="store_true",
                       help="单条记录调试模式，只处理第一条数据（用于调试）")
    parser.add_argument("--max_records", type=int, default=None,
                       help="遍历模式下最大处理记录数，默认处理所有记录")
    parser.add_argument("--record_index", type=int, default=0,
                       help="单条记录调试模式下指定处理哪一条记录（从0开始），默认第一条")
    
    args = parser.parse_args()
    
    try:
        # 创建评估器
        evaluator = StepByStepEvaluator2(
            api_key=args.api_key,
            model_name=args.model_name,
            api_base=args.api_base,
            debug_mode=args.debug_mode,
            llm_debug_mode=args.llm_debug_mode,
            api_mode=args.api_mode,
            verbose_premise=args.verbose_premise,
            single_record_debug_mode=args.single_record_debug,
            max_records=args.max_records,
            record_index=args.record_index
        )
        
        # 执行评估
        results = evaluator.evaluate_reasoning_process(
            log_path=args.log_path,
            output_path=args.output_path
        )
        
        print("\n评估成功完成！")
        
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
