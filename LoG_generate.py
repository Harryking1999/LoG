import argparse
from typing import List, Dict
import random
import string
import json

# 工具函数：复制列表，保持原始 element_ls 不变
def copy_ls(ls_orig: List[str]) -> List[str]:
    return [i for i in ls_orig]

# 元素集合（主语列表）
def generate_element_ls(n=100, seed=42):
    random.seed(seed)
    vowels = 'aeiou'
    consonants = ''.join(set(string.ascii_lowercase) - set(vowels))
    elements = set()

    while len(elements) < n:
        prefix = ''.join([
            random.choice(consonants),
            random.choice(vowels),
            random.choice(consonants)
        ])
        elements.add(prefix + 'pus')

    return sorted(list(elements))

element_ls = generate_element_ls(100)

# element_ls = [
#     'brimpus', 'gorpus', 'shumpus', 'vumpus', 'numpus', 'dumpus',
#     'sterpus', 'jorpus', 'yumpus', 'impus', 'rompus', 'wumpus',  # ' wumpus' 去除首空格
#     'jompus', 'grimpus', 'frompus', 'himpus', 'anpus', 'korpus',
#     'lorpus', 'mumpus', 'orpus', 'porpus', 'quampus', 'sampus',
#     'tinpus', 'unpus', 'zinpus'
# ]

# 演绎规则列表
deduction_rule_ls = ['CE', 'DI', 'CI', 'MP']

# 连接词列表
conjunction = ['and', 'or']


# MP（Modus Ponens）推理生成：从 x is A 推出 x is B, B is A
def MP_generate(tmp_output: str) -> List[str]:
    try:
        if " is " not in tmp_output:
            raise ValueError("Invalid MP input format.")
        subject, A = tmp_output.split(" is ", 1)
        A = A.strip()
        B = random.choice([e for e in element_ls if e != A])
        return [f"{subject.strip()} is {B}", f"{B} is {A}"]
    except Exception as e:
        print(f"MP_generate error: {e} | input: {tmp_output}")
        return [f"{subject if 'subject' in locals() else 'x'} is Something",
                f"Something is SomethingElse"]


# CE（Conjunction Elimination）推理生成：x is A 推出 x is A and B 
def CE_generate(tmp_output: str) -> List[str]:
    try:
        if " is " not in tmp_output:
            raise ValueError("Invalid CE input format.")
        subject, A = tmp_output.split(" is ", 1)
        A = A.strip()
        B = random.choice([e for e in element_ls if e != A])
        return [f"{subject.strip()} is {A} and {B}"]
    except Exception as e:
        print(f"CE_generate error: {e} | input: {tmp_output}")
        return [f"{subject if 'subject' in locals() else 'x'} is Something and SomethingElse"]



# DI（Disjunction Introduction）推理生成：x is A 推出 x is A or B
# def DI_generate(tmp_output: str) -> List[str]:
#     try:
#         if " is " not in tmp_output:
#             raise ValueError("Invalid DI input format.")
#         subject, A = tmp_output.split(" is ", 1)
#         A = A.strip()
#         B = random.choice([e for e in element_ls if e != A])
#         return [f"{subject.strip()} is {A} or {B}"]
#     except Exception as e:
#         print(f"DI_generate error: {e} | input: {tmp_output}")
#         return [f"{subject if 'subject' in locals() else 'x'} is Something or SomethingElse"]

def DI_generate(tmp_output: str) -> List[str]:
    try:
        # 1. 格式验证
        if " is " not in tmp_output:
            raise ValueError("Invalid DI input format.")
        
        # 2. 解析 subject 和选项部分
        subject, options_part = tmp_output.split(" is ", 1)
        options = [opt.strip() for opt in options_part.split(" or ")]
        
        # 3. 如果只有一个选项，无法减少，返回原样
        if len(options) <= 1:
            return [tmp_output]
        
        # 4. 随机选择要保留的选项数量（1到n-1个）
        original_count = len(options)
        target_count = random.randint(1, original_count - 1)
        
        # 5. 随机选择对应数量的选项
        selected_options = random.sample(options, target_count)
        
        # 6. 生成输出字符串
        if target_count == 1:
            result = f"{subject.strip()} is {selected_options[0]}"
        else:
            result = f"{subject.strip()} is {' or '.join(selected_options)}"
        
        return [result]
        
    except Exception as e:
        print(f"DI_generate error: {e} | input: {tmp_output}")
        return ["x is Something"]



# CI（Conjunction Introduction）推理生成：x is A and B 推出 x is A, x is B
# def CI_generate(tmp_output: str) -> List[str]:
#     try:
#         if " is " not in tmp_output or " and " not in tmp_output:
#             raise ValueError("Invalid CI input format.")
#         subject, predicate = tmp_output.split(" is ", 1)
#         parts = predicate.strip().split(" and ")
#         if len(parts) < 2:
#             raise ValueError("CI requires at least two conjuncts")
#         return [f"{subject.strip()} is {part.strip()}" for part in parts]
#     except Exception as e:
#         print(f"CI_generate error: {e} | input: {tmp_output}")
#         return [f"{subject if 'subject' in locals() else 'x'} is Something",
#                 f"{subject if 'subject' in locals() else 'x'} is SomethingElse"]
def CI_generate(tmp_output: str) -> List[str]:
    try:
        if " is " not in tmp_output or " and " not in tmp_output:
            raise ValueError("Invalid CI input format.")
        
        subject, predicate = tmp_output.split(" is ", 1)
        parts = [part.strip() for part in predicate.strip().split(" and ")]
        
        if len(parts) < 2:
            raise ValueError("CI requires at least two conjuncts")
        
        # 如果只有2个部分，只能完全分解
        if len(parts) == 2:
            return [f"{subject.strip()} is {part}" for part in parts]
        
        # 生成所有可能的分解方式
        all_partitions = []
        
        # 1. 完全分解：每个都是单独的
        complete_split = [f"{subject.strip()} is {part}" for part in parts]
        all_partitions.append(complete_split)
        
        # 2. 部分分解：尝试不同的分组方式
        # 例如：A and B and C → [A and B, C] 或 [A, B and C]
        for i in range(1, len(parts)):
            left_group = " and ".join(parts[:i])
            right_group = " and ".join(parts[i:])
            partial_split = [f"{subject.strip()} is {left_group}", 
                           f"{subject.strip()} is {right_group}"]
            all_partitions.append(partial_split)
        
        # 随机选择一种分解方式
        return random.choice(all_partitions)
        
    except Exception as e:
        print(f"CI_generate error: {e} | input: {tmp_output}")
        return ["x is Something", "x is SomethingElse"]


# 根据当前输出与连接词，选择适当推理规则生成输入
def input_generate(pre_conjunction: str, len_output: int, tmp_output: str, pre_deduction_rule: str) -> List[str]:
    tmp_deduction_rule = []

    # if tmp_output.startswith("x is "):
    #     tmp_deduction_rule.append('MP')
    tmp_deduction_rule.append('MP')

    if pre_conjunction == "or":
        tmp_deduction_rule.append('DI')
    if pre_conjunction == 'and':
        tmp_deduction_rule.append('CI')
    if ((pre_conjunction == 'and' and len_output < 4) or len_output == 1) and pre_deduction_rule != 'CE':
        tmp_deduction_rule.append('CE')

    # if not tmp_deduction_rule:
    #     tmp_deduction_rule.append('CE')  # fallback
    # print("pre_deduction_rule:", pre_deduction_rule)
    # print("tmp_deduction_rule", tmp_deduction_rule)

    chosen_deduction_rule = random.choice(tmp_deduction_rule)

    if chosen_deduction_rule == 'MP':
        tmp_res = MP_generate(tmp_output)
    elif chosen_deduction_rule == 'CE':
        tmp_res = CE_generate(tmp_output)
    elif chosen_deduction_rule == 'DI':
        tmp_res = DI_generate(tmp_output)
        # print("DI:", tmp_output)
        # print("DI: ", tmp_res)
    elif chosen_deduction_rule == 'CI':
        tmp_res = CI_generate(tmp_output)
    else:
        tmp_res = ["x is something"]

    return [chosen_deduction_rule] + tmp_res



def generate_single_reasoning_graph(num_hop):
    """生成单个推理图"""
    import time
    # 每次生成新图时都设置新的随机种子
    time.sleep(0.001)
    random.seed(int(time.time() * 1000000) % 2147483647)
    
    ls_hop_dealing = []  # 队列，存储待扩展的叶子结点
    ls_hop_res = []      # 存储所有生成的中间推理步骤
    flag_start = True    # 标记是否为初始结论生成阶段
    element_dynamic = copy_ls(element_ls)

    while True:
        if flag_start:
            # 生成初始结论（树的根节点）
            tmp_dict = {}
            tmp_element_cnt = random.randint(1, 3)
            tmp_conjunction = random.choice(conjunction) if tmp_element_cnt > 1 else ''
            tmp_dict['input'] = None
            tmp_dict['deduction_rule'] = None
            tmp_dict['conjunction'] = tmp_conjunction
            tmp_dict['pre_deduction_rule'] = None
            tmp_dict['depth'] = 1
            tmp_output = 'x is ' + element_dynamic[0]
            for i in range(1, tmp_element_cnt):
                tmp_output += f' {tmp_conjunction} {element_dynamic[i]}'
            tmp_dict['output'] = tmp_output
            flag_start = False
            ls_hop_dealing.append(tmp_dict)
        else:
            if not ls_hop_dealing:
                break  # 推理图生成完毕
            pre_dict = ls_hop_dealing.pop(0)

            if pre_dict['depth'] > num_hop:
                continue  # 达到深度，不再扩展

            tmp_output = pre_dict['output']
            pre_conjunction = pre_dict['conjunction']
            pre_depth = pre_dict['depth']
            pre_deduction_rule = pre_dict['pre_deduction_rule']

            len_output = len(tmp_output.split(pre_conjunction)) if pre_conjunction else 1
            # print(pre_dict)
            tmp_res = input_generate(pre_conjunction, len_output, tmp_output, pre_deduction_rule)

            tmp_dict = {
                'output': tmp_output,
                'conjunction': pre_conjunction,
                'deduction_rule': tmp_res[0],
                'depth': pre_depth,
                'input': tmp_res[1:]
            }

            ls_hop_res.append(tmp_dict)

            # 将输入语句作为下一层待扩展节点加入队列
            for i in tmp_dict['input']:
                tmp_todo_dict = {
                    'output': i,
                    'depth': pre_depth + 1,
                    'deduction_rule': None,
                    'pre_deduction_rule': tmp_res[0],
                    'input': None,
                    'conjunction': 'and' if 'and' in i else 'or' if 'or' in i else ''
                }
                ls_hop_dealing.append(tmp_todo_dict)

    return ls_hop_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hop', type=int, default=2)
    parser.add_argument('--num_graphs', type=int, default=1, help='Number of reasoning graphs to generate')
    args = parser.parse_args()

    f = open(f"./generated_data/LoG_{args.num_hop}.jsonl", "w+")
    
    # 生成指定数量的推理图
    all_graphs = []
    for i in range(args.num_graphs):
        graph = generate_single_reasoning_graph(args.num_hop)
        all_graphs.append(graph)
        tmp_information = ""
        ls_information = []
        tmp_question = "Is it true or false or unkown:"
        for i in graph:
            if(i['depth'] == args.num_hop):
                for j in i['input']:
                    ls_information.append(j)
            if(i['depth'] == 1):
                tmp_question += " " + i['output'] + "?"
            # print(i)
        random.shuffle(ls_information)
        for i in ls_information:
            tmp_information += i + ". " 
        tmp_str = f"Please answer the question based on the given information:\n**Given Information**: {tmp_information}\n**Question**: {tmp_question}\nPlease show your reasoning process and put your final answer in \boxed{{}}."
        tmp_json = {'question': tmp_str, 'answer': 'True'}
        f.write(json.dumps(tmp_json, ensure_ascii=False) + "\n")
        if args.num_graphs == 1:
            # 如果只生成一个图，直接打印
            print(graph)
            for i in graph:
                if(i['depth'] == args.num_hop):
                    for j in i['input']:
                        ls_information.append(j)
                if(i['depth'] == 1):
                    tmp_question += " " + i['output'] + "?"
                print(i)
            # random.shuffle(ls_information)
            # for i in ls_information:
            #     tmp_information += i + ". " 
            # tmp_str = f"Please answer the question based on the given information:\n **Given Infomation**:{tmp_information}\n\n**Question**:{tmp_question}"
            # tmp_json = {'question': tmp_str, 'answer': 'true'}
            # f.write(json.dumps(tmp_json, ensure_ascii=False) + "\n")
            # print(tmp_str)
        # else:
        #     # 如果生成多个图，打印带编号的结果
        #     print(f"Graph {i+1}:")
        #     print(graph)
        #     print()

        # f.write(str(graph) + "\n\n")
    f.close()