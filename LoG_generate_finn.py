import argparse
from typing import List, Dict
import random
import string

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



# DI（Disjunction Introduction）推理生成：x is A or B 推出 x is A
def DI_generate(tmp_output: str) -> List[str]:
    try:
        if " is " not in tmp_output:
            raise ValueError("Invalid DI input format.")
        subject, A = tmp_output.split(" is ", 1)
        A = A.strip()
        B = random.choice([e for e in element_ls if e != A])
        return [f"{subject.strip()} is {A} or {B}"]
    except Exception as e:
        print(f"DI_generate error: {e} | input: {tmp_output}")
        return [f"{subject if 'subject' in locals() else 'x'} is Something or SomethingElse"]



# CI（Conjunction Introduction）推理生成：x is A and B 推出 x is A, x is B
def CI_generate(tmp_output: str) -> List[str]:
    try:
        if " is " not in tmp_output or " and " not in tmp_output:
            raise ValueError("Invalid CI input format.")
        subject, predicate = tmp_output.split(" is ", 1)
        parts = predicate.strip().split(" and ")
        if len(parts) < 2:
            raise ValueError("CI requires at least two conjuncts")
        return [f"{subject.strip()} is {part.strip()}" for part in parts]
    except Exception as e:
        print(f"CI_generate error: {e} | input: {tmp_output}")
        return [f"{subject if 'subject' in locals() else 'x'} is Something",
                f"{subject if 'subject' in locals() else 'x'} is SomethingElse"]


# 根据当前输出与连接词，选择适当推理规则生成输入
def input_generate(pre_conjunction: str, len_output: int, tmp_output: str) -> List[str]:
    tmp_deduction_rule = []

    if tmp_output.startswith("x is "):
        tmp_deduction_rule.append('MP')

    if pre_conjunction == "or":
        tmp_deduction_rule.append('DI')
    if pre_conjunction == 'and':
        tmp_deduction_rule.append('CI')
    if (pre_conjunction == 'and' and len_output < 4) or len_output == 1:
        tmp_deduction_rule.append('CE')

    if not tmp_deduction_rule:
        tmp_deduction_rule.append('CE')  # fallback

    chosen_deduction_rule = random.choice(tmp_deduction_rule)

    if chosen_deduction_rule == 'MP':
        tmp_res = MP_generate(tmp_output)
    elif chosen_deduction_rule == 'CE':
        tmp_res = CE_generate(tmp_output)
    elif chosen_deduction_rule == 'DI':
        tmp_res = DI_generate(tmp_output)
    elif chosen_deduction_rule == 'CI':
        tmp_res = CI_generate(tmp_output)
    else:
        tmp_res = ["x is something"]

    return [chosen_deduction_rule] + tmp_res



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hop', type=int, default=2)
    args = parser.parse_args()

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

            if pre_dict['depth'] > args.num_hop:
                continue  # 达到深度，不再扩展

            tmp_output = pre_dict['output']
            pre_conjunction = pre_dict['conjunction']
            pre_depth = pre_dict['depth']

            len_output = len(tmp_output.split(pre_conjunction)) if pre_conjunction else 1
            tmp_res = input_generate(pre_conjunction, len_output, tmp_output)

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
                    'input': None,
                    'conjunction': 'and' if 'and' in i else 'or' if 'or' in i else ''
                }
                ls_hop_dealing.append(tmp_todo_dict)

    # 打印最终生成的推理图
    print(ls_hop_res)
