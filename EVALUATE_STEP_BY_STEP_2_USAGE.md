# evaluate_step_by_step_2.py 使用说明

## 概述

`evaluate_step_by_step_2.py` 是一个用于逐步评估推理过程的工具，支持处理单条记录或所有记录，并计算平均指标。

## 主要功能

1. **处理所有记录**：默认模式，处理日志文件中的所有记录并计算平均指标
2. **单条记录调试模式**：只处理第一条记录，用于调试和快速测试
3. **指标计算**：计算Coverage和Precision指标，并提供平均值统计
4. **缓存机制**：支持LLM提取结果缓存，避免重复API调用

## 使用方法

### 基本用法

```bash
# 处理所有记录（默认模式）
python evaluate_step_by_step_2.py --log_path evaluation_log/evaluation_results.LoG_5.deepseek-reasoner.json

# 单条记录调试模式
python evaluate_step_by_step_2.py --log_path evaluation_log/evaluation_results.LoG_5.deepseek-reasoner.json --single_record_debug

# 跳过API调用的调试模式
python evaluate_step_by_step_2.py --log_path evaluation_log/evaluation_results.LoG_5.deepseek-reasoner.json --debug_mode
```

### 参数说明

#### 必需参数
- `--log_path`: 评估日志文件路径

#### 可选参数
- `--output_path`: 输出文件路径（默认自动生成）
- `--api_key`: API密钥（默认使用预设值）
- `--model_name`: 模型名称（默认: deepseek-reasoner）
- `--api_base`: API基础URL（默认: https://api.deepseek.com/beta）
- `--debug_mode`: 调试模式，跳过所有API调用
- `--llm_debug_mode`: LLM调试模式，只做提取和记录
- `--api_mode`: API模式，可选 commercial 或 vllm
- `--verbose_premise`: 详细输出模式，显示每个节点的前提信息和推理轨迹
- `--single_record_debug`: **新增** 单条记录调试模式，只处理第一条数据
- `--max_records`: **新增** 遍历模式下最大处理记录数，默认处理所有记录
- `--record_index`: **新增** 单条记录调试模式下指定处理哪一条记录（从0开始），默认第一条

### 使用场景

#### 1. 开发调试
```bash
# 快速测试单条记录，跳过API调用
python evaluate_step_by_step_2.py \
  --log_path evaluation_log/evaluation_results.LoG_5.deepseek-reasoner.json \
  --debug_mode \
  --single_record_debug

# 测试指定的某条记录（例如第5条，索引为4）
python evaluate_step_by_step_2.py \
  --log_path evaluation_log/evaluation_results.LoG_5.deepseek-reasoner.json \
  --debug_mode \
  --single_record_debug \
  --record_index 4
```

#### 2. 部分评估
```bash
# 只处理前10条记录
python evaluate_step_by_step_2.py \
  --log_path evaluation_log/evaluation_results.LoG_5.deepseek-reasoner.json \
  --max_records 10

# 处理前50条记录并显示详细信息
python evaluate_step_by_step_2.py \
  --log_path evaluation_log/evaluation_results.LoG_5.deepseek-reasoner.json \
  --max_records 50 \
  --verbose_premise
```

#### 3. 完整评估
```bash
# 处理所有记录并计算平均指标
python evaluate_step_by_step_2.py \
  --log_path evaluation_log/evaluation_results.LoG_5.deepseek-reasoner.json \
  --verbose_premise
```

#### 4. 使用缓存加速
```bash
# 如果已有缓存，会自动使用缓存结果
python evaluate_step_by_step_2.py \
  --log_path evaluation_log/evaluation_results.LoG_5.deepseek-reasoner.json
```

## 输出结果

### 单条记录模式
- 处理第一条记录的详细信息
- 该记录的完整指标

### 所有记录模式
- 每条记录的处理结果
- **新增** 平均指标统计：
  - Coverage指标平均值
  - Precision指标平均值
  - 数据统计汇总

### 输出文件结构
```json
{
  "log_path": "...",
  "total_records": 100,
  "model_name": "deepseek-reasoner",
  "processed_records": [...],
  "average_metrics": {  // 新增：平均指标
    "record_count": 100,
    "coverage": {
      "depth_coverage": {
        "average_ratio": 0.75,
        "average_max_layer": 3.2
      },
      "node_coverage": {
        "average_ratio": 0.68,
        "overall_ratio": 0.70
      },
      "premise_coverage": {
        "average_ratio": 0.85
      }
    },
    "precision": {
      "error_rate": {
        "average_ratio": 0.15,
        "overall_ratio": 0.12
      },
      "strict_error_rate": {
        "average_ratio": 0.25
      },
      "quality_distribution": {
        "total_perfect": 150,
        "total_partial": 80,
        "total_invalid": 20
      }
    },
    "summary": {
      "total_statements": 2500,
      "average_statements_per_record": 25.0,
      "average_derived_per_record": 15.0
    }
  }
}
```

## 性能优化

1. **缓存机制**：LLM提取结果会缓存在 `./LLM_extract_node/` 目录
2. **并行处理**：每条记录使用独立的PostProcessor实例
3. **内存管理**：避免在处理多条记录时的状态混乱

## 注意事项

1. **文件依赖**：确保存在 `extract_prompt_2.txt` 文件
2. **LoG数据**：程序会自动从日志文件推断对应的LoG数据文件路径
3. **API限制**：如果使用真实API，注意请求频率限制
4. **内存使用**：处理大量记录时可能占用较多内存

## 错误处理

- 如果某条记录处理失败，会跳过该记录并继续处理其他记录
- 最终的平均指标只基于成功处理的记录计算
- 详细的错误信息会输出到控制台

## 版本变更

### v2.0 新功能
- ✅ 支持处理所有记录（原来只处理第一条）
- ✅ 添加平均指标计算
- ✅ 新增单条记录调试模式
- ✅ 改进错误处理和状态管理
- ✅ 优化内存使用和性能
