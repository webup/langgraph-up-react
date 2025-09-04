# RAG 模块性能优化

本目录包含了经过性能优化的RAG（检索增强生成）系统。

## 主要优化项

### 1. 缓存机制
- **查询缓存**: 对相同查询结果进行缓存，避免重复计算
- **Rerank缓存**: 缓存相似度计算结果，提升重排序速度
- **LRU缓存**: 使用LRU策略管理内存使用
- **TTL过期**: 5-10分钟自动过期，确保数据新鲜度

### 2. 检索优化
- **去重优化**: 使用集合(Set)进行O(1)去重，替代列表的O(n)查找
- **批量处理**: 支持多线程并发检索多个查询
- **智能跳过**: 当候选项少于top_k时跳过rerank，直接返回
- **错误处理**: 完善的异常处理和降级策略

### 3. 网络优化
- **连接池**: 复用HTTP连接，减少建立连接的开销
- **超时设置**: 合理的超时时间防止长时间等待
- **重试机制**: 自动重试失败的请求
- **流式处理**: 禁用不必要的流式传输

### 4. 异步支持
- **异步检索**: 提供async_retrieve方法支持异步调用
- **并发处理**: 支持多个查询并发执行
- **线程池**: 使用线程池执行CPU密集型任务

## 性能提升预期

根据优化内容，预期性能提升如下：

1. **缓存命中时**: 90%+ 的响应时间减少
2. **去重优化**: 50%+ 的去重操作时间减少
3. **批量查询**: 60%+ 的总处理时间减少（多查询场景）
4. **网络优化**: 20-30% 的网络请求时间减少

## 使用方法

### 基本使用
```python
from src.rag.rag import KB_Retrieval

# 启用缓存（推荐）
rag = KB_Retrieval(enable_cache=True)
context = rag.retrieve(["查询问题1", "查询问题2"])
```

### 异步使用
```python
import asyncio

async def async_example():
    rag = KB_Retrieval(enable_cache=True)
    context = await rag.async_retrieve(["异步查询"])
    return context

result = asyncio.run(async_example())
```

### 性能测试
```bash
cd src/rag
python benchmark.py
```

## 配置参数

- `similarity_score`: 相似度阈值 (默认: 0.5)
- `top_k`: 返回的chunk数量 (默认: 5)
- `max_workers`: 最大线程数 (默认: 4)
- `enable_cache`: 是否启用缓存 (默认: True)

## 缓存管理

```python
# 清空缓存
rag.clear_cache()

# 禁用缓存
rag = KB_Retrieval(enable_cache=False)
```

## 注意事项

1. **内存使用**: 缓存会占用一定内存，建议监控内存使用情况
2. **缓存一致性**: 如果底层数据发生变化，需要清空缓存
3. **并发限制**: 建议max_workers不超过8，避免过多并发连接
4. **网络依赖**: 优化主要针对网络和计算密集型操作

## 故障排除

### 常见问题
1. **缓存过期**: 缓存会自动过期，无需担心数据过时
2. **内存增长**: 可调用clear_cache()释放内存
3. **网络超时**: 已设置合理超时时间和重试机制
4. **并发错误**: 使用线程锁保证线程安全

### 性能监控
```python
# 使用benchmark.py进行性能监控
python benchmark.py

# 查看缓存统计
print(f"缓存大小: {len(rag._cache)}")
```