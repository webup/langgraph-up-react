#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG系统性能基准测试脚本
用于测试优化后的RAG检索性能
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
from rag import KB_Retrieval
from llm_server import LLM


class RAGBenchmark:
    """RAG性能基准测试类"""
    
    def __init__(self):
        self.rag_with_cache = KB_Retrieval(enable_cache=True)
        self.rag_without_cache = KB_Retrieval(enable_cache=False)
        self.llm = LLM(enable_cache=True)
        
        # 测试查询集
        self.test_queries = [
            ["学费缴纳", "如何缴费", "缴费政策"],
            ["图书馆开放时间", "借书规定"],
            ["选课系统", "课程安排", "学分要求"],
            ["宿舍管理", "住宿申请", "宿舍规定"],
            ["校园网", "网络连接", "VPN使用"],
            ["学生证办理", "证件申请"],
            ["食堂就餐", "餐厅位置", "用餐时间"],
            ["考试安排", "成绩查询", "补考政策"],
            ["奖学金申请", "助学金", "勤工助学"],
            ["毕业要求", "学位申请", "毕业流程"]
        ]
    
    def benchmark_retrieval_speed(self, use_cache: bool = True, runs: int = 3) -> Dict[str, Any]:
        """测试检索速度"""
        rag_client = self.rag_with_cache if use_cache else self.rag_without_cache
        
        times = []
        cache_hits = 0
        
        print(f"\n{'='*50}")
        print(f"测试检索性能 ({'启用缓存' if use_cache else '禁用缓存'})")
        print(f"{'='*50}")
        
        for run in range(runs):
            print(f"\n运行 {run + 1}/{runs}")
            run_times = []
            
            for i, query in enumerate(self.test_queries):
                start_time = time.time()
                
                try:
                    result = rag_client.retrieve(query)
                    end_time = time.time()
                    
                    duration = end_time - start_time
                    run_times.append(duration)
                    
                    # 检测是否命中缓存（第二次查询相同问题）
                    if run > 0 and use_cache:
                        cache_start = time.time()
                        rag_client.retrieve(query)
                        cache_duration = time.time() - cache_start
                        if cache_duration < duration * 0.1:  # 如果时间减少90%以上，认为命中缓存
                            cache_hits += 1
                    
                    print(f"  查询 {i+1}: {duration:.3f}s - {len(result)} chars")
                    
                except Exception as e:
                    print(f"  查询 {i+1} 失败: {e}")
                    run_times.append(float('inf'))
            
            times.extend(run_times)
            print(f"  本轮平均时间: {statistics.mean(run_times):.3f}s")
        
        # 过滤无效结果
        valid_times = [t for t in times if t != float('inf')]
        
        if not valid_times:
            return {"error": "所有查询都失败了"}
        
        return {
            "cache_enabled": use_cache,
            "total_queries": len(valid_times),
            "cache_hits": cache_hits,
            "avg_time": statistics.mean(valid_times),
            "median_time": statistics.median(valid_times),
            "min_time": min(valid_times),
            "max_time": max(valid_times),
            "std_dev": statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
            "success_rate": len(valid_times) / len(times) * 100
        }
    
    async def benchmark_async_retrieval(self, runs: int = 2) -> Dict[str, Any]:
        """测试异步检索性能"""
        print(f"\n{'='*50}")
        print(f"测试异步检索性能")
        print(f"{'='*50}")
        
        times = []
        
        for run in range(runs):
            print(f"\n异步运行 {run + 1}/{runs}")
            start_time = time.time()
            
            # 并发执行所有查询
            tasks = [
                self.rag_with_cache.async_retrieve(query) 
                for query in self.test_queries
            ]
            
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                end_time = time.time()
                
                duration = end_time - start_time
                times.append(duration)
                
                success_count = sum(1 for r in results if not isinstance(r, Exception))
                print(f"  并发执行 {len(tasks)} 个查询: {duration:.3f}s")
                print(f"  成功: {success_count}/{len(tasks)}")
                
            except Exception as e:
                print(f"  异步执行失败: {e}")
        
        if not times:
            return {"error": "异步测试失败"}
        
        return {
            "concurrent_queries": len(self.test_queries),
            "avg_total_time": statistics.mean(times),
            "avg_time_per_query": statistics.mean(times) / len(self.test_queries),
            "runs": runs
        }
    
    def benchmark_memory_usage(self) -> Dict[str, Any]:
        """测试内存使用情况"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # 记录初始内存
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行大量查询
        print(f"\n{'='*50}")
        print(f"测试内存使用情况")
        print(f"{'='*50}")
        
        print(f"初始内存: {initial_memory:.2f} MB")
        
        # 执行查询
        for i in range(5):
            for query in self.test_queries:
                self.rag_with_cache.retrieve(query)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"峰值内存: {peak_memory:.2f} MB")
        
        # 清空缓存
        self.rag_with_cache.clear_cache()
        
        after_clear_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"清除缓存后: {after_clear_memory:.2f} MB")
        
        return {
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "after_clear_memory_mb": after_clear_memory,
            "memory_increase_mb": peak_memory - initial_memory,
            "memory_freed_mb": peak_memory - after_clear_memory
        }
    
    def run_full_benchmark(self):
        """运行完整的基准测试"""
        print("开始RAG系统性能基准测试...")
        
        results = {}
        
        # 测试有缓存的性能
        results["with_cache"] = self.benchmark_retrieval_speed(use_cache=True, runs=3)
        
        # 测试无缓存的性能
        results["without_cache"] = self.benchmark_retrieval_speed(use_cache=False, runs=2)
        
        # 测试异步性能
        results["async_performance"] = asyncio.run(self.benchmark_async_retrieval(runs=2))
        
        # 测试内存使用
        results["memory_usage"] = self.benchmark_memory_usage()
        
        # 打印汇总报告
        self.print_summary_report(results)
        
        return results
    
    def print_summary_report(self, results: Dict[str, Any]):
        """打印汇总报告"""
        print(f"\n{'='*60}")
        print(f"RAG性能测试汇总报告")
        print(f"{'='*60}")
        
        if "with_cache" in results and "without_cache" in results:
            with_cache = results["with_cache"]
            without_cache = results["without_cache"]
            
            if "avg_time" in with_cache and "avg_time" in without_cache:
                speedup = without_cache["avg_time"] / with_cache["avg_time"]
                print(f"\n📊 检索性能对比:")
                print(f"  缓存启用时平均响应时间: {with_cache['avg_time']:.3f}s")
                print(f"  缓存禁用时平均响应时间: {without_cache['avg_time']:.3f}s")
                print(f"  性能提升倍数: {speedup:.2f}x")
                print(f"  缓存命中次数: {with_cache.get('cache_hits', 0)}")
        
        if "async_performance" in results:
            async_perf = results["async_performance"]
            if "avg_time_per_query" in async_perf:
                print(f"\n🚀 异步性能:")
                print(f"  并发查询平均单次时间: {async_perf['avg_time_per_query']:.3f}s")
                print(f"  并发执行 {async_perf['concurrent_queries']} 个查询总时间: {async_perf['avg_total_time']:.3f}s")
        
        if "memory_usage" in results:
            memory = results["memory_usage"]
            print(f"\n💾 内存使用:")
            print(f"  内存增长: {memory['memory_increase_mb']:.2f} MB")
            print(f"  缓存释放: {memory['memory_freed_mb']:.2f} MB")
            print(f"  峰值内存: {memory['peak_memory_mb']:.2f} MB")
        
        print(f"\n✅ 优化建议:")
        
        if "with_cache" in results and results["with_cache"].get("cache_hits", 0) > 0:
            print("  ✓ 缓存机制工作正常，显著提升了重复查询的性能")
        
        if "async_performance" in results:
            print("  ✓ 异步查询支持并发处理，适合批量查询场景")
        
        print("  ✓ 建议在生产环境中启用缓存以获得最佳性能")
        print("  ✓ 对于高并发场景，使用异步接口可以提升吞吐量")


def main():
    """主函数"""
    benchmark = RAGBenchmark()
    results = benchmark.run_full_benchmark()
    
    # 可选：将结果保存到文件
    import json
    with open("rag_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n测试结果已保存到 rag_benchmark_results.json")


if __name__ == "__main__":
    main()