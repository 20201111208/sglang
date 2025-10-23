import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_traces(trace_file):
    """加载trace数据"""
    traces = []
    with open(trace_file, 'r') as f:
        for line in f:
            traces.append(json.loads(line))
    # 按到达时间排序
    traces.sort(key=lambda x: x['arrival_ts'])
    return traces

def temporal_prefix_overlap(traces, window_ms):
    """计算时间窗口内的前缀重叠率"""
    window_sec = window_ms / 1000.0
    overlaps = []
    
    for i, req in enumerate(traces):
        # 找到时间窗口内的其他请求
        window_reqs = [
            r for r in traces[i+1:] 
            if r['arrival_ts'] - req['arrival_ts'] < window_sec
        ]
        
        if not window_reqs:
            continue
        
        # 计算前缀重叠
        overlap = sum(
            1 for r in window_reqs 
            if r['prefix_hash'] == req['prefix_hash']
        )
        overlaps.append(overlap / len(window_reqs))
    
    return np.mean(overlaps) if overlaps else 0.0

def analyze_window_sweep(traces, window_range):
    """扫描不同窗口大小的重叠率"""
    results = []
    for window_ms in window_range:
        overlap_rate = temporal_prefix_overlap(traces, window_ms)
        results.append({
            'window_ms': window_ms,
            'overlap_rate': overlap_rate
        })
        print(f"Window {window_ms}ms: overlap_rate = {overlap_rate:.3f}")
    return results

def plot_results(results):
    """绘制 window_ms vs overlap_rate 曲线"""
    window_sizes = [r['window_ms'] for r in results]
    overlap_rates = [r['overlap_rate'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes, overlap_rates, marker='o', linewidth=2)
    plt.xlabel('Window Size (ms)', fontsize=12)
    plt.ylabel('Prefix Overlap Rate', fontsize=12)
    plt.title('Temporal Prefix Locality Analysis', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 标注关键点
    max_idx = np.argmax(overlap_rates)
    plt.annotate(
        f'Peak: {window_sizes[max_idx]}ms\n{overlap_rates[max_idx]:.3f}',
        xy=(window_sizes[max_idx], overlap_rates[max_idx]),
        xytext=(10, 10), textcoords='offset points',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
    )
    
    plt.savefig('/tmp/sglang_traces/temporal_locality.png', dpi=300, bbox_inches='tight')
    print("Plot saved to /tmp/sglang_traces/temporal_locality.png")

if __name__ == '__main__':
    # 加载数据
    traces = load_traces('/tmp/sglang_traces/temporal_analysis.jsonl')
    print(f"Loaded {len(traces)} requests")
    
    # 扫描窗口大小: 0-100ms, 步长5ms
    window_range = list(range(0, 105, 5))
    results = analyze_window_sweep(traces, window_range)
    
    # 绘图
    plot_results(results)
    
    # 输出统计信息
    print("\n=== Summary ===")
    print(f"Total requests: {len(traces)}")
    print(f"Unique prefixes: {len(set(r['prefix_hash'] for r in traces))}")
    print(f"Time span: {traces[-1]['arrival_ts'] - traces[0]['arrival_ts']:.2f}s")
