#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <torch/script.h>
#include <thread>
#include <pthread.h>
#include <sched.h>

class ModelBenchmark {
private:
    torch::jit::Module model_;
    std::string model_name_;
    
public:
    ModelBenchmark(const std::string& model_path, const std::string& name) 
        : model_name_(name) {
        try {
            std::cout << "Loading model: " << name << std::endl;
            model_ = torch::jit::load(model_path);
            model_.eval();
            std::cout << "Model loaded successfully: " << name << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model " << name << ": " << e.what() << std::endl;
            throw;
        }
    }
    
    struct BenchmarkResult {
        std::string model_name;
        long avg_time_us;
        long min_time_us;
        long max_time_us;
        long p95_time_us;
        long p99_time_us;
        double frequency_hz;
        int iterations;
        bool success;
        std::string error_msg;
    };
    
    BenchmarkResult run_benchmark(torch::Tensor input, int warmup_iterations = 10, int test_iterations = 1000, bool enable_thread_opt = false) {
        BenchmarkResult result;
        result.model_name = model_name_;
        result.iterations = test_iterations;
        result.success = true;
        
        try {
            if (enable_thread_opt) {
                std::cout << "\n[线程优化] 正在配置线程优化参数..." << std::endl;
                
                pthread_t this_thread = pthread_self();
                
                // 1. 设置实时调度策略和最高优先级
                struct sched_param params;
                params.sched_priority = sched_get_priority_max(SCHED_FIFO);
                int ret = pthread_setschedparam(this_thread, SCHED_FIFO, &params);
                if (ret == 0) {
                    std::cout << "[线程优化] ✓ 设置实时调度策略(SCHED_FIFO)成功，优先级: " << params.sched_priority << std::endl;
                } else {
                    std::cerr << "[线程优化] ✗ 设置实时调度失败(需要root权限或CAP_SYS_NICE)，错误码: " << ret << std::endl;
                }
                
                // 2. 绑定到特定CPU核心，避免线程迁移开销
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(0, &cpuset);
                ret = pthread_setaffinity_np(this_thread, sizeof(cpu_set_t), &cpuset);
                if (ret == 0) {
                    std::cout << "[线程优化] ✓ 绑定到CPU核心0成功" << std::endl;
                } else {
                    std::cerr << "[线程优化] ✗ 绑定CPU核心失败，错误码: " << ret << std::endl;
                }
                
                // 3. 设置线程为非抢占模式，减少上下文切换
                std::cout << "[线程优化] ✓ 线程优化配置完成\n" << std::endl;
            }
            
            std::cout << "预热 " << model_name_ << "，迭代 " << warmup_iterations << " 次..." << std::endl;
            for (int i = 0; i < warmup_iterations; ++i) {
                model_.forward({input});
            }
            
            std::vector<long> times;
            times.reserve(test_iterations);
            
            std::cout << "开始性能测试 " << model_name_ << "，迭代 " << test_iterations << " 次..." << std::endl;
            
            int slow_count = 0;
            const long threshold_us = 10000;    //设置超时时间
            
            for (int i = 0; i < test_iterations; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                auto output = model_.forward({input});
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                long time_us = duration.count();
                times.push_back(time_us);
                
                if (time_us > threshold_us) {
                    slow_count++;
                    std::cout << "  ⚠ 第 " << (i + 1) << " 次推理耗时异常: " << time_us << " μs (" << (time_us/1000.0) << " ms)" << std::endl;
                }
                
                if ((i + 1) % 100 == 0) {
                    std::cout << "  已完成 " << (i + 1) << "/" << test_iterations << " 次迭代" << std::endl;
                }
            }
            
            if (!times.empty()) {
                std::sort(times.begin(), times.end());
                long total = std::accumulate(times.begin(), times.end(), 0L);
                result.avg_time_us = total / times.size();
                result.min_time_us = times.front();
                result.max_time_us = times.back();
                result.p95_time_us = times[static_cast<int>(times.size() * 0.95)];
                result.p99_time_us = times[static_cast<int>(times.size() * 0.99)];
                result.frequency_hz = 1000000.0 / result.avg_time_us;
                
                std::cout << "\n异常慢推理统计 (>" << threshold_us << "μs): " << slow_count << "/" << test_iterations 
                          << " (" << (100.0*slow_count/test_iterations) << "%)" << std::endl;
            }
            
        } catch (const c10::Error& e) {
            result.success = false;
            result.error_msg = e.what();
            std::cerr << "✗ 测试出错 " << model_name_ << ": " << e.what() << std::endl;
        }
        
        return result;
    }
    
    void print_result(const BenchmarkResult& result) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "=== " << result.model_name << " 性能测试结果 ===" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        if (!result.success) {
            std::cout << "测试失败: " << result.error_msg << std::endl;
            return;
        }
        
        std::cout << "测试状态: 成功" << std::endl;
        std::cout << "测试迭代次数: " << result.iterations << std::endl;
        std::cout << "平均推理时间: " << result.avg_time_us << " μs" << std::endl;
        std::cout << "最小推理时间: " << result.min_time_us << " μs" << std::endl;
        std::cout << "最大推理时间: " << result.max_time_us << " μs" << std::endl;
        std::cout << "P95推理时间: " << result.p95_time_us << " μs" << std::endl;
        std::cout << "P99推理时间: " << result.p99_time_us << " μs" << std::endl;
        std::cout << "推理频率: " << std::fixed << std::setprecision(2) << result.frequency_hz << " Hz" << std::endl;
        
        // 时间单位转换
        std::cout << "\n其他时间单位:" << std::endl;
        std::cout << "  " << (result.avg_time_us / 1000.0) << " ms" << std::endl;
        std::cout << "  " << (result.avg_time_us / 1000000.0) << " s" << std::endl;
    }
};

void print_comparison_table(const std::vector<ModelBenchmark::BenchmarkResult>& results) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "=== 模型性能比较表 ===" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << std::left 
              << std::setw(20) << "模型名称"
              << std::setw(15) << "状态"
              << std::setw(15) << "平均时间(μs)"
              << std::setw(15) << "频率(Hz)"
              << std::setw(15) << "P95时间(μs)"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::left 
                  << std::setw(20) << result.model_name
                  << std::setw(15) << (result.success ? "成功" : "失败");
        
        if (result.success) {
            std::cout << std::setw(15) << result.avg_time_us
                      << std::setw(15) << std::fixed << std::setprecision(2) << result.frequency_hz
                      << std::setw(15) << result.p95_time_us;
        } else {
            std::cout << std::setw(15) << "N/A"
                      << std::setw(15) << "N/A"
                      << std::setw(15) << "N/A";
        }
        std::cout << std::endl;
    }
}

int main() {
    std::cout << " 开始模型性能基准测试..." << std::endl;
    // std::cout << "Torch版本: " << TORCH_VERSION << std::endl;
    
    std::vector<ModelBenchmark::BenchmarkResult> all_results;
    
    try {
        // 创建模型测试实例
        ModelBenchmark body_benchmark(
            "/home/zzf/RL/unitree_rl/src/unitree_guide/unitree_guide/model/body.jit", 
            "Body Model"
        );
        
        ModelBenchmark adapt_benchmark(
            "/home/zzf/RL/unitree_rl/src/unitree_guide/unitree_guide/model/adapt.jit", 
            "Adapt Model"
        );
        
        // 创建测试输入
        auto body_input = torch::randn({1, 66});
        auto adapt_input = torch::randn({1, 450});
        
        std::cout << "\n 输入张量形状:" << std::endl;
        std::cout << "  Body Model输入: [1, 66]" << std::endl;
        std::cout << "  Adapt Model输入: [1, 45]" << std::endl;
        
        // 运行基准测试
        auto body_result = body_benchmark.run_benchmark(body_input, 5, 5000, false); // 减少迭代次数用于测试
        auto adapt_result = adapt_benchmark.run_benchmark(adapt_input, 5, 5000, false);
        
        // 收集结果
        all_results.push_back(body_result);
        all_results.push_back(adapt_result);
        
        // 打印详细结果
        body_benchmark.print_result(body_result);
        adapt_benchmark.print_result(adapt_result);
        
        // 打印比较表
        print_comparison_table(all_results);
        
    } catch (const std::exception& e) {
        std::cerr << " 程序执行出错: " << e.what() << std::endl;
        return 1;
    }
    
    // 总结
    int success_count = 0;
    for (const auto& result : all_results) {
        if (result.success) success_count++;
    }
    
    std::cout << "\n 测试完成! " << success_count << "/" << all_results.size() << " 个模型测试成功" << std::endl;
    
    return 0;
}