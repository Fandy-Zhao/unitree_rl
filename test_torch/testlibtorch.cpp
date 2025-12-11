#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

void print_model_structure(const torch::jit::script::Module& module) {
    std::cout << "=== Model Structure ===" << std::endl;
    
    // 打印参数名称和形状
    for (const auto& param : module.named_parameters()) {
        std::cout << "Layer: " << param.name;
        std::cout << " | Shape: [";
        auto sizes = param.value.sizes();
        for (int64_t i = 0; i < sizes.size(); ++i) {
            std::cout << sizes[i];
            if (i < sizes.size() - 1) std::cout << ", ";
        }
        std::cout << "]";
        std::cout << " | Type: " << param.value.dtype();
        std::cout << std::endl;
    }
    
    // 打印缓冲区信息 - 使用 size() 而不是 empty()
    auto buffers = module.named_buffers();
    if (buffers.size() > 0) {  // 修改这里
        std::cout << "\nBuffers:" << std::endl;
        for (const auto& buffer : buffers) {
            std::cout << "  " << buffer.name;
            std::cout << " | Shape: [";
            auto sizes = buffer.value.sizes();
            for (int64_t i = 0; i < sizes.size(); ++i) {
                std::cout << sizes[i];
                if (i < sizes.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
}


int main() {
    try {
        auto body_module = torch::jit::load("/home/zzf/RL/unitree_rl/src/unitree_guide/unitree_guide/model/body.jit");
        auto adapt_module = torch::jit::load("/home/zzf/RL/unitree_rl/src/unitree_guide/unitree_guide/model/adapt.jit");
        
        auto himloco_module = torch::jit::load("/home/zzf/RL/rl_sar/policy/go2/himloco/himloco.pt");

        auto extremeparkour_module = torch::jit::load("/home/zzf/RL/Extreme-Parkour-Onboard/traced/base_jit.pt");

        std::cout << "Body Model Structure:" << std::endl;
        print_model_structure(body_module);
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        
        std::cout << "Adapt Model Structure:" << std::endl;
        print_model_structure(adapt_module);

        std::cout << "\n" << std::string(50, '=') << std::endl;
        
        std::cout << "himloco Model Structure:" << std::endl;
        print_model_structure(himloco_module);

        std::cout << "\n" << std::string(50, '=') << std::endl;
        
        std::cout << "extremeparkour Model Structure:" << std::endl;
        print_model_structure(extremeparkour_module);
        
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
    }
    return 0;
}

