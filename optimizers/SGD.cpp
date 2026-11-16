#include "SGD.hpp"

SGD::SGD(std::vector<Parameter*> parameters, float learningRate, float momentum, float weightDecay)
    : Optimizer(std::move(parameters), learningRate),
      m_momentum(momentum),
      m_weightDecay(weightDecay) {}

void SGD::step() {
    for (Parameter* param : m_parameters) {
        if (!param || !param->requiresGrad() || param->data().empty()) {
            continue;
        }

        float* wData = param->data().data();
        const float* gData = param->grad().data();
        float* vData = (m_momentum != 0.0f) ? param->velocity().data() : nullptr;
        size_t n = param->data().size();

        CUDABackend::sgdUpdate(wData, gData, vData, n, m_learningRate, m_momentum, m_weightDecay, param->device());
    }
}

