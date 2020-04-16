#pragma once
#include <array>
#include <cstdint>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <gsl/span>
#include <execution>

static std::default_random_engine defaultRandomEngine{ std::random_device{}() };

template<size_t NumInputNodes, size_t NumHiddenNodes, size_t NumOutputNodes>
struct NeuralNetwork
{
public:
	NeuralNetwork(float learningRate) : learningRate{ learningRate }
	{
		static std::normal_distribution<float> normalDistributionHidden{ 0.0, 1 / std::sqrtf(static_cast<float>(NumHiddenNodes)) };
		static std::normal_distribution<float> normalDistributionOutput{ 0.0, 1 / std::sqrtf(static_cast<float>(NumOutputNodes)) };

		std::for_each(std::execution::par_unseq, std::begin(inputWeights), std::end(inputWeights), [](std::array<float, NumInputNodes>& row)
			{
				std::generate(std::execution::seq, std::begin(row), std::end(row), []() { return normalDistributionHidden(defaultRandomEngine); });
			});

		std::for_each(std::execution::par_unseq, std::begin(outputWeights), std::end(outputWeights), [](std::array<float, NumHiddenNodes>& row)
			{
				std::generate(std::execution::seq, std::begin(row), std::end(row), []() { return normalDistributionOutput(defaultRandomEngine); });
			});
	}

	NeuralNetwork(const NeuralNetwork& other) = delete;
	NeuralNetwork& operator=(const NeuralNetwork& other) = delete;
	NeuralNetwork(NeuralNetwork&& other) = delete;
	NeuralNetwork& operator=(NeuralNetwork&& other) = delete;
	~NeuralNetwork() = default;

	constexpr static size_t numInputNodes = NumInputNodes;
	constexpr static size_t numHiddenNodes = NumHiddenNodes;
	constexpr static size_t numOutputNodes = NumOutputNodes;

	float learningRate;
	std::array<std::array<float, NumInputNodes>, NumHiddenNodes> inputWeights;
	std::array<std::array<float, NumHiddenNodes>, NumOutputNodes> outputWeights;
};

template<size_t NumInputNodes, size_t NumHiddenNodes, size_t NumOutputNodes>
auto Query(const NeuralNetwork<NumInputNodes, NumHiddenNodes, NumOutputNodes>& neuralNetwork, 
	gsl::span<float, NumInputNodes> input) -> std::array<float, NumOutputNodes>
{
	std::array<float, NumHiddenNodes> hiddenValues{};

	for (size_t i{}; i < NumHiddenNodes; ++i)
	{
		for (size_t j{}; j < NumInputNodes; ++j)
		{
			hiddenValues[i] += input[j] * neuralNetwork.inputWeights[i][j];
		}
	}

	for (auto& hiddenValue : hiddenValues)
	{
		hiddenValue = 1 / (1 + std::exp(-hiddenValue));
	}

	std::array<float, NumOutputNodes> outputValues{};

	for (size_t i{}; i < NumOutputNodes; ++i)
	{
		for (size_t j{}; j < NumHiddenNodes; ++j)
		{
			outputValues[i] += hiddenValues[j] * neuralNetwork.outputWeights[i][j];
		}
	}

	for (auto& outputValue : outputValues)
	{
		outputValue = 1 / (1 + std::exp(-outputValue));
	}

	return outputValues;
}

template<size_t NumInputNodes, size_t NumHiddenNodes, size_t NumOutputNodes>
void Train(NeuralNetwork<NumInputNodes, NumHiddenNodes, NumOutputNodes>& neuralNetwork,
	gsl::span<float, NumInputNodes> input,
	gsl::span<float, NumOutputNodes> target)
{
	std::array<float, NumHiddenNodes> hiddenValues{};

	std::generate(std::execution::seq, std::begin(hiddenValues), std::end(hiddenValues), [&input, &neuralNetwork, i = 0]() mutable
		{
			auto out =  std::transform_reduce(
				std::execution::seq,
				std::begin(input),
				std::end(input),
				std::begin(neuralNetwork.inputWeights[i]),
				0.0f,
				std::plus<>(),
				std::multiplies<>());
			++i;
			return out;
		});

	std::transform(std::execution::seq, std::begin(hiddenValues), std::end(hiddenValues), std::begin(hiddenValues), [](float value)
		{
			return 1 / (1 + std::exp(-value));
		});

	std::array<float, NumOutputNodes> outputValues{};

	std::generate(std::execution::seq, std::begin(outputValues), std::end(outputValues), [&hiddenValues, &neuralNetwork, i = 0]() mutable
		{

			auto out = std::transform_reduce(
				std::execution::seq,
				std::begin(hiddenValues),
				std::end(hiddenValues),
				std::begin(neuralNetwork.outputWeights[i]),
				0.0f,
				std::plus<>(),
				std::multiplies<>());

			++i;
			return out;
		});

	std::transform(std::execution::seq, std::begin(outputValues), std::end(outputValues), std::begin(outputValues), [](float value)
		{
			return 1 / (1 + std::exp(-value));
		});

	std::array<float, NumOutputNodes> outputErrorValues{};
	std::transform(std::execution::seq, std::begin(target), std::end(target), std::begin(outputValues), std::begin(outputErrorValues), std::minus<>());

	std::array<float, NumHiddenNodes> hiddenErrorValues{};
	for (size_t i{}; i < NumHiddenNodes; ++i)
	{
		for (size_t j{}; j < NumOutputNodes; ++j)
		{
			hiddenErrorValues[i] += neuralNetwork.outputWeights[j][i] * outputErrorValues[j];
		}
	}

	for (size_t i{}; i < NumOutputNodes; ++i)
	{
		for (size_t j{}; j < NumHiddenNodes; ++j)
		{
			neuralNetwork.outputWeights[i][j] += neuralNetwork.learningRate * (outputErrorValues[i] * outputValues[i] * (1 - outputValues[i]) * hiddenValues[j]);
		}
	}

	for (size_t i{}; i < NumHiddenNodes; ++i)
	{
		for (size_t j{}; j < NumInputNodes; ++j)
		{
			neuralNetwork.inputWeights[i][j] += neuralNetwork.learningRate * (hiddenErrorValues[i] * hiddenValues[i] * (1 - hiddenValues[i]) * input[j]);
		}
	}
}