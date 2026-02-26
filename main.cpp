#include "logging.h"
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <fstream>
#include <string>
#include <memory>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

const int INPUT_SIZE = 224;
const int OUTPUT_SIZE = 1000;


nvinfer1::ICudaEngine* createDeserializeCudaEngine(nvinfer1::IRuntime* runtime, const std::string model_file)
{
    std::ifstream in_file(model_file, std::ios::binary | std::ios::in);
    std::streampos begin, end;
    begin = in_file.tellg();
    in_file.seekg(0, std::ios::end);
    end = in_file.tellg();
    const std::size_t engine_size = end - begin;
    in_file.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> engine_data(new char[engine_size]);
    in_file.read((char*)engine_data.get(), engine_size);
    in_file.close();
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine((const void*)engine_data.get(), engine_size);
    return engine;
}



// 图像预处理
void preprocess(const cv::Mat& img, float* input) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(INPUT_SIZE, INPUT_SIZE));
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    resized.convertTo(resized, CV_32F, 1.0/255.0);
    cv::subtract(resized, cv::Scalar(0.485f, 0.456f, 0.406f), resized);
    cv::divide(resized, cv::Scalar(0.229f, 0.224f, 0.225f),resized);


    std::vector<cv::Mat> channels;
    cv::split(resized, channels);
    CV_Assert(channels[0].isContinuous() && channels[1].isContinuous() && channels[2].isContinuous());

    int dst_size = INPUT_SIZE * INPUT_SIZE;

    memcpy(input, channels[0].data, dst_size * sizeof(float));
    memcpy(input + dst_size, channels[1].data, dst_size * sizeof(float));
    memcpy(input + 2 * dst_size, channels[2].data, dst_size * sizeof(float));


    //    for (int i = 0; i < INPUT_SIZE * INPUT_SIZE * 3; i++) 
    //      input[i] = resized.data[i];
}


static void softmax(float* array, int size)
{
	// Find the maximum value in the array
	float max_val = array[0];
	for (int i = 1; i < size; i++)
	{
		if (array[i] > max_val)
			max_val = array[i];
	}

	// Subtract the maximum value from each element to avoid overflow
	for (int i = 0; i < size; i++)
		array[i] -= max_val;

	// Compute the exponentials and sum
	float sum = 0.0;
	for (int i = 0; i < size; i++)
	{
		array[i] = expf(array[i]);
		sum += array[i];
	}

	// Normalize the array by dividing each element by the sum
	for (int i = 0; i < size; i++)
		array[i] /= sum;
}


// 结果后处理
void postprocess(float* output, const std::vector<std::string>& labels) {	

	softmax(output,OUTPUT_SIZE);

	std::vector<std::pair<float, int>> probs;
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		probs.push_back({output[i], i});
	}
	std::sort(probs.rbegin(), probs.rend());

	std::cout << "Top 5 predictions:" << std::endl;
	for (int i = 0; i < 5; i++) {
		std::cout << labels[probs[i].second] << ": " << probs[i].first << std::endl;
	}
}



int main(int argc, const char* argv[])
{
	// deserialize TensorRT Engine
	sample::Logger logger;
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
	const std::string model_file = "models/resnet50.engine";
	nvinfer1::ICudaEngine* engine = createDeserializeCudaEngine(runtime, model_file);

	const int num_io_tensors = engine->getNbIOTensors();
	for (int i = 0; i < num_io_tensors; ++i) {
		const char* tensor_name = engine->getIOTensorName(i);
		std::cout << "Valid IO Tensor Name: " << tensor_name << std::endl;
	}

	// create context
	nvinfer1::IExecutionContext* context = engine->createExecutionContext();


	// 分配GPU内存
	const int input_elem_num = 1 * 3 * INPUT_SIZE * INPUT_SIZE;
	const int output_elem_num = 1 * OUTPUT_SIZE;
	//float* input = nullptr;
	//float* output = nullptr;
	std::vector<float> input(input_elem_num, 1.0f);
	std::vector<float> output(output_elem_num, 0.0f);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// allocate device memory
	float *d_input = nullptr;
	float *d_output = nullptr;
	cudaMallocAsync(&d_input, input_elem_num * sizeof(float), stream);
	cudaMallocAsync(&d_output, output_elem_num * sizeof(float), stream);


	// 加载并预处理图像
	cv::Mat img = cv::imread("images/tabby_tiger_cat.jpg"); //binoculars.jpeg");
	if (img.empty()) {
		std::cerr << "Failed to load image" << std::endl;
		return -1;
	}
	preprocess(img, input.data());
	std::cout<<"preprocess success!"<<std::endl;

	// copy HtoD
	cudaMemcpyAsync(d_input, input.data(), input_elem_num * sizeof(float), cudaMemcpyHostToDevice, stream);
	std::cout<<"copy HtoD  success!"<<std::endl;


	// inference
	context->setTensorAddress("input", d_input);
	context->setTensorAddress("output", d_output);
	bool status = context->enqueueV3(stream);
	std::cout<<"inference success!"<<std::endl;

	cudaStreamSynchronize(stream);

	// copy DtoH
	cudaMemcpyAsync(output.data(), d_output, output_elem_num * sizeof(float), cudaMemcpyDeviceToHost, stream);


	// 加载类别标签
	std::vector<std::string> labels;
	std::ifstream labelFile("class_labels.txt");
	std::string line;
	while (getline(labelFile, line)) {
		labels.push_back(line);
	}

	postprocess(output.data(), labels);
	std::cout<<"postprocess success!"<<std::endl;



	// release device memory
	cudaFree(d_input);
	cudaFree(d_output);

	cudaStreamDestroy(stream);

	return 0;
}
