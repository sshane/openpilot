#pragma clang diagnostic ignored "-Wexceptions"

#include "selfdrive/modeld/runners/snpemodel.h"

#include <cstring>

#include "common/util.h"
#include "common/timing.h"

void PrintErrorStringAndExit() {
  std::cerr << zdl::DlSystem::getLastErrorString() << std::endl;
  std::exit(EXIT_FAILURE);
}

SNPEModel::SNPEModel(const std::string path, float *_output, size_t _output_size, int runtime, bool _use_tf8, cl_context context) {
  output = _output;
  output_size = _output_size;
  use_tf8 = _use_tf8;

#ifdef QCOM2
  if (runtime == USE_GPU_RUNTIME) {
    snpe_runtime = zdl::DlSystem::Runtime_t::GPU;
  } else if (runtime == USE_DSP_RUNTIME) {
    snpe_runtime = zdl::DlSystem::Runtime_t::DSP;
  } else {
    snpe_runtime = zdl::DlSystem::Runtime_t::CPU;
  }
  LOGW("got runtime");
  assert(zdl::SNPE::SNPEFactory::isRuntimeAvailable(snpe_runtime));
#endif
  model_data = util::read_file(path);
  assert(model_data.size() > 0);

  // load model
  std::unique_ptr<zdl::DlContainer::IDlContainer> container = zdl::DlContainer::IDlContainer::open((uint8_t*)model_data.data(), model_data.size());
  if (!container) { PrintErrorStringAndExit(); }
  LOGW("loaded model with size: %lu", model_data.size());

  // create model runner
  zdl::SNPE::SNPEBuilder snpe_builder(container.get());
  LOGW("snpe_builder");
  while (!snpe) {
    LOGW("!snpe");
#ifdef QCOM2
    snpe = snpe_builder.setOutputLayers({})
                       .setRuntimeProcessor(snpe_runtime)
                       .setUseUserSuppliedBuffers(true)
                       .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
                       .build();
#else
    snpe = snpe_builder.setOutputLayers({})
                       .setUseUserSuppliedBuffers(true)
                       .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
                       .build();
#endif
//    if (!snpe) std::cerr << zdl::DlSystem::getLastErrorString() << std::endl;
    if (!snpe) LOGE(zdl::DlSystem::getLastErrorString());
  }

  LOGW("got snpe");
  // create output buffer
  zdl::DlSystem::UserBufferEncodingFloat ub_encoding_float;
  zdl::DlSystem::IUserBufferFactory &ub_factory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
  LOGW("created output buffer");

  const auto &output_tensor_names_opt = snpe->getOutputTensorNames();
  LOGW("debug print1");
  if (!output_tensor_names_opt) throw std::runtime_error("Error obtaining output tensor names");
  const auto &output_tensor_names = *output_tensor_names_opt;
  assert(output_tensor_names.size() == 1);
  const char *output_tensor_name = output_tensor_names.at(0);
  LOGW("debug print2");
  const zdl::DlSystem::TensorShape &buffer_shape = snpe->getInputOutputBufferAttributes(output_tensor_name)->getDims();
  LOGW("debug print3");
  if (output_size != 0) {
    assert(output_size == buffer_shape[1]);
  } else {
    output_size = buffer_shape[1];
  }
  std::vector<size_t> output_strides = {output_size * sizeof(float), sizeof(float)};
  LOGW("debug print4");
  output_buffer = ub_factory.createUserBuffer(output, output_size * sizeof(float), output_strides, &ub_encoding_float);
  LOGW("debug print5");
  output_map.add(output_tensor_name, output_buffer.get());
  LOGW("debug print6");

#ifdef USE_THNEED
  if (snpe_runtime == zdl::DlSystem::Runtime_t::GPU) {
    thneed.reset(new Thneed());
  }
#endif
}

void SNPEModel::addInput(const std::string name, float *buffer, int size) {
  const int idx = inputs.size();
  const auto &input_tensor_names_opt = snpe->getInputTensorNames();
  LOGW("got input tensor names");
  if (!input_tensor_names_opt) throw std::runtime_error("Error obtaining input tensor names");
  const auto &input_tensor_names = *input_tensor_names_opt;
  const char *input_tensor_name = input_tensor_names.at(idx);
  LOGW("got single tensor name");
  const bool input_tf8 = use_tf8 && strcmp(input_tensor_name, "input_img") == 0;  // TODO: This is a terrible hack, get rid of this name check both here and in onnx_runner.py
  LOGW("adding index %d: %s", idx, input_tensor_name);

  // Never got here online, shouldn't be the cause
  LOGW("new debug print1");
  zdl::DlSystem::UserBufferEncodingFloat ub_encoding_float;
  zdl::DlSystem::UserBufferEncodingTf8 ub_encoding_tf8(0, 1./255); // network takes 0-1
  zdl::DlSystem::IUserBufferFactory &ub_factory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
  zdl::DlSystem::UserBufferEncoding *input_encoding = input_tf8 ? (zdl::DlSystem::UserBufferEncoding*)&ub_encoding_tf8 : (zdl::DlSystem::UserBufferEncoding*)&ub_encoding_float;
  LOGW("new debug print2");

  const auto &buffer_shape_opt = snpe->getInputDimensions(input_tensor_name);
  const zdl::DlSystem::TensorShape &buffer_shape = *buffer_shape_opt;
  size_t size_of_input = input_tf8 ? sizeof(uint8_t) : sizeof(float);
  std::vector<size_t> strides(buffer_shape.rank());
  strides[strides.size() - 1] = size_of_input;
  size_t product = 1;
  LOGW("new debug print3");
  for (size_t i = 0; i < buffer_shape.rank(); i++) product *= buffer_shape[i];
  size_t stride = strides[strides.size() - 1];
  for (size_t i = buffer_shape.rank() - 1; i > 0; i--) {
    stride *= buffer_shape[i];
    strides[i-1] = stride;
  }
  LOGW("new debug print4");

  auto input_buffer = ub_factory.createUserBuffer(buffer, product*size_of_input, strides, input_encoding);
  LOGW("new debug print5");
  input_map.add(input_tensor_name, input_buffer.get());
  inputs.push_back(std::unique_ptr<SNPEModelInput>(new SNPEModelInput(name, buffer, size, std::move(input_buffer))));
  LOGW("new debug print6");
}

void SNPEModel::execute() {
  if (!snpe->execute(input_map, output_map)) {
    PrintErrorStringAndExit();
  }
}
