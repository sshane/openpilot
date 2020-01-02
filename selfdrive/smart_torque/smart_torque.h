#include <SNPE/SNPE.hpp>
#include <SNPE/SNPEBuilder.hpp>
#include <SNPE/SNPEFactory.hpp>
#include <DlContainer/IDlContainer.hpp>
#include <DlSystem/DlError.hpp>
#include <DlSystem/ITensor.hpp>
#include <DlSystem/ITensorFactory.hpp>
#include <DlSystem/IUserBuffer.hpp>
#include <DlSystem/IUserBufferFactory.hpp>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <array>

extern "C"{
  void init_model();
  float run_model(float inputData[1000]);
}