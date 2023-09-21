#import "MetalRuntime.h"
#include <iostream>

void computeDuration(std::string text, std::function<void(void)> closure);

void metal() {
  void lifeMetal();
  void lifeCPU();

  computeDuration("GPU Elapsed time: ", []() { lifeMetal(); });

  computeDuration("CPU Elapsed time: ", []() { lifeCPU(); });
}

void add() {
  void addMetal();

  computeDuration("GPU Elapsed time: ", []() { addMetal(); });
}

int main() {
  metal();
  //  add();
}
