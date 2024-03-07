#include <functional>
#include <iostream>

void computeDuration(std::string text, std::function<void(void)> closure) {
  auto start = std::chrono::system_clock::now();
  closure();
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << text << elapsed.count() << "s\n";
}
