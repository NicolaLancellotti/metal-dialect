kernel void life(device int32_t *matrix [[buffer(0)]],
                 device int32_t *sides [[buffer(1)]],
                 device int32_t *result [[buffer(2)]],
                 uint3 id [[thread_position_in_grid]]) {
  int32_t x = id.x;
  int32_t y = id.y;
  int32_t rows = sides[0];
  int32_t columns = sides[1];
  int32_t liveNeighbors = 0;
  
  for (int32_t i = x - 1; i <= x + 1; ++i) {
    for (int32_t j = y - 1; j <= y + 1; ++j) {
      if (i >= 0 && i < rows && j >= 0 && j < columns &&
          (i != x || j != y) && matrix[i * columns + j]) {
        liveNeighbors = liveNeighbors + 1;
      }
    }
  }
  
  int32_t index = x * columns + y;
  if (matrix[index]) {
    result[index] = (liveNeighbors == 2 || liveNeighbors == 3) ? 1 : 0;
  } else {
    result[index] = liveNeighbors == 3 ? 1 : 0;
  }
}

kernel void addArrays(constant const float* arg0 [[buffer(0)]],
                      constant const float* arg1 [[buffer(1)]],
                      device float *result [[buffer(2)]],
                      uint index [[thread_position_in_grid]]) {
  result[index] = arg0[index] + arg1[index];
}

