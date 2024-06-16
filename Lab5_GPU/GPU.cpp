#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
// 定义T是float类型
typedef float T;
const int N = 2048;
// 定义核函数
__global__ void division_kernel(T* data, int k, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x; // 计算线程索引
    if (tid == k) { // 只有当tid等于k时，才进行除法操作
        T element = data[k*N+k]; 
        if (element != 0) { // 检查除数是否为0
            for (int i = k; i < N; i++) {
                data[k*N+i] = data[k*N+i] / element; // 除法操作
            }
        }
    }
}

__global__ void eliminate_kernel(T* data, int k, int N) {
    int tx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tx == 0) {
        data[k*N+k] = 1.0; // 对角线元素设为1
    }
    __syncthreads(); // 确保所有线程都完成了对角线元素的设置

    int row = k + blockIdx.x; // 每个块负责一行，从k+1开始
    for (int i = row; i < N; i++) {
        T factor = data[i*N+k] / data[k*N+k]; // 计算消去因子
        data[i*N+k] = 1.0; // 将本行的对角线元素设为1
        for (int j = k+1; j < N; j++) {
            data[i*N+j] -= factor * data[k*N+j]; // 消去操作
        }
    }
}

// 主函数，用于调用核函数
int main() {
    T* data_H; // 主机端数据
    T* data_D; // 设备端数据

    // 分配和初始化主机端数据
    data_H = (T*)malloc(N*N * sizeof(T));
    //初始化data_H的数据
	for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j)
				data_H[j][i] = rand();
		}
    // 分配设备端数据
    cudaMalloc(&data_D, N*N * sizeof(T));
    cudaMemcpy(data_D, data_H, N*N * sizeof(T), cudaMemcpyHostToDevice);

    dim3 grid((N-1)/32 + 1, 1, 1); // 假设每个线程块有32个线程
    dim3 block(32, 1, 1);

    // 核函数调用
    for (int k = 0; k < N; k++) {
        division_kernel<<<grid, block>>>(data_D, k, N);
        cudaDeviceSynchronize(); // CPU 与 GPU 之间的同步
        cudaError_t ret = cudaGetLastError();
        if (ret != cudaSuccess) {
            printf("division_kernel failed: %s\n", cudaGetErrorString(ret));
            return -1;
        }

        eliminate_kernel<<<grid, block>>>(data_D, k, N);
        cudaDeviceSynchronize();
        ret = cudaGetLastError();
        if (ret != cudaSuccess) {
            printf("eliminate_kernel failed: %s\n", cudaGetErrorString(ret));
            return -1;
        }
    }

    // 将结果从设备端复制回主机端
    cudaMemcpy(data_H, data_D, N*N * sizeof(T), cudaMemcpyDeviceToHost);

    // 释放设备端内存
    cudaFree(data_D);
    // 释放主机端内存
    free(data_H);

    return 0;
}