#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <random>
#include <pmmintrin.h> //SSE3
#include <windows.h>
#include <stdlib.h>
#include <fstream>
#include <omp.h>
using namespace std;

const int N = 2048;
const int NUM_THREADS = 8;
const bool parallel = true;

float** A = nullptr;
float* b = nullptr;
float C[N][N];
float d[N];
bool judge = true;
void Init() {
	if (A == nullptr || b == nullptr) {// 为A和b分配内存
		A = new float* [N];
		for (int i = 0; i < N; ++i)
			A[i] = new float[N];
		b = new float[N];
	}
	if (judge) {
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j)
				C[j][i] = rand() % 100;
			d[i] = rand() % 100;
		}
		judge = false;
	}
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j)
			A[j][i] = C[j][i];
		b[i] = d[i];
	}
}

// 释放内存函数
void Free() {
	for (int i = 0; i < N; ++i) {
		delete[] A[i];
	}
	delete[] A;
	delete[] b;
	A = nullptr;
	b = nullptr;
}

float* serial() {
	//原始串行算法
	//消去过程
	for (int k = 0; k < N; k++) {
		for (int i = k + 1; i < N; i++) {
			float factor = A[i][k] / A[k][k];
			for (int j = k + 1; j < N; j++)
				A[i][j] -= factor * A[k][j];
			b[i] -= factor * b[k];
		}
	}
	//回带过程
	float* x = new float[N];
	for (int i = 0; i < N; i++)
		x[i] = 0.0;
	for (int i = N - 1; i >= 0; i--) {
		float sum = b[i];
		for (int j = i + 1; j < N; j++)
			sum -= A[i][j] * x[j];
		x[i] = sum / A[i][i];
	}
	return x;
}

float* OpenMP() {
	int i, j, k;
	float tmp;
	// 在外循环之外创建线程，避免线程反复创建销毁
#pragma omp parallel if(parallel) num_threads(NUM_THREADS) private(i, j, k, tmp)
	{
		for (k = 0; k < N; ++k) {
			tmp = A[k][k];
#pragma omp for //使用行划分
			for (j = k + 1; j < N; ++j) 
				A[k][j] = A[k][j] / tmp;
			b[k] /= A[k][k];
			A[k][k] = 1.0;
#pragma omp for //使用行划分
			for (i = k + 1; i < N; ++i) {
				tmp = A[i][k];
				for (j = k + 1; j < N; ++j) {
					A[i][j] -= tmp * A[k][j];
				}
				b[i] -= tmp * b[k];
				A[i][k] = 0.0;
			}
		}
	}
	//回带过程
	float* x = new float[N];
	for (int i = 0; i < N; i++)
		x[i] = 0.0;
	for (int i = N - 1; i >= 0; i--) {
		float sum = b[i];
		for (int j = i + 1; j < N; j++)
			sum -= A[i][j] * x[j];
		x[i] = sum / A[i][i];
	}
	return x;
}

float* OpenMP_simd() {
	// 在外循环之外创建线程，避免线程反复创建销毁
#pragma omp parallel num_threads(NUM_THREADS)
	{
		// 每个线程计算自己的临时变量tmp
		float tmp;
		for (int k = 0; k < N; ++k) {
			tmp = A[k][k];

			// SIMD化处理除法操作
#pragma omp simd
			for (int j = k + 1; j < N; ++j) {
				A[k][j] = A[k][j] / tmp;
			}
			// 标量处理剩余操作
			b[k] /= tmp;
			A[k][k] = 1.0f;
			// 行划分处理消元操作
#pragma omp for
			for (int i = k + 1; i < N; ++i) {
				float multiplier = A[i][k];
#pragma omp simd
				for (int j = k + 1; j < N; ++j) {
					A[i][j] -= multiplier * A[k][j];
				}
				b[i] -= multiplier * b[k];
				A[i][k] = 0.0f;
			}
		}
	}
	//回带过程
	float* x = new float[N];
	for (int i = 0; i < N; i++)
		x[i] = 0.0;
	for (int i = N - 1; i >= 0; i--) {
		float sum = b[i];
		for (int j = i + 1; j < N; j++)
			sum -= A[i][j] * x[j];
		x[i] = sum / A[i][i];
	}
	return x;
}

float* OpenMP_dyn() {
	int i, j, k;
	float tmp;
	// 在外循环之外创建线程，避免线程反复创建销毁
#pragma omp parallel if(parallel) num_threads(NUM_THREADS) private(i, j, k, tmp)
	{
		for (k = 0; k < N; ++k) {
			tmp = A[k][k];
#pragma omp for schedule(dynamic)//使用行划分
			for (j = k + 1; j < N; ++j)
				A[k][j] = A[k][j] / tmp;
			b[k] /= A[k][k];
			A[k][k] = 1.0;
#pragma omp for schedule(dynamic)//使用行划分
			for (i = k + 1; i < N; ++i) {
				tmp = A[i][k];
				for (j = k + 1; j < N; ++j) {
					A[i][j] -= tmp * A[k][j];
				}
				b[i] -= tmp * b[k];
				A[i][k] = 0.0;
			}
		}
	}
	//回带过程
	float* x = new float[N];
	for (int i = 0; i < N; i++)
		x[i] = 0.0;
	for (int i = N - 1; i >= 0; i--) {
		float sum = b[i];
		for (int j = i + 1; j < N; j++)
			sum -= A[i][j] * x[j];
		x[i] = sum / A[i][i];
	}
	return x;
}

float* OpenMP_guided() {
	int i, j, k;
	float tmp;
	// 在外循环之外创建线程，避免线程反复创建销毁
#pragma omp parallel if(parallel) num_threads(NUM_THREADS) private(i, j, k, tmp)
	{
		for (k = 0; k < N; ++k) {
			tmp = A[k][k];
#pragma omp for schedule(guided)//使用行划分
			for (j = k + 1; j < N; ++j)
				A[k][j] = A[k][j] / tmp;
			b[k] /= A[k][k];
			A[k][k] = 1.0;
#pragma omp for schedule(guided)//使用行划分
			for (i = k + 1; i < N; ++i) {
				tmp = A[i][k];
				for (j = k + 1; j < N; ++j) {
					A[i][j] -= tmp * A[k][j];
				}
				b[i] -= tmp * b[k];
				A[i][k] = 0.0;
			}
		}
	}
	//回带过程
	float* x = new float[N];
	for (int i = 0; i < N; i++)
		x[i] = 0.0;
	for (int i = N - 1; i >= 0; i--) {
		float sum = b[i];
		for (int j = i + 1; j < N; j++)
			sum -= A[i][j] * x[j];
		x[i] = sum / A[i][i];
	}
	return x;
}

float* OpenMP_simd_guided() {
	// 在外循环之外创建线程，避免线程反复创建销毁
#pragma omp parallel num_threads(NUM_THREADS)
	{
		// 每个线程计算自己的临时变量tmp
		float tmp;
		for (int k = 0; k < N; ++k) {
			tmp = A[k][k];

			// SIMD化处理除法操作
#pragma omp simd
			for (int j = k + 1; j < N; ++j) {
				A[k][j] = A[k][j] / tmp;
			}
			// 标量处理剩余操作
			b[k] /= tmp;
			A[k][k] = 1.0f;
			// 行划分处理消元操作
#pragma omp for schedule(guided)
			for (int i = k + 1; i < N; ++i) {
				float multiplier = A[i][k];
#pragma omp simd
				for (int j = k + 1; j < N; ++j) {
					A[i][j] -= multiplier * A[k][j];
				}
				b[i] -= multiplier * b[k];
				A[i][k] = 0.0f;
			}
		}
	}
	//回带过程
	float* x = new float[N];
	for (int i = 0; i < N; i++)
		x[i] = 0.0;
	for (int i = N - 1; i >= 0; i--) {
		float sum = b[i];
		for (int j = i + 1; j < N; j++)
			sum -= A[i][j] * x[j];
		x[i] = sum / A[i][i];
	}
	return x;
}

int main() {
	int loop = 11;
	ofstream outputFile("output.txt"); // 创建一个输出文件流对象  

	if (!outputFile.is_open()) {
		cerr << "Failed to open the output file!" << endl;
		return 1; // 如果文件打开失败，返回错误码  
	}

	// 保存原始cout的缓冲区指针  
	streambuf* coutbuf = cout.rdbuf();

	// 重定向cout到文件  
	cout.rdbuf(outputFile.rdbuf());
	for (int r = 0; r < loop; ++r) {
		Init();
		long long head1, tail1, freq1;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq1);
		QueryPerformanceCounter((LARGE_INTEGER*)&head1);
		float* x1 = serial();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail1);
		/*cout << "x=[";
		for (int i = 0; i < N; i++) {
			cout << x1[i];
			if (i != N - 1)  cout << ",";
		}
		cout << "]" << endl;*/
		cout << "Round" << r << ":Serial(" << N << "):" << (tail1 - head1) * 1000.0 / freq1 << "ms" << endl;

		Init();
		long long head2, tail2, freq2;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq2);
		QueryPerformanceCounter((LARGE_INTEGER*)&head2);
		float* x2 = OpenMP();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
		/*cout << "x=[";
		for (int i = 0; i < N; i++) {
			cout << x2[i];
			if (i != N - 1)  cout << ",";
		}
		cout << "]" << endl;*/
		cout << "Round" << r << ":OpenMP(" << N << "):" << (tail2 - head2) * 1000.0 / freq2 << "ms" << endl;

		Init();
		long long head3, tail3, freq3;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq3);
		QueryPerformanceCounter((LARGE_INTEGER*)&head3);
		float* x3 = OpenMP_dyn();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail3);
		/*cout << "x=[";
		for (int i = 0; i < N; i++) {
			cout << x3[i];
			if (i != N - 1)  cout << ",";
		}
		cout << "]" << endl;*/
		cout << "Round" << r << ":OpenMP_dyn(" << N << "):" << (tail3 - head3) * 1000.0 / freq3 << "ms" << endl;

		Init();
		long long head4, tail4, freq4;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq4);
		QueryPerformanceCounter((LARGE_INTEGER*)&head4);
		float* x4 = OpenMP_guided();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail4);
		/*cout << "x=[";
		for (int i = 0; i < N; i++) {
			cout << x4[i];
			if (i != N - 1)  cout << ",";
		}
		cout << "]" << endl;*/
		cout << "Round" << r << ":OpenMP_guided(" << N << "):" << (tail4 - head4) * 1000.0 / freq4 << "ms" << endl;
		
		Init();
		long long head5, tail5, freq5;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq5);
		QueryPerformanceCounter((LARGE_INTEGER*)&head5);
		float* x5 = OpenMP_simd();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail5);
		/*cout << "x=[";
		for (int i = 0; i < N; i++) {
			cout << x5[i];
			if (i != N - 1)  cout << ",";
		}
		cout << "]" << endl;*/
		cout << "Round" << r << ":OpenMP_simd(" << N << "):" << (tail5 - head5) * 1000.0 / freq5 << "ms" << endl;
		
		Init();
		long long head6, tail6, freq6;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq6);
		QueryPerformanceCounter((LARGE_INTEGER*)&head6);
		float* x6 = OpenMP_simd_guided();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail6);
		/*cout << "x=[";
		for (int i = 0; i < N; i++) {
			cout << x6[i];
			if (i != N - 1)  cout << ",";
		}
		cout << "]" << endl;*/
		cout << "Round" << r << ":OpenMP_simd_guided(" << N << "):" << (tail6 - head6) * 1000.0 / freq6 << "ms" << endl;
		
		judge = true;
		cout << endl;
	}

	// 恢复cout的原始缓冲区  
	cout.rdbuf(coutbuf);
	// 关闭文件流  
	outputFile.close();
	Free();

	return 0;
}


/*cout << "A3:" << endl;
for (int i = 0; i < N; i++) {
	for (int j = 0; j < N; j++)
		cout << A[i][j] << " ";
	cout << endl;
}
cout << "b3:" << endl;
for (int i = 0; i < N; i++)
	cout << b[i] << " ";
cout << endl;*/