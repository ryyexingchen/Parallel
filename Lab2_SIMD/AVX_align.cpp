#include <immintrin.h> // AVX
#include <iostream>
#include <ctime>
#include <windows.h>
#include <stdlib.h>
#include <fstream>
#include<typeinfo>
using namespace std;

const int N = 4096;
float A[N][N];
float b[N];
float** A_align = NULL;
float C[N][N];
float d[N];
bool judge = true;
void m_reset() {
	if(judge){
		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++)
				C[j][i] = rand();
			d[i] = rand();
		}
		judge = false;
	}
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++)
			A[j][i] = C[j][i];
		b[i] = d[i];
	}
}

void m_reset_aligned(int alignment = 16) {//对齐矩阵的初始化
    if (A_align == NULL) {
        A_align = (float**)_aligned_malloc(sizeof(float*) * N, alignment);
        for (int i = 0; i < N; i++) {//使得矩阵每一行在内存中按照alignment对齐
            A_align[i] = (float*)_aligned_malloc(sizeof(float) * N, alignment);   
        }
    }
	if(judge){
		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++)
				C[j][i] = rand();
			d[i] = rand();
		}
		judge = false;
	}
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++)
			A_align[j][i] = C[j][i];
		b[i] = d[i];
	}
}


float* Gaussian_Elimination_serial() {
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

float* Gaussian_Elimination_parallel_unaligned() {
	//消去过程
	for (int k = 0; k < N; k++) { //归一化  
    __m256 vt = _mm256_set1_ps(A[k][k]); 
    int j = 0;  
    for (j = k + 1; j + 8 <= N; j += 8) { //每次处理8个浮点数  
        __m256 va = _mm256_loadu_ps(A[k] + j);   
        va = _mm256_div_ps(va, vt);
        _mm256_storeu_ps(A[k] + j, va);
    }  
    for (; j < N; j++) {  
        A[k][j] = A[k][j] / A[k][k];  
    }  
    b[k] = b[k] / A[k][k];  
    A[k][k] = 1.0;  
    for (int i = k + 1; i < N; i++) { //消元  
        __m256 vaik = _mm256_set1_ps(A[i][k]);  
  
        for (j = k + 1; j + 8 <= N; j += 8) { //每次处理8个浮点数  
            __m256 vakj = _mm256_loadu_ps(A[k] + j);   
            __m256 vaij = _mm256_loadu_ps(A[i] + j);  
            __m256 vx = _mm256_mul_ps(vaik, vakj);  
            vaij = _mm256_sub_ps(vaij, vx); 
            _mm256_storeu_ps(A[i] + j, vaij); 
        }  
        for (; j < N; j++) {  
            A[i][j] = A[i][j] - A[i][k] * A[k][j];  
        }  
        b[i] -= b[k] * A[i][k];  
        A[i][k] = 0;  
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

float* Gaussian_Elimination_parallel_aligned() {
	//消去过程
    for (int k = 0; k < N; k++) { // 归一化  
        __m256 vt = _mm256_set1_ps(A_align[k][k]);  
        int j = k + 1;  
        while ((long long)(&A_align[k][j]) % 32) { // 检查每一行最前面的元素是否对齐，手动处理未对齐的元素   
            A_align[k][j] = A_align[k][j] / A_align[k][k];  
            j++;  
        }  
        for (; j + 8 <= N; j += 8) { // AVX一次处理8个浮点数  
            __m256 va = _mm256_load_ps(A_align[k] + j);  
            va = _mm256_div_ps(va, vt);  
            _mm256_store_ps(A_align[k] + j, va);  
        }  
        for (; j < N; j++) { // 处理剩下的元素  
            A_align[k][j] = A_align[k][j] / A_align[k][k];  
        }  
        b[k] = b[k] / A_align[k][k];    
        A_align[k][k] = 1.0;    
        for (int i = k + 1; i < N; i++) { // 消元  
            __m256 vaik = _mm256_set1_ps(A_align[i][k]);  
            j = k + 1;  
            while ((long long)(&A_align[k][j]) % 32) { // 检查每一行最前面的元素是否对齐，手动处理未对齐的元素  
                A_align[i][j] -= A_align[i][k] * A_align[k][j];  
                j++;  
            }  
            for (; j + 8 <= N; j += 8) { // AVX一次处理8个浮点数  
                __m256 vakj = _mm256_load_ps(A_align[k] + j);    
                __m256 vaij = _mm256_loadu_ps(A_align[i] + j);  
                __m256 vx = _mm256_mul_ps(vaik, vakj);  
                vaij = _mm256_sub_ps(vaij, vx);  
                _mm256_storeu_ps(A_align[i] + j, vaij);  
            }  
            for (; j < N; j++) { // 处理剩下的元素  
                A_align[i][j] -= A_align[i][k] * A_align[k][j];  
            }  
            b[i] -= b[k] * A_align[i][k];   
            A_align[i][k] = 0;  
        }  
    }  
	//回带过程
	float* x = new float[N];
	for (int i = 0; i < N; i++)
		x[i] = 0.0;
	for (int i = N - 1; i >= 0; i--) {
		float sum = b[i];
		for (int j = i + 1; j < N; j++)
			sum -= A_align[i][j] * x[j];
		x[i] = sum / A_align[i][i];
	}
	return x;
}



int main() {
	int loop = 10;
	ofstream outputFile("output.txt"); // 创建一个输出文件流对象  
  
    if (!outputFile.is_open()) {  
        cerr << "Failed to open the output file!" << endl;  
        return 1; // 如果文件打开失败，返回错误码  
    }  
	
    // 保存原始cout的缓冲区指针  
    streambuf* coutbuf = cout.rdbuf();  
  
    // 重定向cout到文件  
    cout.rdbuf(outputFile.rdbuf());  
	for(int i = 0;i < loop;i++){
		m_reset();
		long long head1, tail1, freq1;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq1);
		QueryPerformanceCounter((LARGE_INTEGER*)&head1);
		float* x1 = Gaussian_Elimination_serial();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail1);
		//cout << "x=[";
		//for (int i = 0; i < N; i++) {
		//	cout << x1[i];
		//	if (i != N - 1)  cout << ",";
		//}
		//cout << "]" << endl;
		cout << "Serial(" << N << "):" << (tail1 - head1) * 1000.0 / freq1 << "ms" << endl;

		m_reset();
		long long head2, tail2, freq2;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq2);
		QueryPerformanceCounter((LARGE_INTEGER*)&head2);
		float* x2 = Gaussian_Elimination_parallel_unaligned();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
		//cout << "x=[";
		//for (int i = 0; i < N; i++) {
		//	cout << x2[i];
		//	if (i != N - 1)  cout << ",";
		//}
		//cout << "]" << endl;
		cout << "Parallel_unaligned(" << N << "):" << (tail2 - head2) * 1000.0 / freq2 << "ms" << endl;

		m_reset_aligned();
		long long head3, tail3, freq3;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq3);
		QueryPerformanceCounter((LARGE_INTEGER*)&head3);
		float* x3 = Gaussian_Elimination_parallel_aligned();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail3);
		//cout << "x=[";
		//for (int i = 0; i < N; i++) {
		//	cout << x3[i];
		//	if (i != N - 1)  cout << ",";
		//}	
		//cout << "]" << endl;
		cout << "Parallel_aligned(" << N << "):" << (tail3 - head3) * 1000.0 / freq3 << "ms" << endl;
	}
	 // 恢复cout的原始缓冲区  
    cout.rdbuf(coutbuf);  
  
    // 关闭文件流  
    outputFile.close();
		
	
	return 0;
}