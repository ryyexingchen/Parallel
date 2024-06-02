#include <iostream>
#include <pthread.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <random>
#include <pmmintrin.h> //SSE3
#include <windows.h>
#include <stdlib.h>
#include <fstream>
using namespace std;

const int N = 2048;
const int NUM_THREADS = 16;
typedef struct {
    int t_id; // 线程 id
} threadParam_t;

pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;
pthread_barrier_t barrier0;
float** A = nullptr;
float* b = nullptr;
float C[N][N];
float d[N];
bool judge = true;
void Init() {
	if(A == nullptr || b == nullptr){// 为A和b分配内存
		A = new float*[N];
		for (int i = 0; i < N; ++i)
			A[i] = new float[N];
		b = new float[N];
	}
	if(judge){
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

// 线程函数定义
void* threadFunc1(void* param) {  
    threadParam_t* p = static_cast<threadParam_t*>(param);  
    int t_id = p->t_id;  
	for (int k = 0; k < N; ++k) {
		int columns_per_thread = (N - (k + 1)) / NUM_THREADS; // 每线程处理的列数  
		int start_col = (k + 1) + t_id * columns_per_thread; // 线程处理的起始列  
		int end_col = min(N, start_col + columns_per_thread); // 线程处理的结束列  
    	if(t_id == 0){
			// t_id = 0的线程做除法操作
			for (int j = k + 1; j < N; j++) {
				A[k][j] = A[k][j] / A[k][k];
			}		
			b[k] /= A[k][k];
			A[k][k] = 1.0;
		}
		//第一个同步点
		pthread_barrier_wait(&barrier_Divsion);
		// 动态列划分处理消元步骤  
		for (int i = k + 1; i < N; ++i) {
			if(t_id == 0){
				b[i] -= A[i][k] * b[k];
			}
            // 更新矩阵A的消元操作
            for (int j = start_col; j < end_col; ++j) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }			
        }
		// 第二个同步点，确保所有线程都完成了消元步骤  
		pthread_barrier_wait(&barrier_Elimination);  
		if(t_id == 0){
			for (int i = k + 1; i < N; ++i){
				A[i][k] = 0.0;
			}
		}
    }  
    pthread_exit(nullptr);  
}  

float* pthread1(){
	pthread_barrier_init(&barrier_Divsion,nullptr,NUM_THREADS);
	pthread_barrier_init(&barrier_Elimination,nullptr,NUM_THREADS);
	pthread_barrier_init(&barrier0,nullptr,NUM_THREADS);
	//创建线程
	pthread_t handles[NUM_THREADS];//创建对应的Handle
	threadParam_t param[NUM_THREADS];//创建对应的数据结构
	for(int t_id = 0;t_id < NUM_THREADS;++t_id){
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], nullptr, threadFunc1, &param[t_id]);
	}
	//等待所有线程结束
	for(int t_id = 0;t_id < NUM_THREADS;++t_id){
		pthread_join(handles[t_id], nullptr);
	}
	//销毁所有的barrier
	pthread_barrier_destroy(&barrier_Divsion);
	pthread_barrier_destroy(&barrier_Elimination);
	pthread_barrier_destroy(&barrier0);
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
	for(int r = 0;r < loop;++r){
		Init();
		long long head1, tail1, freq1;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq1);
		QueryPerformanceCounter((LARGE_INTEGER*)&head1);
		float* x1 = serial();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail1);
		cout << "Round"<<r+1<<":Serial(" << N << "):" << (tail1 - head1) * 1000.0 / freq1 << "ms" << endl;
		
		Init();

		long long head2, tail2, freq2;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq2);
		QueryPerformanceCounter((LARGE_INTEGER*)&head2);
		float* x2 = pthread1();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
		cout << "Round"<<r+1<<":pthread1(" << N << "):" << (tail2 - head2) * 1000.0 / freq2 << "ms" << endl;	
		
		judge = true;
		cout<<endl;
	}

	// 恢复cout的原始缓冲区  
    cout.rdbuf(coutbuf);  
    // 关闭文件流  
    outputFile.close();  
	Free();

    return 0;
}