#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <random>
#include <pmmintrin.h> //SSE3
#include <windows.h>
#include <stdlib.h>
#include <fstream>
using namespace std;

const int N = 512;
const int NUM_THREADS = 4;
typedef struct {
    int t_id; // 线程 id
} threadParam_t;
sem_t sem_leader;
sem_t sem_Divsion[NUM_THREADS-1];
sem_t sem_Elimination[NUM_THREADS-1];

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
		if(t_id == 0){
			for (int k = 0; k < N; ++k) {
				// 0号线程做除法操作
				__m128 vt = _mm_set1_ps(A[k][k]);
				int j = 0;
				for (j = k + 1; j + 4 <= N; j += 4) {
					__m128 va = _mm_loadu_ps(A[k]+j);
					va = _mm_div_ps(va, vt);
					_mm_storeu_ps(A[k]+j, va);
				}
				for (; j < N; j++) {
					A[k][j] = A[k][j] / A[k][k];
				}
				b[k] = b[k] / A[k][k];
				A[k][k] = 1.0;
			}
		}
		else{
			sem_wait(&sem_Divsion[t_id-1]); // 阻塞，等待0号线程完成除法操作
		}
        if(t_id == 0){
			// 开始唤醒其他线程
			for (int id = 0; id < NUM_THREADS; ++id) {
				sem_post(&sem_Divsion[id]);
			}
		}
		for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {//消元
			__m128 vaik = _mm_set1_ps(A[i][k]);
			int j = k + 1;
            for (; j + 4 <= N; j += 4) {
                __m128 vakj = _mm_loadu_ps(A[k]+j);
                __m128 vaij = _mm_loadu_ps(A[i]+j);
                __m128 vx = _mm_mul_ps(vaik, vakj);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_storeu_ps(A[i]+j, vaij);
            }
            for (; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
			b[i] -= b[k] * A[i][k];
            A[i][k] = 0;
		}
    }
	if(t_id == 0){
		// 等待其他线程完成此轮消去任务
        for (int id = 0; id < NUM_THREADS - 1; ++id) {
            sem_wait(&sem_leader);
        }
		// 通知其他线程进入下一轮次的消去任务
        for (int id = 0; id < NUM_THREADS - 1; ++id) {
            sem_post(&sem_Elimination[id]);
        }
	}
	else{
		 sem_post(&sem_leader); // 唤醒主线程
        sem_wait(&sem_Elimination[t_id-1]); //阻塞，等待主线程唤醒进入下一轮
	}
    pthread_exit(nullptr);
}

float* pthread1(){
	// 初始化信号量
    sem_init(&sem_leader, 0, 0);
    for (int i = 0; i < NUM_THREADS-1; ++i) {
        sem_init(&sem_Divsion[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }

    // 创建线程
    pthread_t handles[NUM_THREADS];
    threadParam_t param[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], nullptr, threadFunc1, &param[t_id]);
    }
 
    // 等待所有线程结束
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], nullptr);
    }
    // 销毁所有信号量
    sem_destroy(&sem_leader);
    for (int i = 0; i < NUM_THREADS; ++i) {
        sem_destroy(&sem_Divsion[i]);
        sem_destroy(&sem_Elimination[i]);
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