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

const int N = 2048;
const int NUM_THREADS = 4;
typedef struct {
    int t_id; // 线程 id
} threadParam_t;
sem_t sem_main;
sem_t sem_workerstart[NUM_THREADS]; // 每个线程有自己专属的信号量
sem_t sem_workerend[NUM_THREADS];

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


// 线程函数定义
void* threadFunc1(void* param) {
    threadParam_t* p = static_cast<threadParam_t*>(param);
    int t_id = p->t_id;

    for (int k = 0; k < N; ++k) {
        sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作
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
        sem_post(&sem_main); // 唤醒主线程
        sem_wait(&sem_workerend[t_id]); //阻塞，等待主线程唤醒进入下一轮
    }
    pthread_exit(nullptr);
}

float* pthread1(){
	// 初始化信号量
    sem_init(&sem_main, 0, 0);
    sem_t sem_workerstart[NUM_THREADS], sem_workerend[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; ++i) {
        sem_init(&sem_workerstart[i], 0, 0);
        sem_init(&sem_workerend[i], 0, 0);
    }

    // 创建线程
    pthread_t handles[NUM_THREADS];
    threadParam_t param[NUM_THREADS];
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], nullptr, threadFunc1, &param[t_id]);
    }

    for (int k = 0; k < N; ++k) {
        // 主线程做除法操作
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
        // 开始唤醒工作线程
        for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
            sem_post(&sem_workerstart[t_id]);
        }
        // 主线程睡眠（等待所有的工作线程完成此轮消去任务）
        for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
            sem_wait(&sem_main);
        }
        // 主线程再次唤醒工作线程进入下一轮次的消去任务
        for (int t_id = 0; t_id < NUM_THREADS; ++t_id) {
            sem_post(&sem_workerend[t_id]);
        }
    }
    // 等待所有线程结束
    for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
        pthread_join(handles[t_id], nullptr);
    }
    // 销毁所有信号量
    sem_destroy(&sem_main);
    for (int i = 0; i < NUM_THREADS; ++i) {
        sem_destroy(&sem_workerstart[i]);
        sem_destroy(&sem_workerend[i]);
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
	Init();
	float* x2 = pthread1();
	Free();
    return 0;
}