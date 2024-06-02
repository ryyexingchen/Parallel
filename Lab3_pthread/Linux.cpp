#include <arm_neon.h> //Neon
#include <iostream>
#include <stdlib.h>
#include<sys/time.h>
#include <time.h>
#include <stdio.h> 
#include <pthread.h>
using namespace std;

const int N = 512;
float A[N][N];
float b[N];
float C[N][N];
float d[N];
bool judge = true;
const int NUM_THREADS = 8;
typedef struct {
    int t_id; // 线程 id
} threadParam_t;

pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;

void m_reset() {
	if(judge){
		for (int i = 0; i < N; i++){
			for (int j = 0; j < N; j++)
				C[j][i] = rand() % 100;
			d[i] = rand() % 100;
		}
		judge = false;
	}
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++)
			A[j][i] = C[j][i];
		b[i] = d[i];
	}
}

float* Gaussian_Elimination_serial() {
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

float* Gaussian_Elimination_parallel() {
	//消去过程
	for(int k = 0; k < N; k++){
		//归一化
        float32x4_t vt = vdupq_n_f32(A[k][k]);
        int j = 0;
        for(j = k+1; j+4 <= N; j+=4){
            float32x4_t va = vld1q_f32(A[k]+j);
            va = vdivq_f32(va, vt);
            vst1q_f32(A[k]+j, va);
        }
        for( ;j < N; j++){
            A[k][j] = A[k][j]/A[k][k];
        }
		b[k] /= A[k][k];  
        A[k][k] = 1.0;
		
        for(int i = k + 1; i < N; i++){
                float32x4_t vaik = vdupq_n_f32(A[i][k]);
                for(j = k+1; j+4 <= N; j+=4){
                    float32x4_t vakj = vld1q_f32(A[k]+j);
                    float32x4_t vaij = vld1q_f32(A[i]+j);
                    float32x4_t vx = vmulq_f32(vakj, vaik);
                    vaij = vsubq_f32(vaij , vx);
                    vst1q_f32(A[i]+j , vaij);
				}
			for( ; j < N; j++){
				A[i][j] = A[i][j]-A[i][k]*A[k][j];
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

// 线程函数定义
void* threadFunc1(void* param) {
    threadParam_t* p = static_cast<threadParam_t*>(param);
    int t_id = p->t_id;

    for (int k = 0; k < N; ++k) {
		if(t_id == 0){
			// t_id = 0的线程做除法操作
			float32x4_t vt = vdupq_n_f32(A[k][k]);
			int j = 0;
			for (j = k + 1; j + 4 <= N; j += 4) {
				float32x4_t va = vld1q_f32(A[k]+j);
				va = vdivq_f32(va, vt);
				vst1q_f32(A[k]+j, va);
			}
			for (; j < N; j++) {
				A[k][j] = A[k][j] / A[k][k];
			}
			b[k] = b[k] / A[k][k];
			A[k][k] = 1.0;
		}
		//第一个同步点
		pthread_barrier_wait(&barrier_Divsion);
		for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {//消元
            float32x4_t vaik = vdupq_n_f32(A[i][k]);
			int j = k + 1;
            for (; j + 4 <= N; j += 4) {
                float32x4_t vakj = vld1q_f32(A[k]+j);
                float32x4_t vaij = vld1q_f32(A[i]+j);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij , vx);
                vst1q_f32(A[i]+j , vaij);
            }
            for (; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
			b[i] -= b[k] * A[i][k];
            A[i][k] = 0;
		}
        //第二个同步点
		pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(nullptr);
}

float* pthread1(){
	pthread_barrier_init(&barrier_Divsion,nullptr,NUM_THREADS);
	pthread_barrier_init(&barrier_Elimination,nullptr,NUM_THREADS);
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
	for(int i = 0;i < loop;i++){
		cout<<"i:"<<i<<endl;
		timeval* start1 = new timeval();
		timeval* stop1 = new timeval();
		double durationTime1 = 0.0;
		m_reset();
		gettimeofday(start1, NULL);
		float* x1 = Gaussian_Elimination_serial();
		gettimeofday(stop1, NULL);
		//cout << "x=[";
		//for (int i = 0; i < N; i++) {
		//	cout << x1[i];
		//	if (i != N - 1)  cout << ",";
		//}
		//cout << "]" << endl;
		durationTime1 = stop1->tv_sec * 1000 + double(stop1->tv_usec) / 1000 - start1->tv_sec * 1000 - double(start1->tv_usec) / 1000;
		cout << " serial time(" << N << "): " << double(durationTime1) << " ms" << endl;
		
	
		timeval* start2 = new timeval();
		timeval* stop2 = new timeval();
		double durationTime2 = 0.0;
		m_reset();
		gettimeofday(start2, NULL);
		float* x2 = pthread1();
		gettimeofday(stop2, NULL);
		//cout << "x=[";
		//for (int i = 0; i < N; i++) {
		//	cout << x2[i];
		//	if (i != N - 1)  cout << ",";
		//}
		//cout << "]" << endl;
		durationTime2 = stop2->tv_sec * 1000 + double(stop2->tv_usec) / 1000 - start2->tv_sec * 1000 - double(start2->tv_usec) / 1000;
		cout << " ParallelAlgorithm time(" << N << "): " << double(durationTime2) << " ms" << endl;
		cout <<"----------------------------" <<endl;
	}

	return 0;
}

