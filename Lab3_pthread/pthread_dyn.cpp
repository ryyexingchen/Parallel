#include <cstdlib>
#include <ctime>
#include <random>
#include <pmmintrin.h> //SSE3
#include <iostream>
#include <windows.h>
#include <stdlib.h>
#include <fstream>
#include <pthread.h>
#include <vector>
using namespace std;

const int N = 2048;
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
				C[j][i] = rand();
			d[i] = rand();
		}
		judge = false;
	}		
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j)
			A[j][i] = C[j][i];
		b[i] = d[i];
	}
}

void Free() {
    // 释放内存
    for (int i = 0; i < N; ++i) {
        delete[] A[i];
    }
    delete[] A;
    delete[] b;
}

typedef struct {
    int k; // 消去的轮次
    int t_id; // 线程 id
} threadParam_t;

void* threadFunc_dyn(void* param) {
    threadParam_t* p = static_cast<threadParam_t*>(param);
    int k = p->k; // 消去的轮次
    int t_id = p->t_id; // 线程编号
    int i = k + t_id + 1; // 获取自己的计算任务

    for (; i < N; i++) {//消元
		int j = k + 1;
        __m128 vaik = _mm_set1_ps(A[i][k]);
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

    pthread_exit(NULL);
}
float* pthread_dyn(){//动态线程版本
	//消去过程
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
		b[k] /= A[k][k];
        A[k][k] = 1.0;
		// 创建工作线程，进行消去操作
		int worker_count = N - 1 - k;
		vector<pthread_t> handles(worker_count);
		vector<threadParam_t> p+arams(worker_count);
		// 分配任务
		for (int t_id = 0; t_id < worker_count; ++t_id) {
			params[t_id].k = k;
			params[t_id].t_id = t_id;
		}
		// 创建线程
		for (int t_id = 0; t_id < worker_count; ++t_id) {
			pthread_create(&handles[t_id], NULL, threadFunc_dyn, &params[t_id]);
		}
		// 主线程挂起等待所有的工作线程完成此轮消去工作
		for (int t_id = 0; t_id < worker_count; ++t_id) {
			pthread_join(handles[t_id], NULL);
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
		//cout << "x=[";
		//for (int i = 0; i < N; i++) {
		//	cout << x1[i];
		//	if (i != N - 1)  cout << ",";
		//}
		//cout << "]" << endl;
		cout << "Round"<<r<<":Serial(" << N << "):" << (tail1 - head1) * 1000.0 / freq1 << "ms" << endl;
		
		Init();
		long long head2, tail2, freq2;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq2);
		QueryPerformanceCounter((LARGE_INTEGER*)&head2);
		float* x2 = pthread_dyn();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
		//cout << "x=[";
		//for (int i = 0; i < N; i++) {
		//	cout << x2[i];
		//	if (i != N - 1)  cout << ",";
		//}
		//cout << "]" << endl;
		cout << "Round"<<r<<":Pthread_dyn(" << N << "):" << (tail2 - head2) * 1000.0 / freq2 << "ms" << endl;	
		
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