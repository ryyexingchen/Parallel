#include <pmmintrin.h> //SSE3
#include <iostream>
#include <ctime>
#include <windows.h>
#include <stdlib.h>
#include <fstream>
using namespace std;

const int N = 512;
float A[N][N];
float b[N];
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

float* Gaussian_Elimination_parallel_10() {
	//只将前半段算法进行并行
	//消去过程
	for (int k = 0; k < N; k++) {//归一化
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
        for (int i = k + 1; i < N; i++) {//消元
            __m128 vaik = _mm_set1_ps(A[i][k]);
            for (j = k + 1; j + 4 <= N; j += 4) {
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

float* Gaussian_Elimination_parallel_01() {
	//只将后半段算法进行并行
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
	int i = 0;
	__m128 zero = _mm_setzero_ps();
	for (; i + 4 < N; i += 4)
		_mm_storeu_ps(x + i, zero);
	for (; i < N; i++)
		x[i] = 0.0;
	for (i = N - 1; i >= 0; i--) {
		float sum = b[i];
		__m128 sum_v = _mm_set1_ps(sum / 4);
		int j = i + 1;
		for (; j + 4 < N; j += 4) {
			//sum -= A[i][j] * x[j];
			__m128 a_v = _mm_loadu_ps(A[i] + j);
			__m128 x_v = _mm_loadu_ps(x + j);
			a_v = _mm_mul_ps(a_v, x_v);
			sum_v = _mm_sub_ps(sum_v, a_v);
		}
		sum_v = _mm_hadd_ps(sum_v, sum_v);
		sum_v = _mm_hadd_ps(sum_v, sum_v);
		_mm_store_ss(&sum, sum_v);
		for(;j < N;j++)
			sum -= A[i][j] * x[j];
		x[i] = sum / A[i][i];
	}
	return x;
}

float* Gaussian_Elimination_parallel_11() {
	//全部进行并行
	//消去过程
	for (int k = 0; k < N; k++) {//归一化
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
        for (int i = k + 1; i < N; i++) {//消元
            __m128 vaik = _mm_set1_ps(A[i][k]);
            for (j = k + 1; j + 4 <= N; j += 4) {
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
	//回带过程
	float* x = new float[N];
	int i = 0;
	__m128 zero = _mm_setzero_ps();
	for (; i + 4 < N; i += 4)
		_mm_storeu_ps(x + i, zero);
	for (; i < N; i++)
		x[i] = 0.0;
	for (i = N - 1; i >= 0; i--) {
		float sum = b[i];
		__m128 sum_v = _mm_set1_ps(sum / 4);
		int j = i + 1;
		for (; j + 4 < N; j += 4) {
			//sum -= A[i][j] * x[j];
			__m128 a_v = _mm_loadu_ps(A[i] + j);
			__m128 x_v = _mm_loadu_ps(x + j);
			a_v = _mm_mul_ps(a_v, x_v);
			sum_v = _mm_sub_ps(sum_v, a_v);
		}
		sum_v = _mm_hadd_ps(sum_v, sum_v);
		sum_v = _mm_hadd_ps(sum_v, sum_v);
		_mm_store_ss(&sum, sum_v);
		for (; j < N; j++)
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
		float* x2 = Gaussian_Elimination_parallel_10();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
		//cout << "x=[";
		//for (int i = 0; i < N; i++) {
		//	cout << x2[i];
		//	if (i != N - 1)  cout << ",";
		//}
		//cout << "]" << endl;
		cout << "Parallel_10(" << N << "):" << (tail2 - head2) * 1000.0 / freq2 << "ms" << endl;

		m_reset();
		long long head3, tail3, freq3;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq3);
		QueryPerformanceCounter((LARGE_INTEGER*)&head3);
		float* x3 = Gaussian_Elimination_parallel_01();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail3);
		//cout << "x=[";
		//for (int i = 0; i < N; i++) {
		//	cout << x3[i];
		//	if (i != N - 1)  cout << ",";
		//}
		//cout << "]" << endl;
		cout << "Parallel_01(" << N << "):" << (tail3 - head3) * 1000.0 / freq3 << "ms" << endl;
	
		m_reset();
		long long head4, tail4, freq4;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq4);
		QueryPerformanceCounter((LARGE_INTEGER*)&head4);
		float* x4 = Gaussian_Elimination_parallel_11();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail4);
		//cout << "x=[";
		//for (int i = 0; i < N; i++) {
		//	cout << x4[i];
		//	if (i != N - 1)  cout << ",";
		//}
		//cout << "]" << endl;
		cout << "Parallel_11(" << N << "):" << (tail4 - head4) * 1000.0 / freq4 << "ms" << endl;
	}
	 // 恢复cout的原始缓冲区  
    cout.rdbuf(coutbuf);  
  
    // 关闭文件流  
    outputFile.close();  

	return 0;
}

//cout << "A:" << endl;
//for (int i = 0; i < N; i++) {
//	for (int j = 0; j < N; j++)
//		cout << A[i][j] << " ";
//	cout << endl;
//}
//cout << "b:" << endl;
//for (int j = 0; j < N; j++)
//	cout << b[j] << " ";
//cout << endl;
