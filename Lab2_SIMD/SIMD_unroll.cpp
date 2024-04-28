#include <pmmintrin.h> //SSE3
#include <iostream>
#include <ctime>
#include <windows.h>
#include <stdlib.h>
#include <fstream>
using namespace std;

const int loop = 10;
const int N = 4096;
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

float* Gaussian_Elimination_parallel() {
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

float* Gaussian_Elimination_parallel_unroll() {
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
           for (j = k + 1; j + 8 <= N; j += 8) {  
				// 首先加载所有需要的数据  
				__m128 vakj1 = _mm_loadu_ps(A[k] + j);  
				__m128 vakj2 = _mm_loadu_ps(A[k] + j + 4);  
				__m128 vaij1 = _mm_loadu_ps(A[i] + j);  
				__m128 vaij2 = _mm_loadu_ps(A[i] + j + 4);  
				// 执行乘法运算  
				__m128 vx1 = _mm_mul_ps(vaik, vakj1);  
				__m128 vx2 = _mm_mul_ps(vaik, vakj2);  
				// 执行减法运算，并存储结果  
				vaij1 = _mm_sub_ps(vaij1, vx1);  
				_mm_storeu_ps(A[i] + j, vaij1);  
				vaij2 = _mm_sub_ps(vaij2, vx2);  
				_mm_storeu_ps(A[i] + j + 4, vaij2);  
			}
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

float* Gaussian_Elimination_parallel_unroll2() {
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
        for (int i = k + 1; i + 1 < N; i+=2) {//消元
            __m128 vaik1 = _mm_set1_ps(A[i][k]);
			__m128 vaik2 = _mm_set1_ps(A[i+1][k]);
            for (j = k + 1; j + 4 <= N; j += 4) {
				// 首先加载所有需要的数据 
                __m128 vakj1 = _mm_loadu_ps(A[k]+j);
                __m128 vaij1 = _mm_loadu_ps(A[i]+j);
				__m128 vakj2 = _mm_loadu_ps(A[k]+j);
                __m128 vaij2 = _mm_loadu_ps(A[i+1]+j);
                // 执行乘法运算
                __m128 vx1 = _mm_mul_ps(vaik1, vakj1);
				__m128 vx2 = _mm_mul_ps(vaik2, vakj2);
				// 执行减法运算，并存储结果  
                vaij1 = _mm_sub_ps(vaij1, vx1);
                _mm_storeu_ps(A[i]+j, vaij1);
                vaij2 = _mm_sub_ps(vaij2, vx2);
                _mm_storeu_ps(A[i+1]+j, vaij2);
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

int main() {
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
		float* x2 = Gaussian_Elimination_parallel();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
		//cout << "x=[";
		//for (int i = 0; i < N; i++) {
		//	cout << x2[i];
		//	if (i != N - 1)  cout << ",";
		//}
		//cout << "]" << endl;
		cout << "Parallel(" << N << "):" << (tail2 - head2) * 1000.0 / freq2 << "ms" << endl;

		m_reset();
		long long head3, tail3, freq3;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq3);
		QueryPerformanceCounter((LARGE_INTEGER*)&head3);
		float* x3 = Gaussian_Elimination_parallel();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail3);
		//cout << "x=[";
		//for (int i = 0; i < N; i++) {
		//	cout << x3[i];
		//	if (i != N - 1)  cout << ",";
		//}
		//cout << "]" << endl;
		cout << "Parallel_unroll(" << N << "):" << (tail3 - head3) * 1000.0 / freq3 << "ms" << endl;
		
		m_reset();
		long long head4, tail4, freq4;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq4);
		QueryPerformanceCounter((LARGE_INTEGER*)&head4);
		float* x4 = Gaussian_Elimination_parallel_unroll2();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail4);
		//cout << "x=[";
		//for (int i = 0; i < N; i++) {
		//	cout << x4[i];
		//	if (i != N - 1)  cout << ",";
		//}
		//cout << "]" << endl;
		cout << "Parallel_unroll2(" << N << "):" << (tail4 - head4) * 1000.0 / freq4 << "ms" << endl;
		cout << "-------------------------------------------------------" <<endl;
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
