#include <iostream>
#include <windows.h>
#include <stdlib.h>

using namespace std;

const int N = 10240;// matrix size

double b[N][N], a[N];
int c[N];

void init(int n)
{
    for(int i = 0;i < N;i++)
    {
        for(int j = 0;j < N;j++)
            b[i][j] = i + j;
        a[i] = (double)i;
        c[i] = i;
    }
}

void Lab1_1_uncache(int n,int round)
{
    double col_sum [N];
    for(int i = 0; i < n; i++)
        col_sum[i] = 0.0;
    long long head, tail, freq; //timers
     // similar to CLOCKS_PRE_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for(int r = 0;r < round;r++)
    {
        for(int i = 0;i < n;i++)
        {
            col_sum[i] =0.0;
            for(int j = 0;j < n;j++)
                col_sum[i] += b[j][i] * a[i];
        }
    }
    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    //cout <<"Round:"<< n <<" Col:"<< (tail - head) * 1000.0 / freq / (double)round << "ms"<< endl;
    cout << n << ',' << (tail - head) * 1000.0 / freq / (double)round << endl;

}

void Lab1_1_cache(int n,int round)
{
    double col_sum [N];
    for(int i = 0; i < n; i++)
        col_sum[i] = 0.0;
    long long head, tail, freq; //timers
     // similar to CLOCKS_PRE_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for(int k = 0;k < round;k++)
	{
		for(int j = 0; j < n; j++)
		{
			for(int i = 0; i < n; i++)
				col_sum[i] += b[j][i] * a[j];
		}
	}
    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    //cout <<"Round:"<< n <<" Col:"<< (tail - head) * 1000.0 / freq / (double)round << "ms"<< endl;
    cout << n << ',' << (tail - head) * 1000.0 / freq / (double)round << endl;

}

void Lab1_2_un(int n,int round)
{
    int sum = 0;
    long long head, tail, freq; //timers
     // similar to CLOCKS_PRE_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for(int j = 0;j < round;j++)
    {
        for(int i = 1;i <= n;i++)
        {
            sum += a[i];
        }
    }

    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

    cout << n << ',' << (tail - head) * 1000.0 / freq / (double)round << endl;

}

void Lab1_2_way1(int n,int round)
{
    int sum1 = 0.0;
    int sum2 = 0.0;
    long long head, tail, freq; //timers
     // similar to CLOCKS_PRE_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for(int j = 0;j < round;j++)
    {
        for(int i = 1;i <= n;i += 2)
        {
            sum1 += c[i];
            sum2 += c[i + 1];
        }
    }
    int sum = sum1 + sum2;

    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

    cout << n << ',' << (tail - head) * 1000.0 / freq / (double)round << endl;

}

void Lab1_2_way2(int n,int round)
{
    long long head, tail, freq; //timers
     // similar to CLOCKS_PRE_SEC
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    // start time
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for(int j = 0;j < round;j++)
    {
        for (int m = n; m > 1; m /= 2)
        {
            for (int i = 0; i < m / 2; i++) a[i] = a[i * 2] + a[i * 2 + 1]; // 相邻元素相加连续存储到数组最前面
        }
    }
    // end time
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

    cout << n << ',' << (tail - head) * 1000.0 / freq / (double)round << endl;

}

void chain()
{
     for(int j = 0;j < 5000;j++)
    {
        int sum = 0;
        for(int i = 1;i <= 4096;i += 1)
        {
            sum += a[i];
        }
    }
}

void chain_2()
{

     for(int j = 0;j < 5000;j++)
    {
        int sum1 = 0;
        int sum2 = 0;
        for(int i = 1;i <= 4096;i += 2)
        {
            sum1 += a[i];
            sum2 += a[i + 1];
        }
        int sum = sum1 + sum2;
    }
}

void chain_3()
{
     for(int j = 0;j < 5000;j++)
    {
         for (int m = 4096; m > 1; m /= 2)
        {
            for (int i = 0; i < m / 2; i++) a[i] = a[i * 2] + a[i * 2 + 1]; // 相邻元素相加连续存储到数组最前面
        }
    }
}

int main()
{
    init(N);
    for(int i = 0;i < 8991;i++){
        Lab1_2_way1(i,1000);
    }

    return 0;
}
