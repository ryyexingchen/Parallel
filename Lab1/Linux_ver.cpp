#include<iostream>
#include<sys/time.h>
#include<stdlib.h>
using namespace std;
const int N = 10240;// matrix size

double b[N][N], a[N];

void init()
{
    for(int i = 0;i < N;i++)
    {
        for(int j = 0;j < N;j++)
            b[i][j] = i + j;
        a[i] = i;
    }
}

int main(){
	struct timeval start;
	struct timeval end;
	float timecount;
	init();
	double col_sum [N];
    for(int i = 0; i < 8192; i++)
        col_sum[i] = 0.0;
	gettimeofday(&start,NULL);
	for(int k = 0;k < 10;k++)
	{
		for(int j = 0; j < 8192; j++)
		{
			for(int i = 0; i < 8192; i++)
				col_sum[i] += b[j][i] * a[j];
		}
	}
	gettimeofday(&end,NULL);
	timecount+=(end.tv_sec-start.tv_sec)*1000000+end.tv_usec-start.tv_usec;
	cout<<timecount/20<<endl;
	return 0;
}