#include<mpi.h>
#include<stdio.h>
#include<math.h>
#include<cstring>
#include <iostream>
#include <windows.h>
#include<algorithm>
#pragma comment(lib,"mpi.lib")
using namespace std;
const int N = 512, numProcess = 4;
float A[N][N];

void init()
{
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			A[i][j] = float(rand()) / 10;
}

int main(int argc, char *argv[])
{
	int myid;
	MPI_Status status;
	MPI_Init(0, 0);
	double start, end;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	int r1 = myid * (N / numProcess), r2 = (myid == numProcess - 1) ? N - 1 : (myid + 1)*(N / numProcess) - 1;
	if (myid == 0)
	{
		init();
		for (int i = 1; i < numProcess; i++)
		{
			int r11 = i * (N / numProcess), r22 = (i == numProcess - 1) ? N - 1 : (i + 1)*(N / numProcess) - 1;
			MPI_Send(&A[r11][0], (r22 - r11 + 1)* N, MPI_FLOAT, i, N + 1, MPI_COMM_WORLD);
		}
	}
	else
	{
		MPI_Recv(&A[r1][0], (r2 - r1 + 1)*N, MPI_FLOAT, 0, N + 1, MPI_COMM_WORLD, &status);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
	for (int k = 0; k < N; k++)
	{
		if (myid == 0)
		{
			for (int j = k + 1; j < N; j++)
				A[k][j] /= A[k][k];
			A[k][k] = 1.0;
			for (int j = 1; j < numProcess; j++)
				MPI_Send(&A[k][0], N, MPI_FLOAT, j, k + 1, MPI_COMM_WORLD);
		}
		else
			MPI_Recv(&A[k][0], N, MPI_FLOAT, 0, k + 1, MPI_COMM_WORLD, &status);
		if (r2 >= k + 1)
		{
			for (int i = max(r1, k + 1); i <= r2; i++)
			{
				for (int j = k + 1; j < N; j++)
				{
					A[i][j] -= A[k][j] * A[i][k];
				}
				A[i][k] = 0.0;
				if (i == k + 1 && myid != 0)
					MPI_Send(&A[i][0], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
		if (myid == 0 && k + 1 > r2&&k + 1 < N)
			MPI_Recv(&A[k + 1][0], N, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	end = MPI_Wtime();
	cout << "MPItime " << (end - start) * 1000 << " ms" << endl;
	MPI_Finalize();
	return 0;
}