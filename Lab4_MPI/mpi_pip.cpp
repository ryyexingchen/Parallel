#include<mpi.h>
#include<stdio.h>
#include<math.h>
#include<cstring>
#include <iostream>
#include<immintrin.h>
#include <windows.h>
#include<algorithm>
#pragma comment(lib,"mpi.lib")
using namespace std;
const int N = 2048, numProcess = 4;
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
	MPI_Init(NULL, NULL);
	double start, end;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	if (myid == 0)
	{
		init();
		for (int i = 1; i < numProcess; i++)
			for (int j = i; j < N; j += numProcess)
				MPI_Send(&A[j][0], N, MPI_FLOAT, i, j, MPI_COMM_WORLD);
	}
	else
	{
		for (int j = myid; j < N; j += numProcess)
			MPI_Recv(&A[j][0], N, MPI_FLOAT, 0, j, MPI_COMM_WORLD, &status);
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
	for (int k = 0; k < N; ++k)
	{
		if (myid == 0)
		{
			
			for (int j = k + 1; j < N; j++)
				A[k][j] /= A[k][k];
			A[k][k] = 1.0;
			MPI_Send(&A[k][0], N, MPI_FLOAT, myid+1, 0, MPI_COMM_WORLD);
		}
		else if (myid == numProcess - 1)
		{
			MPI_Recv(&A[k][0], N, MPI_FLOAT, myid - 1, 0, MPI_COMM_WORLD, &status);
		}
		else
		{
			MPI_Recv(&A[k][0], N, MPI_FLOAT, myid - 1, 0, MPI_COMM_WORLD, &status);
			MPI_Send(&A[k][0], N, MPI_FLOAT, myid + 1, 0, MPI_COMM_WORLD);
		}
		int r2 = myid;
		while (r2 < k + 1)r2 += numProcess;
		for (int i = r2; i < N; i += numProcess)
		{
			for (int j = k + 1; j < N; j++)
				A[i][j] -= A[k][j] * A[i][k];
			A[i][k] = 0.0;
		}
		if ((k + 1) % numProcess == myid && myid != 0)
			MPI_Send(&A[k + 1][0], N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		if (myid == 0 && (k + 1) % numProcess != 0 && k + 1 < N)
			MPI_Recv(&A[k + 1][0], N, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
	}
	end = MPI_Wtime();
	if (myid == 0) {
		cout << "MPItime " << (end - start) * 1000 << " ms" << endl;
	}
	MPI_Finalize();
	return 0;
}