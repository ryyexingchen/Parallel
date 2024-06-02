#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<map>
#include<windows.h>
#include<pmmintrin.h>//SSE3
#include<immintrin.h>//AVX
using namespace std;

const int maxsize = 3000;
const int maxrow = 3000;
const int numBasis = 90000;

map<int, int*>iToBasis;
map<int, int>iToFirst;
map<int, int*>ans;

fstream RowFile("被消元行.txt", ios::in | ios::out);
fstream BasisFile("消元子.txt", ios::in | ios::out);

int gRows[maxrow][maxsize];
int gBasis[numBasis][maxsize];

void reset() {
	memset(gRows, 0, sizeof(gRows));
	memset(gBasis, 0, sizeof(gBasis));
	RowFile.close();
	BasisFile.close();
	RowFile.open("被消元行.txt", ios::in | ios::out);
	BasisFile.open("消元子.txt", ios::in | ios::out);
	iToBasis.clear();
	iToFirst.clear();
	ans.clear();

}

void readBasis() {
	for (int i = 0; i < maxrow; i++) {
		if (BasisFile.eof()) {
			return;
		}
		string tmp;
		bool flag = false;
		int row = 0;
		getline(BasisFile, tmp);
		stringstream s(tmp);
		int pos;
		while (s >> pos) {
			if (!flag) {
				row = pos;
				flag = true;
				iToBasis.insert(pair<int,int*>(row, gBasis[row]));
			}
			int index = pos / 32;
			int offset = pos % 32;
			gBasis[row][index] = gBasis[row][index]| (1 << offset);
		}
		flag = false;
		row = 0;
	}
}

int readRowsFrom(int pos) {
	iToFirst.clear();
	if (RowFile.is_open())
		RowFile.close();
	RowFile.open("被消元行.txt", ios::in | ios::out);
	memset(gRows, 0, sizeof(gRows));
	string line;
	for (int i = 0; i < pos; i++) {
		getline(RowFile, line);
	}
	for (int i = pos; i < pos + maxsize; i++) {
		int tmp;
		getline(RowFile, line);
		if (line.empty()) {
			cout << "End of File!" << endl;
			return i;
		}
		bool flag = false;
		stringstream s(line);
		while (s >> tmp) {
			if (!flag) {
				iToFirst.insert(pair<int, int>(i - pos, tmp));
			}
			int index = tmp / 32;
			int offset = tmp % 32;
			gRows[i-pos][index] = gRows[i-pos][index] | (1 << offset);
			flag = true;
		}
	}
	return -1;
}

void update(int row) {
	bool flag = 0;
	for (int i = maxsize - 1; i >= 0; i--) {
		if (gRows[row][i] == 0)
			continue;
		else {
			if (!flag)
				flag = true;
			int pos = i * 32;
			int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (gRows[row][i] & (1 << k))
				{
					offset = k;
					break;
				}
			}
			int newfirst = pos + offset;
			iToFirst.erase(row);
			iToFirst.insert(pair<int, int>(row, newfirst));
			break;
		}
	}
	if (!flag) {
		iToFirst.erase(row);
	}
	return;
}

void writeResult(ofstream& out) {
	for (auto it = ans.rbegin(); it != ans.rend(); it++) {
		int* result = it->second;
		int max = it->first/32 + 1;
		for (int i = max; i >= 0; i--) {
			if (result[i] == 0)
				continue;
			int pos = i * 32;
			for (int k = 31; k >= 0; k--) {
				if (result[i] & (1 << k)) {
					out << k + pos << " ";
				}
			}
		}
		out << endl;
	}
}

void GE() {
	int begin = 0;
	int flag;
	while (true) {
		flag = readRowsFrom(begin);
		int num = (flag == -1) ? maxsize : flag;
		for (int i = 0; i < num; i++) {
			while (iToFirst.find(i) != iToFirst.end()) {
				int first = iToFirst.find(i)->second;
				if (iToBasis.find(first) != iToBasis.end()) {
					int* basis = iToBasis.find(first)->second;
					for (int j = 0; j < maxsize; j++) {
						gRows[i][j] = gRows[i][j] ^ basis[j];
					}
					update(i);
				}
		 		else {
					for (int j = 0; j < maxsize; j++) {
						gBasis[first][j] = gRows[i][j];
					}
					iToBasis.insert(pair<int,int*>(first, gBasis[first]));
					ans.insert(pair<int, int*>(first, gBasis[first]));
					iToFirst.erase(i);
				}
			}
		}
		if (flag == -1)
			begin += maxsize;
		else
			break;
	}
}

void GE_SSE() {  
    int begin = 0;  
    int flag;  
    while (true) {  
        flag = readRowsFrom(begin);  
        int num = (flag == -1) ? maxsize : flag;  
        for (int i = 0; i < num; i++) {  
            if (iToFirst.find(i) != iToFirst.end()) {  
                int first = iToFirst.find(i)->second;  
                if (iToBasis.find(first) != iToBasis.end()) {  
                    int* basis = iToBasis.find(first)->second;  
                    int j = 0;  
                    for (; j + sizeof(__m128i) < maxsize; j += sizeof(__m128i)) {  
                        __m128i gRowVector = _mm_load_si128((__m128i const*)&gRows[i][j]);  
                        __m128i basisVector = _mm_load_si128((__m128i const*)&basis[j]);  
                        __m128i resultVector = _mm_xor_si128(gRowVector, basisVector);  
                        _mm_store_si128((__m128i*)&gRows[i][j], resultVector);  
                    }  
					for (; j < maxsize; j++) {
						gRows[i][j] = gRows[i][j] ^ basis[j];
					}
                    update(i);  
                } else {  
                   int j=0;
					for ( ; j + sizeof(__m128i) < maxsize; j += sizeof(__m128i)) {
						__m128i vij = _mm_loadu_si128((__m128i*)&gRows[i][j]);
						_mm_storeu_si128((__m128i*) & gRows[first][j],vij);
					}
					for ( ; j < maxsize; j++) {
						gBasis[first][j] = gRows[i][j];
					}
                    iToBasis.insert(make_pair(first, gBasis[first]));  
                    ans.insert(make_pair(first, gBasis[first]));  
                    iToFirst.erase(i);  
                }  
            }  
        }  
        if (flag == -1) {  
            begin += maxsize;  
        } else {  
            break;  
        }  
    }  
}

void GE_AVX() {  
    int begin = 0;  
    int flag;  
    while (true) {  
        flag = readRowsFrom(begin);  
        int num = (flag == -1) ? maxsize : flag;  
        for (int i = 0; i < num; i++) {  
            if (iToFirst.find(i) != iToFirst.end()) {  
                int first = iToFirst.find(i)->second;  
                if (iToBasis.find(first) != iToBasis.end()) {  
                    int* basis = iToBasis.find(first)->second;  
                    int j = 0;  
                    for (; j + sizeof(__m256i) < maxsize; j += sizeof(__m256i)) {  
                        __m256i gRowVector = _mm256_load_si256((__m256i const*)&gRows[i][j]);  
                        __m256i basisVector = _mm256_load_si256((__m256i const*)&basis[j]);  
                        __m256i resultVector = _mm256_xor_si256(gRowVector, basisVector);  
                        _mm256_store_si256((__m256i*)&gRows[i][j], resultVector);  
                    }  
					for (; j < maxsize; j++) {
						gRows[i][j] = gRows[i][j] ^ basis[j];
					}
                    update(i);  
                } else {  
                   int j=0;
					for ( ; j + sizeof(__m256i) < maxsize; j += sizeof(__m256i)) {
						__m256i vij = _mm256_loadu_si256((__m256i*)&gRows[i][j]);
						_mm256_storeu_si256((__m256i*) & gRows[first][j],vij);
					}
					for ( ; j < maxsize; j++) {
						gBasis[first][j] = gRows[i][j];
					}
                    iToBasis.insert(std::make_pair(first, gBasis[first]));  
                    ans.insert(std::make_pair(first, gBasis[first]));  
                    iToFirst.erase(i);  
                }  
            }  
        }  
        if (flag == -1) {  
            begin += maxsize;  
        } else {  
            break;  
        }  
    }  
}

int main() {
	ofstream out("result.txt");
	for (int i = 0; i < 11; i++) {
		cout<<"i:"<<i<<endl;
        out<<"i:"<<i<<endl;
		long long head1, tail1, freq1;
	    readBasis();
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq1);
		QueryPerformanceCounter((LARGE_INTEGER*)&head1);
		GE();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail1);
		out << "Ordinary_GE:" << (tail1 - head1) * 1000.0 / freq1 << "ms" << endl;
		cout << "Ordinary_GE:" << (tail1 - head1) * 1000.0 / freq1 << "ms" << endl;
		
		reset();
		readBasis();
		long long head2, tail2, freq2;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq2);
		QueryPerformanceCounter((LARGE_INTEGER*)&head2);
		GE_SSE();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail2);
		out << "SSE_GE:" << (tail2 - head2) * 1000.0 / freq2 << "ms" << endl;
		cout << "SSE_GE:" << (tail2 - head2) * 1000.0 / freq2 << "ms" << endl;

		reset();
		readBasis();
		long long head3, tail3, freq3;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq3);
		QueryPerformanceCounter((LARGE_INTEGER*)&head3);
		GE_AVX();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail3);
		out << "AVX_GE:" << (tail3 - head3) * 1000.0 / freq3 << "ms" << endl;
		cout << "AVX_GE:" << (tail3 - head3) * 1000.0 / freq3 << "ms" << endl;
		reset();
	}
	out.close();
}

