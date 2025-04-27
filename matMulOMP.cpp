#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>

void matrixMultiply(float* A, float* B, float* C, int N) {
#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			C[i*N + j] = 0;
			for (int k = 0; k < N; k++) {
				C[i*N + j] += A[i*N + k] * B[k*N + j];
			}
		}
	}
}

void runBenchmark(int min_size, int max_size, int step, int num_threads) {
	// Настройка вывода в фиксированном формате с 7 знаками после запятой
	std::cout << std::fixed << std::setprecision(7);

	std::cout << "Matrix Size\tTime (seconds)\n";
	std::cout << "---------------------------\n";

	for (int N = min_size; N <= max_size; N += step) {
		// Выделение памяти
		float* A = new float[N*N];
		float* B = new float[N*N];
		float* C = new float[N*N];

		// Инициализация матриц
		for (int i = 0; i < N; i++) {
			for (int j = 0; j < N; j++) {
				A[i*N + j] = (i + 1) * (j + 1);
				B[i*N + j] = (i + 1) + 2 * (j + 1);
			}
		}

		// Установка числа потоков
		omp_set_num_threads(num_threads);

		// Прогрев (чтобы избежать влияния "холодного старта")
		matrixMultiply(A, B, C, N);

		// Замер времени
		double start = omp_get_wtime();
		matrixMultiply(A, B, C, N);
		double elapsed = omp_get_wtime() - start;

		// Вывод результатов в удобном формате
		std::cout << N << "\t\t" << elapsed << "\n";

		// Освобождение памяти
		delete[] A;
		delete[] B;
		delete[] C;
	}
}

int main() {
	int min_size = 100;
	int max_size = 2000;
	int step = 100;
	int threads = 8; // Можно изменить для тестирования разного числа потоков

	std::cout << "Running benchmark with " << threads << " threads...\n";
	runBenchmark(min_size, max_size, step, threads);

	return 0;
}