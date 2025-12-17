#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Matrix = std::vector<std::vector<double>>;

static Matrix make_matrix(std::size_t n) {
    return Matrix(n, std::vector<double>(n));
}

static void fill_random(Matrix& m, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    const std::size_t n = m.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            m[i][j] = dist(rng);
        }
    }
}

static void matmul_baseline(const Matrix& A, const Matrix& B, Matrix& C) {
    const std::size_t n = A.size();

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

static double checksum(const Matrix& C) {
    double s = 0.0;
    const std::size_t n = C.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            s += C[i][j];
        }
    }
    return s;
}

static double run_once(std::size_t n) {
    Matrix A = make_matrix(n);
    Matrix B = make_matrix(n);
    Matrix C = make_matrix(n);

    fill_random(A, 42);
    fill_random(B, 1337);

    const auto t0 = std::chrono::steady_clock::now();
    matmul_baseline(A, B, C);
    const auto t1 = std::chrono::steady_clock::now();

    volatile double sink = checksum(C);
    (void)sink;

    std::chrono::duration<double> dt = t1 - t0;
    return dt.count();
}

int main(int argc, char** argv) {
    std::size_t n = 768;
    int repeats = 7;

    if (argc >= 2) n = static_cast<std::size_t>(std::stoul(argv[1]));
    if (argc >= 3) repeats = std::stoi(argv[2]);
    if (repeats < 1) repeats = 1;

    (void)run_once(n);

    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(repeats));

    for (int r = 0; r < repeats; ++r) {
        times.push_back(run_once(n));
    }

    std::sort(times.begin(), times.end());
    const double median = times[times.size() / 2];
    const double best = times.front();
    const double worst = times.back();

    std::cout << "N=" << n
              << " repeats=" << repeats
              << " median=" << std::fixed << std::setprecision(6) << median << " s"
              << " best=" << best << " s"
              << " worst=" << worst << " s\n";

    return 0;
}
