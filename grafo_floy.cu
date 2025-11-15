#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include <cuda_runtime.h>

#define INF 9999999
#define N 1024        
#define BLOCK_SIZE 16

static void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(result));
        exit(1);
    }
}

//Kernel: lê de d_in, escreve em d_out(iteracao fixa k)
__global__ void floyd_kernel(const int* d_in, int* d_out, int k, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n) {
        int ij_idx = i * n + j;
        int ik_idx = i * n + k;
        int kj_idx = k * n + j;

        int dij = d_in[ij_idx];
        int dik = d_in[ik_idx];
        int dkj = d_in[kj_idx];

        if (dik == INF || dkj == INF) {
            d_out[ij_idx] = dij;
        } else {
            int new_dist = dik + dkj;
            d_out[ij_idx] = (dij > new_dist) ? new_dist : dij;
        }
    }
}

//Versão sequencial do Floyd-Warshall.
void floyd_cpu(int* dist, int n) {
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int ij_idx = i * n + j;
                int ik_idx = i * n + k;
                int kj_idx = k * n + j;

                int dik = dist[ik_idx];
                int dkj = dist[kj_idx];

                if (dik == INF || dkj == INF) continue;

                int new_dist = dik + dkj;
                if (dist[ij_idx] > new_dist) dist[ij_idx] = new_dist;
            }
        }
    }
}


void load_graph_from_file(const char* filename, int* dist, int n) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Erro ao abrir %s: %s\n", filename, strerror(errno));
        exit(1);
    }
    for (int i = 0; i < n * n; i++) {
        if (fscanf(f, "%d", &dist[i]) != 1) {
            fprintf(stderr, "Erro ao ler grafo na posicao %d.\n", i);
            fclose(f);
            exit(1);
        }
    }
    fclose(f);
}

void save_result_to_file(const char* filename, int* dist, int n) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Erro ao criar %s\n", filename);
        return;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (dist[i * n + j] == INF) fprintf(f, "INF ");
            else fprintf(f, "%d ", dist[i * n + j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

bool verify_results(int* cpu_res, int* gpu_res, int n) {
    for (int i = 0; i < n * n; i++) {
        if (cpu_res[i] != gpu_res[i]) {
            printf("Erro na verificacao! Posicao %d: CPU=%d, GPU=%d\n",
                   i, cpu_res[i], gpu_res[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Uso: %s grafo.txt\n", argv[0]);
        return 1;
    }

    const char* graph_filename = argv[1];

    size_t matrix_size = (size_t)N * N * sizeof(int);
    int* h_dist_cpu = (int*)malloc(matrix_size);
    int* h_dist_in  = (int*)malloc(matrix_size);
    int* h_dist_out = (int*)malloc(matrix_size);

    if (!h_dist_cpu || !h_dist_in || !h_dist_out) {
        fprintf(stderr, "Falha ao alocar memoria no host\n");
        return -1;
    }

    //Carrega grafo e copia para CPU
    load_graph_from_file(graph_filename, h_dist_in, N);
    memcpy(h_dist_cpu, h_dist_in, matrix_size);


    // Execução na CPU
    clock_t cpu_start = clock();
    floyd_cpu(h_dist_cpu, N);
    clock_t cpu_end = clock();
    double cpu_time_s = ((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC);


    // Execução na GPU
    int *d_in = NULL, *d_out = NULL;

    //Medição total (inclui alocação + cópia H2D + kernels + cópia D2H)
    double gpu_total_time_s = 0.0;

    //Início da medição total
    clock_t total_start = clock();

    checkCuda(cudaMalloc((void**)&d_in, matrix_size));
    checkCuda(cudaMalloc((void**)&d_out, matrix_size));

    //Cópia host --> device
    checkCuda(cudaMemcpy(d_in, h_dist_in, matrix_size, cudaMemcpyHostToDevice));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    //Loop principal do algoritmo (GPU)
    for (int k = 0; k < N; k++) {
        floyd_kernel<<<gridSize, blockSize>>>(d_in, d_out, k, N);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
        int* tmp = d_in; d_in = d_out; d_out = tmp;
    }

    //Cópia device --> host
    checkCuda(cudaMemcpy(h_dist_out, d_in, matrix_size, cudaMemcpyDeviceToHost));

    //Fim da medição total
    clock_t total_end = clock();
    gpu_total_time_s = ((double)(total_end - total_start) / CLOCKS_PER_SEC);

    //Verificação de resultados
    bool ok = verify_results(h_dist_cpu, h_dist_out, N);
    if (!ok) {
        fprintf(stderr, "Falha na verificacao entre CPU e GPU. Abortando.\n");
    } else {
        printf("Sucesso! CPU e GPU produzem o mesmo resultado.\n");
    }


    //Cálculo e impressão do speedup
    double speedup_total = cpu_time_s / gpu_total_time_s;

    printf("Tempo CPU: %.6f s\n", cpu_time_s);
    printf("Tempo GPU: %.6f s\n", gpu_total_time_s);
    printf("Speedup (CPU / GPU total): %.2fx\n", speedup_total);

    //Limpeza
    checkCuda(cudaFree(d_in));
    checkCuda(cudaFree(d_out));
    free(h_dist_cpu);
    free(h_dist_in);
    free(h_dist_out);

    return 0;
}
