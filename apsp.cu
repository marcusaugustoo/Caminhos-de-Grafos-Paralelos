#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include <cuda_runtime.h>

#define INF 9999999
#define BLOCK_SIZE 16

//Macro para verificação de erros CUDA
static void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(result));
        exit(1);
    }
}

//Kernel
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

//Versão sequencial (CPU) 
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

//Carrega grafo do arquivo de texto plano
void load_graph_from_file(const char* filename, int* dist, int n) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Erro ao abrir arquivo '%s': %s\n", filename, strerror(errno));
        exit(1);
    }
    //Tenta ler exatamente n*n inteiros
    for (int i = 0; i < n * n; i++) {
        if (fscanf(f, "%d", &dist[i]) != 1) {
            fprintf(stderr, "Erro: Arquivo acabou antes do esperado ou formato invalido para N=%d.\n", n);
            fclose(f);
            exit(1);
        }
    }
    fclose(f);
}

//Verifica se o resultado da GPU bate com a CPU
bool verify_results(int* cpu_res, int* gpu_res, int n) {
    for (int i = 0; i < n * n; i++) {
        if (cpu_res[i] != gpu_res[i]) {
            printf("ERRO: Divergencia na posicao [%d][%d] (Indice %d). CPU=%d, GPU=%d\n", 
                   i/n, i%n, i, cpu_res[i], gpu_res[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    //Validação de argumentos para evitar erros de execução
    if (argc < 3) {
        printf("Uso: %s <N> <arquivo_grafo>\n", argv[0]);
        printf("Exemplo: %s 1024 grafo.txt\n", argv[0]);
        return 1;
    }

    int n_arg = atoi(argv[1]);
    const char* graph_filename = argv[2];

    printf("Tamanho do Grafo (N): %d\n", n_arg);
    printf("Arquivo de Entrada: %s\n", graph_filename);

    size_t matrix_size = (size_t)n_arg * n_arg * sizeof(int);

    int* h_dist_cpu = (int*)malloc(matrix_size);
    int* h_dist_in  = (int*)malloc(matrix_size);
    int* h_dist_out = (int*)malloc(matrix_size);

    if (!h_dist_cpu || !h_dist_in || !h_dist_out) {
        fprintf(stderr, "Falha ao alocar memoria no host.\n");
        return 1;
    }

    //Carga dos dados
    load_graph_from_file(graph_filename, h_dist_in, n_arg);
    
    //Copia entrada para buffer de validação da CPU
    memcpy(h_dist_cpu, h_dist_in, matrix_size);

    //Execução CPU 
    printf("Executando CPU...\n"); fflush(stdout);
    clock_t cpu_start = clock();
    floyd_cpu(h_dist_cpu, n_arg);
    clock_t cpu_end = clock();
    double cpu_time_s = ((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC);


    //Execução GPU
    printf("Executando GPU...\n"); fflush(stdout);
    
    int *d_in = NULL, *d_out = NULL;
    clock_t total_start = clock(); // Medindo tempo total (Alocação + Cópia + Kernel)

    checkCuda(cudaMalloc((void**)&d_in, matrix_size));
    checkCuda(cudaMalloc((void**)&d_out, matrix_size));
    checkCuda(cudaMemcpy(d_in, h_dist_in, matrix_size, cudaMemcpyHostToDevice));

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n_arg + BLOCK_SIZE - 1) / BLOCK_SIZE, (n_arg + BLOCK_SIZE - 1) / BLOCK_SIZE);

    //Loop principal do Floyd-Warshall
    for (int k = 0; k < n_arg; k++) {
        floyd_kernel<<<gridSize, blockSize>>>(d_in, d_out, k, n_arg);
        
        //Troca de ponteiros 
        int* tmp = d_in; 
        d_in = d_out; 
        d_out = tmp;
    }
    
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaMemcpy(h_dist_out, d_in, matrix_size, cudaMemcpyDeviceToHost));

    clock_t total_end = clock();
    double gpu_total_time_s = ((double)(total_end - total_start) / CLOCKS_PER_SEC);

    //Verificação e Métricas
    if (verify_results(h_dist_cpu, h_dist_out, n_arg)) {
        printf("\nResultados da CPU e GPU sao identicos.\n");
        printf("Tempo CPU: %.6f s\n", cpu_time_s);
        printf("Tempo GPU: %.6f s\n", gpu_total_time_s);
        printf("Speedup  : %.2fx\n", cpu_time_s / gpu_total_time_s);
    } else {
        printf("\n[FALHA] Os resultados nao conferem.\n");
    }

    //Limpeza
    checkCuda(cudaFree(d_in)); 
    checkCuda(cudaFree(d_out));
    free(h_dist_cpu);
    free(h_dist_in);
    free(h_dist_out);

    return 0;
}
