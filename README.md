APSP com CUDA — Floyd–Warshall Paralelo

Implementação paralela do algoritmo Floyd–Warshall utilizando CUDA,
para cálculo de todos os pares de caminhos mínimos (APSP) em um grafo ponderado.

------------------------------------------------------------
DESCRIÇÃO
------------------------------------------------------------
O programa compara o desempenho entre:
- Versão sequencial (CPU)
- Versão paralela (GPU com CUDA)

e calcula o speedup obtido, verificando se os resultados são idênticos.

------------------------------------------------------------
REQUISITOS
------------------------------------------------------------
- GPU NVIDIA compatível com CUDA (Compute Capability >= 5.0)
- CUDA Toolkit (versão 11.0 ou superior)
- Compilador nvcc instalado e no PATH (verifique com: nvcc --version)
- Linux ou WSL2 (Windows Subsystem for Linux)

------------------------------------------------------------
COMPILAÇÃO
------------------------------------------------------------
No terminal, dentro da pasta do projeto:

    nvcc -O2 -arch=sm_86 -o apsp apsp.cu

(Substitua sm_86 conforme sua GPU, por exemplo, sm_75 para RTX 2070.)

------------------------------------------------------------
EXECUÇÃO
------------------------------------------------------------
O programa recebe o nome do arquivo do grafo como argumento:

    ./apsp grafo.txt

Exemplo de saída:
    
    Tamanho do Grafo (N): 1024
    Arquivo de Entrada: grafo.txt
    Executando CPU...
    Executando GPU...
    
    Resultados da CPU e GPU sao identicos.
    Tempo CPU: 4.432122 s
    Tempo GPU: 0.199180 s
    Speedup: 22.25x

------------------------------------------------------------
FORMATO DO ARQUIVO DE ENTRADA (grafo.txt)
------------------------------------------------------------
O arquivo deve conter a matriz de adjacência (N × N),
onde cada valor representa o peso da aresta entre dois vértices.
Use um valor alto (ex: 9999999) para representar INF (ausência de aresta).

Exemplo (N = 4):
    0 5 9999999 10
    9999999 0 3 9999999
    9999999 9999999 0 1
    9999999 9999999 9999999 0

------------------------------------------------------------
RESULTADOS
------------------------------------------------------------
O programa imprime:
- Tempo de execução da CPU
- Tempo de execução total da GPU
- Speedup entre CPU e GPU
- Confirmação de que os resultados são idênticos

------------------------------------------------------------
OBSERVAÇÕES
------------------------------------------------------------
- O tempo medido da GPU inclui todas as operações (alocação, cópias H↔D e execução).
- Os resultados podem variar conforme:
  - o tamanho da matriz (N)
  - a densidade de arestas no grafo
  - o modelo da GPU utilizada


