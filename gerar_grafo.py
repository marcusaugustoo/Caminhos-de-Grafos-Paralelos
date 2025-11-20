import sys
import random

def generate_graph(n_size, density_perc, inf_val, output_file):  
    print(f"Gerando grafo {n_size}x{n_size} com densidade {density_perc}%")
    
    #Probabilidade de não ter uma aresta
    inf_prob = (100 - density_perc) / 100.0
    
    with open(output_file, 'w') as f:
        for i in range(n_size):
            line_values = []
            for j in range(n_size):
                if i == j:
                    line_values.append(0)  #Diagonal é sempre 0
                else:
                    if random.random() < inf_prob:
                        line_values.append(inf_val)
                    else:
                        line_values.append(random.randint(1, 10))

            f.write(' '.join(map(str, line_values)) + '\n')
            
    print(f"Arquivo '{output_file}' gerado com sucesso.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python3 gerar_grafo.py <N> <Densidade> <Arquivo_Saida>")
        print("Exemplo: python3 gerar_grafo.py 1024 75 grafo.txt")
        sys.exit(1)
        
    N = int(sys.argv[1])
    DENSIDADE = int(sys.argv[2])
    ARQUIVO = sys.argv[3]
    INF = 9999999
    
    generate_graph(N, DENSIDADE, INF, ARQUIVO)