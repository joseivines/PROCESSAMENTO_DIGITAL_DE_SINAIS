#ALUNO: JOSÉ IVINES MATOS SILVA
#DICSCIPLINA: PROCESSAMENTO DIGITAL DE SINAIS (UFCG)
#OBJETIVO DO CÓDIGO:IMPLEMENTAÇÃO DA TRANSFORMADA RÁPIDA DE FOURIER (FFT) - FFT por Decimação no tempo por metodo iterativo

def fft_iterativo(x):
    N = len(x)

    if not ((N > 0) and (N & (N - 1) == 0)):
        raise ValueError("O tamanho do sinal de entrada deve ser uma potência de 2.")

    #Reordenação por Bit-Reverso Direta (substitui a matriz P)
    #Copia os dados para um novo array para realizar os cálculos
    X = np.asarray(x, dtype=complex)

    #Calcula os índices de bit reverso
    num_bits = int(np.log2(N))
    revertendo_indices = np.zeros(N, dtype=int)
    for i in range(N):
        indice = format(i, f'0{num_bits}b')
        revertendo_indices[i] = int(indice[::-1], 2)

    #Reordenação da matriz de entrada
    X = X[revertendo_indices]

    #Loop através dos estágios (m = 1, 2, ..., log2(N))
    for m in range(1, num_bits + 1):
        # L é o tamanho da DFT que está sendo construída neste estágio
        L = 2**m
        L_half = L // 2

        # Calcula os "twiddle factors" para este estágio
        twiddle_factors = np.exp(-2j * np.pi * np.arange(L_half) / L)

        # Loop através dos blocos de borboleta dentro do estágio
        for k in range(0, N, L):
            # Loop através das operações de borboleta dentro de cada bloco
            for j in range(L_half):
                # Índices dos dois elementos da borboleta
                idx1 = k + j
                idx2 = k + j + L_half

                # Aplica a operação de borboleta
                t = twiddle_factors[j] * X[idx2]
                X[idx2] = X[idx1] - t
                X[idx1] = X[idx1] + t

    return X

import numpy as np
import matplotlib.pyplot as plt

def convolucao_circular_fft(x, h):
    N = max(len(x), len(h))

    #Find the next power of 2 greater than or equal to N
    N_fft = 2**int(np.ceil(np.log2(N)))

    #preenchendo os sinais de entrada com zeros para ambos ter o mesmo tamanho
    sinal_x = np.pad(x, (0, N_fft - len(x)), 'constant')
    sinal_h = np.pad(h, (0, N_fft - len(h)), 'constant')

    #calculo da FFT de cada sinal
    X = fft_iterativo(sinal_x)
    H = fft_iterativo(sinal_h)

    #multiplicação no domínio da frequência
    Y = X * H

    #ajuste do fator de escala
    y = np.conj(fft_iterativo(np.conj(Y))) / N_fft

    return np.real(y)[:N]

#testando a implementação do algoritmo da convolução circular
sinal1 = np.array([1, 2, 1, 0])
sinal2 = np.array([1, 0, 1, 8,9])

resultado_convolucao = convolucao_circular_fft(sinal1, sinal2)

#plotando os sinais e o resultado da convolução
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.stem(sinal1)
plt.title('Sinal 1')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.stem(sinal2)
plt.title('Sinal 2')
plt.xlabel('t')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.stem(resultado_convolucao)
plt.title('Resultado da Convolução Circular')
plt.xlabel('w')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()
