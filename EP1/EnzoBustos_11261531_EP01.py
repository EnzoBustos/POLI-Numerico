# Autor: Enzo Bustos Da Silva
# NUSP: 11261531
# EP1 - Calculo Numerico

####################################################################################################
####################################################################################################
####################################################################################################

# BIBLIOTECAS NECESSARIAS

import matplotlib.pyplot as plt  # Visualizacao de dados
import seaborn as sns            # Visualizacao de dados, so que mais bonita
import numpy as np               # Calculos e Aritmetica
import pandas as pd              # Auxiliar para a criacao de tabelas e graficos
import time                      # Tempo de execucao de funcoes
import cv2                       # Manipular imagens

####################################################################################################
####################################################################################################
####################################################################################################

# INTERFACE

def main():
    print("\n" + "="*100 + "\n")
    print("Voce esta no menu inicial !!!")
    print()
    print("Diga qual Tarefa voce quer realizar:")
    print()
    print("Para tarefa A) digite A")
    print("Para tarefa B) digite B")
    print("Para tarefa C) digite C")
    print("Para Finalizar digite 0")
    print()
    mode = input("Digite a tarefa desejada: ")
    print()

    if mode == 'A' or mode == 'a':
        do_mode_A()
    elif mode == 'B' or mode == 'b':
        do_mode_B()
    elif mode == 'C' or mode == 'c':
        do_mode_C()
    elif mode == '0':
        print("Finalizando")
        print("=============================")
        print()
    else:
        print("A opcao digitada nao e valida")
        print("=============================")
        print()
        main()

####################################################################################################
####################################################################################################
####################################################################################################

# TAREFAS DO MODO A

def do_mode_A():
    print("\n" + "="*100 + "\n")
    print("Voce esta no modo para resolver a tarefa A!!!")
    print()
    print("Aqui queremos Calcular os Autovalores e Autovetores para uma matriz A")
    print("Essa matriz tem 0's em todas as posicoes, exceto em sua diagonal e suas subdiagonais")
    print("Para nosso exercicio alpha = 2 (diagonal) e beta = -1 (subdiagonal)")
    print()
    print("Para as rotinas do exercicio calculamos para N = 4, 8, 16 e 32 com epsilon = 1e-6 (Erro)")
    print("Voce quer fazer as rotinas do exercicio (digite 1) ou personalizado (digite 2) ?")
    print("Digite outral coisa caso queira retornar")
    option = int(input("Digite sua opcao: "))
    print()

    if option == 1:
        print("Digite Sim ou Nao")
        save = input("Deseja salvar os Autovalores e Autovetores em um arquivo .csv?: ")
        print()
        
        for i in [4, 8, 16, 32]:
            Autovalores = {}
            Autovetores = {}

            A = make_A(i, 2, -1)
            eigenvalues_, eigenvectors_, iterations_ = QR_algorithm(A, threshold=1e-6, spectral_shifts=True)

            print("\n" + "="*50 + " N = " + str(n) + " " + "="*50 + "\n")
            print("Sem deslocamento espectral")
            print("Numero de iteracoes =", iterations)
            print("Com deslocamento espectral")
            print("Numero de iteracoes =", iterations_)
            print_values(eigenvalues_, symbol='lambda')
            print_values(eigenvectors_, symbol='ni')
        
        if save == "Sim" or save == "sim" or save == "S" or save == "s" or save == True:
            Autovalores['Nome'] = np.array(['lambda{}'.format(j + 1) for j in range(i)])
            Autovalores['lambda'] = eigenvalues_

            pd.DataFrame(Autovalores).to_csv('Autovalores {}.csv'.format(i), index=False)

            Autovetores['Nome'] = np.array(['ni {}'.format(j + 1) for j in range(i)])

            for k in range(i):
                Autovetores['Pos {}'.format(k)] = eigenvectors_[:, k]

            pd.DataFrame(Autovetores).to_csv('Autovetores {}.csv'.format(i), index=False)
        
        main()
    
    elif option == 2:
        N = int(input("Digite o tamanho da sua matriz: "))
        print()
        epsilon = float(input("Digite o valor de epsilon desejado: "))
        print()
        spectral_shifts = bool(input("Quer deslocamento espectral? (Digite True ou False): "))
        print()
        A = make_A(N, 2, -1)
        eigenvalues_, eigenvectors_, iterations_ = QR_algorithm(A, threshold=epsilon, spectral_shifts=True)

        print("Numero de iteracoes =", iterations_)
        print()
        print_values(eigenvalues_, symbol='lambda')
        print_values(eigenvectors_, symbol='ni')

        print("Voce gostaria de criar um grafico para visualizar a diferenca de Tempo e Iteracoes que o uso do Deslocamento gera?")
        graphs = bool(input("Digite True ou False: "))
        print()
        if graphs:
            print("ATENCAO!!! - Escolha com parcimonia seus valores, essa funcao demora")
            initial = int(input("Digite o valor inicial para seu estudo: "))
            print()
            final   = int(input("Digite o valor final para seu estudo: "))
            print()
            make_iteration_and_time_graphs(initial, final)
        
        main()

    else:
        main()

## FUNcoES AUXILIARES MODO A

# Recebe uma matriz quadrada A e um valor K e devolve os indices da subdiagonal correspondente
# Ex.: K = 0 corresponde aos indices da diagonal    que contem alfa
# Ex.: K =-1 corresponde aos indices da subdiagonal que contem gamma
# Ex.: K = 1 corresponde aos indices da subdiagonal que contem beta
# Pagina 3 do Enunciado
def K_diagonal_indices(A, K):
    rows, cols = np.diag_indices_from(A)
    if K < 0:
        return rows[-K:], cols[:K]
    elif K > 0:
        return rows[:-K], cols[K:]
    else:
        return rows, cols

# Cria uma matriz com 0's em todas as posicoes, exceto nas diagonais e subdiagonais que podem ter seus valores alterados pelos parametros
# make_A(4, 2, 1) --> Cria matriz A da pagina 3 do enunciado
# make_A(3, 4, 3) --> Cria matriz A da pagina 4 do enunciado
def make_A(N, diag_values=1, subdiag_values=0):
    A = np.zeros((N, N))
    np.fill_diagonal(A, diag_values)
    A[K_diagonal_indices(A, -1)] = subdiag_values
    A[K_diagonal_indices(A,  1)] = subdiag_values
    return A

# Tem como objetivo retornar o cosseno e seno de theta respectivos à K-esima iteracao do algoritmo QR
def get_cos_and_sen_k(A, k):
    cos = np.divide(A[k  , k], np.sqrt(np.square(A[k, k]) + np.square(A[k+1, k])))
    sin = np.divide(A[k+1, k], np.sqrt(np.square(A[k, k]) + np.square(A[k+1, k])))
    return cos, sin

# Retorna a matriz de rotacao de Givens G para uma determinada iteracao do algoritmo QR
def rotation_matrix(A, k):
    n = A.shape[0]
    i, j = k, k + 1

    G = np.zeros((n, n))

    c, s = get_cos_and_sen_k(A, k)

    np.fill_diagonal(G, 1)
    G[i, i] =  c
    G[j, j] =  c
    G[j, i] = -s
    G[i, j] =  s

    return G

# Funcao implementa a decomposicao QR de uma matriz A dada em uma matriz R triangular superior e uma matriz Q
# Mesma forma que o mostrado no exemplo no comeco da pagina 5
def QR_decomposition(A):
    R = A.copy()
    Q = []

    for k in range(A.shape[0] - 1):
        Q.append(rotation_matrix(R, k))
        R = Q[k] @ R
    
    final_Q = Q[-1]
    for i in range(len(Q) - 2, -1, -1):
        final_Q = final_Q @ Q[i]

    Q = final_Q

    return final_Q, R

# Funcao sinal
def sgn(number):
    if number >= 0:
        return 1
    return -1

# Funcao para calculo da Heuristica de Wilkinson
def Wilkinson(A):
    n = A.shape[0]
    d_k = (A[n-2, n-2] - A[n-1, n-1])/2
    return A[n-1, n-1] + d_k - sgn(d_k) * np.sqrt(np.square(d_k) + np.square(A[n-1, n-2]))

# Funcao implementa o algoritmo QR do mesmo modo que esta indicado no pseudo-codigo da pagina 7 do enunciado
def QR_algorithm(A, threshold=1e-6, spectral_shifts=True):
    A0 = A.copy()
    V0 = np.identity(A.shape[0])
    mi_k = 0
    contador = 0
    eigenvalues = []
    eigenvectors = []

    if spectral_shifts:
        for k in range(A.shape[0], 1, -1):
            while abs(A0[-1, -2]) >= threshold and A0.shape[0] > 2:
                Q, R = QR_decomposition(A0 - mi_k * np.identity(A0.shape[0]))
                A0 = R @ Q.T + mi_k * np.identity(A0.shape[0])
                V0 = V0 @ Q.T
                mi_k = Wilkinson(A0)
                contador += 1
            
            eigenvalues.append(A0[-1, -1])
            eigenvectors.append(V0[ :, -1])

            A0 = A0[:k-1, :k-1]
            V0 = V0[:   , :k-1]            
    
    else:
        for k in range(A.shape[0], 1, -1):
            while abs(A0[-1, -2]) > threshold:
            
                Q, R = QR_decomposition(A0)
                A0 = R @ Q.T
                V0 = V0 @ Q.T
                contador += 1

            eigenvalues.append(A0[-1, -1])
            eigenvectors.append(V0[ :, -1])

            A0 = A0[:k-1, :k-1]
            V0 = V0[:   , :k-1]
        
    
    eigenvalues.append(A0[-1, -1])
    eigenvectors.append(V0[ :, -1])
    
    A0[abs(A0) < threshold] = 0
    V0[abs(V0) < threshold] = 0
    
    return np.asarray(eigenvalues)[::-1], np.asarray(eigenvectors)[::-1], contador

# Funcao para ajudar a imprimir os autovalores e autovetores
def print_values(values, symbol='theta'):
    if symbol == "lambda":
        for i in range(len(values)):
            if i > 9:
                print("{}{}:".format(symbol, i + 1), values[i])
            else:
                print("{}{} :".format(symbol, i + 1), values[i])
    
    if symbol == "ni":
        for i in range(len(values)):
            if i > 9:
                print("{}{}:\n".format(symbol, i + 1), values[i])
            else:
                print("{}{} :\n".format(symbol, i + 1), values[i])

    print()

# Funcao que cria um DataFrame com o tempo e iteracoes com e sem o deslocamento espectral
def get_iterations_and_time(initial=4, final=76):
    k_with_shift = []
    k_without_shift = []
    t_with_shift = []
    t_without_shift = []

    for n in np.arange(initial, final):
        A = make_A(n, 2, -1)

        t = time.process_time()
        eigenvalues , eigenvectors , iterations  = QR_algorithm(A, threshold=1e-6, spectral_shifts=False)
        deltat = time.process_time() - t

        t = time.process_time()
        eigenvalues_, eigenvectors_, iterations_ = QR_algorithm(A, threshold=1e-6, spectral_shifts=True)
        deltat_ = time.process_time() - t
    
        k_with_shift.append(iterations_)
        k_without_shift.append(iterations)
        t_with_shift.append(deltat_)
        t_without_shift.append(deltat)

    df = pd.DataFrame({
        'Matrix Dimension'         : np.array(np.arange(initial, final)),
        'Iterations_Shifted'       : k_with_shift,
        'Iterations_Not_Shifted'   : k_without_shift,
        'deltaTime_Shifted'        : t_with_shift,
        'deltaTime_Not_Shifted'    : t_without_shift,
    })

    return df

# Usa o DataFrame criado na funcao get_iterations_and_time para criar um grafico 
def make_iteration_and_time_graphs(initial=4, final=76):

    df = get_iterations_and_time(initial, final)

    sns.set(rc={'figure.figsize':(8.26772, 11.69291/2)})

    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    sns.lineplot(ax=ax1, x='Matrix Dimension', y='Iterations_Not_Shifted', data=df, color='red')
    sns.lineplot(ax=ax2, x='Matrix Dimension', y='Iterations_Shifted', data=df, color='blue')
    sns.lineplot(ax=ax3, x='Matrix Dimension', y='deltaTime_Not_Shifted', data=df, color='red')
    sns.lineplot(ax=ax4, x='Matrix Dimension', y='deltaTime_Shifted', data=df, color='blue')

    ax1.set_title("Iteracoes x Dimensao", fontsize=12, fontweight='bold')
    ax2.set_title("Iteracoes x Dimensao", fontsize=12, fontweight='bold')
    ax3.set_title("Tempo x Dimensao", fontsize=12, fontweight='bold')
    ax4.set_title("Tempo x Dimensao", fontsize=12, fontweight='bold')

    ax1.set_xlabel('Dimensao da Matriz', fontsize=12, fontstyle='italic')
    ax2.set_xlabel('Dimensao da Matriz', fontsize=12, fontstyle='italic')
    ax3.set_xlabel('Dimensao da Matriz', fontsize=12, fontstyle='italic')
    ax4.set_xlabel('Dimensao da Matriz', fontsize=12, fontstyle='italic')
    ax1.set_ylabel('Numero de Iteracoes', fontsize=12, fontstyle='italic')
    ax2.set_ylabel('Numero de Iteracoes', fontsize=12, fontstyle='italic')
    ax3.set_ylabel('Variacao de Tempo (s)', fontsize=12, fontstyle='italic')
    ax4.set_ylabel('Variacao de Tempo (s)', fontsize=12, fontstyle='italic')

    plt.suptitle("Relacao de Tempo e Iteracoes pelo uso de Deslocamento Espectral", fontweight='bold', fontsize=18, y=1.03)

    plt.tight_layout()

    fig = plt.gcf()
    fig.savefig("Tempo vs Iteracoes.png", format='png', dpi=200)

    plt.show()


####################################################################################################
####################################################################################################
####################################################################################################

# TAREFAS DO MODO B

def do_mode_B():
    print("\n" + "="*100 + "\n")
    print("Voce esta no modo para resolver a tarefa B!!!")
    print()
    print("Aqui temos um sistema com 5 massas de 2kg e com molas com ki = (40 + 2i)N/m com i = 1, 2, 3, 4, 5, 6")
    print("Desejamos criar graficos sobre a evolucao da solucao, sabendo que a velocidade inicial e nula")
    print()
    print("1) X(0) = [-2, -3, -1, -3, -1]")
    print("2) X(0) = [ 1, 10, -4,  3, -2]")
    print("3) X(0) = [Modo de maior frequencia]")
    option = int(input("Digite a opcao desejada: "))
    print()
    t = int(input("Digite o tempo em segundos (recomendado 4): "))
    print()

    # Monta a matriz do modo B
    A = np.zeros((5, 5))
    k = [40 + 2*i for i in np.arange(1, 7)]
    A[K_diagonal_indices(A,  0)] = np.array([k[i] + k[i+1] for i in range(len(k) - 1)])
    A[K_diagonal_indices(A, -1)] = np.array([-k[i] for i in range(1, len(k) - 1)])
    A[K_diagonal_indices(A,  1)] = np.array([-k[i] for i in range(1, len(k) - 1)])
    A = 1/2 * A

    Autovalores, Autovetores, _ = QR_algorithm(A, threshold=1e-20)

    omega = []
    print("Frequencias de vibracao omega = sqrt(lambda):")
    for i in range(len(Autovalores)):
        omega.append(np.sqrt(Autovalores[i]))
        print("omega{}:".format(i + 1), omega[i])
    omega = np.asarray(omega)
    
    print()

    ni = []
    print("Modos naturais de vibracao ni:")
    for i in range(len(Autovetores)):
        ni.append(Autovetores[i]/Autovetores[i, -1])
        print("ni{}:".format(i + 1), ni[i])
    ni = np.asarray(ni)
    
    if option == 1:
        X0 = np.array([-2, -3, -1, -3, -1])
        df = make_graphs(X0, A, t, 
            title="Deslocamento de cada uma das Massas\npelo seu Ponto de Equilibrio\n" + r"$X_{0}$" + " = [-2, -3, -1, -3, -1]",
            dim=(8.26772, 11.6929/2))
        multiple_graphs_5(X0, df)
        save_imgs(str(X0)+'.png', "splt " + str(X0)+'.png', 'Grafico 1.1.png')

    elif option == 2:
        X0 = np.array([1, 10, -4, 3, -2])
        df = make_graphs(X0, A, t,
                 title="Deslocamento de cada uma das Massas\npelo seu Ponto de Equilibrio\n" + r"$X_{0}$" + " = [1, 10, -4, 3, -2]",
                 dim=(8.26772, 11.6929/2))
        multiple_graphs_5(X0, df)
        save_imgs(str(X0)+'.png', "splt " + str(X0)+'.png', 'Grafico 1.2.png')

    elif option == 3:
        X0 = ni[0]
        df = make_graphs(X0, A, t,
                 title="Deslocamento de cada uma das Massas\npelo seu Ponto de Equilibrio\n" + r"$X_{0}$" + " = Modo de maior frequencia",
                 dim=(8.26772, 11.6929/2))
        multiple_graphs_5(X0, df)
        save_imgs(str(X0)+'.png', "splt " + str(X0)+'.png', 'Grafico 1.3.png')
    
    else:
        print("Opcao invalida, retornando ao Menu Inicial")
        main()
    
    main()
    
## FUNCOES AUXILIARES MODO B

# Funcao usada para atualizar os valores de X para cada instante de tempo
def X_update(X0, eingenvalues, eingenvectors, t):

    C = np.cos(np.sqrt(eingenvalues) * t)
    A = eingenvectors.T @ X0

    Y = [A[i] * C[i] for i in range(len(X0))]

    return eingenvectors @ Y

# Funcao que cria o grafico superior
def make_graphs(X0, A, t, xlabel='Tempo [s]', ylabel='Deslocamento [m]',
                title="Deslocamento de cada uma das Massas pelo seu Ponto de Equilibrio\n" + r"$X_{0}$",
                dim=(11.69291*2, 8.26772)):
    eingenvalues, eingenvectors, _ = QR_algorithm(A)
    eingenvectors = eingenvectors.T

    time_ = np.array(np.arange(0, t + 0.001, 0.001))
    X = []

    for t in time_: 
        X.append(X_update(X0, eingenvalues, eingenvectors, t))
    
    X = np.asarray(X)
    X = X.T

    D = {"Time" : time_,}

    for i in range(len(X0)):
        D['X{}'.format(i+1)] = X[i]
    
    data = pd.DataFrame(D)
    colors = sns.color_palette("bright", len(X0))
    
    sns.set(rc={'figure.figsize':dim})

    for i in range(len(X0)):
        sns.lineplot(x='Time', y='X{}'.format(i+1), data=data,
                     label=r'$X_{%d}$' % (i+1) + "(t)",color=colors[i])

    plt.xlabel(xlabel, fontstyle='italic', fontsize=12)
    plt.ylabel(ylabel, fontstyle='italic', fontsize=12)
    plt.title(title, fontweight='bold', fontsize=18, y=1.075)
    plt.legend(loc='upper right')

    plt.tight_layout()

    fig = plt.gcf()
    fig.savefig(str(X0) + ".png", format='png', dpi=200)
    plt.show()

    return data

# Funcao que cria um graficos inferiores
def multiple_graphs_5(X0, df):
    sns.set(rc={'figure.figsize':(8.26772, 11.6929/2)})
    colors = sns.color_palette("bright", len(X0))

    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(2, 3, 3, sharex=ax1, sharey=ax1)
    ax4 = plt.subplot(2, 3, 4, sharex=ax1, sharey=ax1)
    ax5 = plt.subplot(2, 3, 5, sharex=ax1, sharey=ax1)

    sns.lineplot(ax=ax1, x='Time', y='X1', data=df, color=colors[1-1])
    sns.lineplot(ax=ax2, x='Time', y='X2', data=df, color=colors[2-1])
    sns.lineplot(ax=ax3, x='Time', y='X3', data=df, color=colors[3-1])
    sns.lineplot(ax=ax4, x='Time', y='X4', data=df, color=colors[4-1])
    sns.lineplot(ax=ax5, x='Time', y='X5', data=df, color=colors[5-1])

    ax1.set_title("Deslocamento da Massa " + r"$X_{1}$", fontsize=12, fontweight='bold')
    ax2.set_title("Deslocamento da Massa " + r"$X_{2}$", fontsize=12, fontweight='bold')
    ax3.set_title("Deslocamento da Massa " + r"$X_{3}$", fontsize=12, fontweight='bold')
    ax4.set_title("Deslocamento da Massa " + r"$X_{4}$", fontsize=12, fontweight='bold')
    ax5.set_title("Deslocamento da Massa " + r"$X_{5}$", fontsize=12, fontweight='bold')

    ax1.set_xlabel('Tempo [s]', fontsize=12, fontstyle='italic')
    ax2.set_xlabel('Tempo [s]', fontsize=12, fontstyle='italic')
    ax3.set_xlabel('Tempo [s]', fontsize=12, fontstyle='italic')
    ax4.set_xlabel('Tempo [s]', fontsize=12, fontstyle='italic')
    ax5.set_xlabel('Tempo [s]', fontsize=12, fontstyle='italic')
    ax1.set_ylabel('Deslocamento [m]', fontsize=12, fontstyle='italic')
    ax2.set_ylabel('Deslocamento [m]', fontsize=12, fontstyle='italic')
    ax3.set_ylabel('Deslocamento [m]', fontsize=12, fontstyle='italic')
    ax4.set_ylabel('Deslocamento [m]', fontsize=12, fontstyle='italic')
    ax5.set_ylabel('Deslocamento [m]', fontsize=12, fontstyle='italic')

    plt.tight_layout()

    fig = plt.gcf()
    fig.savefig("splt " + str(X0) + ".png", format='png', dpi=200)

    plt.show()

# Funcao que concatena e salva as imagens
def save_imgs(path1, path2, file_name='Grafico.png'):
    im1 = cv2.imread(path1)
    im2 = cv2.imread(path2)

    im_v = cv2.vconcat([im1, im2])
    cv2.imwrite(file_name, im_v)

####################################################################################################
####################################################################################################
####################################################################################################

# TAREFAS DO MODO C

def do_mode_C():
    print("\n" + "="*100 + "\n")
    print("Voce esta no modo para resolver a tarefa C!!!")
    print()
    print("Aqui temos um sistema com 10 massas de 2kg e com molas com ki = (40 + 2*(-1)^i)N/m com i = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11")
    print("Desejamos criar graficos sobre a evolucao da solucao, sabendo que a velocidade inicial e nula")
    print()
    print("1) X(0) = [-2, -3, -1, -3, -1, -2, -3, -1, -3, -1]")
    print("2) X(0) = [ 1, 10, -4,  3, -2,  1, 10, -4,  3, -2]")
    print("3) X(0) = [Modo de maior frequencia]")
    option = int(input("Digite a opcao desejada: "))
    print()
    t = int(input("Digite o tempo em segundos (recomendado 4): "))
    print()

    # Monta a matriz do modo B
    A = np.zeros((10, 10))
    k = [40 + 2*(-1)**i for i in np.arange(1, 12)]
    A[K_diagonal_indices(A,  0)] = np.array([k[i] + k[i+1] for i in range(len(k) - 1)])
    A[K_diagonal_indices(A, -1)] = np.array([-k[i] for i in range(1, len(k) - 1)])
    A[K_diagonal_indices(A,  1)] = np.array([-k[i] for i in range(1, len(k) - 1)])
    A = 1/2 * A

    Autovalores, Autovetores, _ = QR_algorithm(A)

    omega = []
    print("Frequencias de vibracao omega = sqrt(lambda):")
    for i in range(len(Autovalores)):
        omega.append(np.sqrt(Autovalores[i]))
        print("omega{}:".format(i + 1), omega[i])
    omega = np.asarray(omega)
    print()
    ni = []
    print("Modos naturais de vibracao ni:")
    for i in range(len(Autovetores)):
        ni.append(Autovetores[i]/Autovetores[i, -1])
        print("ni{}:".format(i + 1), ni[i])
    ni = np.asarray(ni)

    if option == 1:
        X0 = np.array([-2, -3, -1, -3, -1, -2, -3, -1, -3, -1])
        df = make_graphs(X0, A, t, 
        title="Deslocamento de cada uma das Massas\npelo seu Ponto de Equilibrio\n" + r"$X_{0}$" + " = [-2, -3, -1, -3, -1, -2, -3, -1, -3, -1")
        multiple_graphs_10(X0, df)
        save_imgs(str(X0)+'.png', "splt " + str(X0)+'.png', 'Grafico 2.1.png')
    
    elif option == 2:
        X0 = np.array([1, 10, -4, 3, -2, 1, 10, -4, 3, -2])
        df = make_graphs(X0, A, t,
        title="Deslocamento de cada uma das Massas\npelo seu Ponto de Equilibrio\n" + r"$X_{0}$" + " = [1, 10, -4, 3, -2, 1, 10, -4, 3, -2]")
        multiple_graphs_10(X0, df)
        save_imgs(str(X0)+'.png', "splt " + str(X0)+'.png', 'Grafico 2.2.png')
    
    elif option == 3:
        X0 = np.array(ni[0])
        df = make_graphs(X0, A, t,
        title="Deslocamento de cada uma das Massas\npelo seu Ponto de Equilibrio\n" + r"$X_{0}$" + " = Modo de maior frequencia")
        save_imgs(str(X0)+'.png', "splt " + str(X0)+'.png', 'Grafico 2.3.png')
    
    else:
        print("Opcao invalida, retornando ao Menu Inicial")
        main()
    
    main()

## FUNcoES AUXILIARES MODO C

def multiple_graphs_10(X0, df):
    sns.set(rc={'figure.figsize':(11.69291*2, 8.26772)})
    colors = sns.color_palette("bright", len(X0))

    ax1  = plt.subplot(2, 5, 1 )
    ax2  = plt.subplot(2, 5, 2 , sharex=ax1, sharey=ax1)
    ax3  = plt.subplot(2, 5, 3 , sharex=ax1, sharey=ax1)
    ax4  = plt.subplot(2, 5, 4 , sharex=ax1, sharey=ax1)
    ax5  = plt.subplot(2, 5, 5 , sharex=ax1, sharey=ax1)
    ax6  = plt.subplot(2, 5, 6 , sharex=ax1, sharey=ax1)
    ax7  = plt.subplot(2, 5, 7 , sharex=ax1, sharey=ax1)
    ax8  = plt.subplot(2, 5, 8 , sharex=ax1, sharey=ax1)
    ax9  = plt.subplot(2, 5, 9 , sharex=ax1, sharey=ax1)
    ax10 = plt.subplot(2, 5, 10, sharex=ax1, sharey=ax1)

    sns.lineplot(ax=ax1, x='Time', y='X1', data=df, color=colors[1 -1])
    sns.lineplot(ax=ax2, x='Time', y='X2', data=df, color=colors[2 -1])
    sns.lineplot(ax=ax3, x='Time', y='X3', data=df, color=colors[3 -1])
    sns.lineplot(ax=ax4, x='Time', y='X4', data=df, color=colors[4 -1])
    sns.lineplot(ax=ax5, x='Time', y='X5', data=df, color=colors[5 -1])
    sns.lineplot(ax=ax6, x='Time', y='X6', data=df, color=colors[6 -1])
    sns.lineplot(ax=ax7, x='Time', y='X7', data=df, color=colors[7 -1])
    sns.lineplot(ax=ax8, x='Time', y='X8', data=df, color=colors[8 -1])
    sns.lineplot(ax=ax9, x='Time', y='X9', data=df, color=colors[9 -1])
    sns.lineplot(ax=ax10, x='Time', y='X10', data=df, color=colors[10-1])

    ax1.set_title("Deslocamento da Massa " + r"$X_{1}$", fontsize=12, fontweight='bold')
    ax2.set_title("Deslocamento da Massa " + r"$X_{2}$", fontsize=12, fontweight='bold')
    ax3.set_title("Deslocamento da Massa " + r"$X_{3}$", fontsize=12, fontweight='bold')
    ax4.set_title("Deslocamento da Massa " + r"$X_{4}$", fontsize=12, fontweight='bold')
    ax5.set_title("Deslocamento da Massa " + r"$X_{5}$", fontsize=12, fontweight='bold')
    ax6.set_title("Deslocamento da Massa " + r"$X_{6}$", fontsize=12, fontweight='bold')
    ax7.set_title("Deslocamento da Massa " + r"$X_{7}$", fontsize=12, fontweight='bold')
    ax8.set_title("Deslocamento da Massa " + r"$X_{8}$", fontsize=12, fontweight='bold')
    ax9.set_title("Deslocamento da Massa " + r"$X_{9}$", fontsize=12, fontweight='bold')
    ax10.set_title("Deslocamento da Massa " + r"$X_{10}$", fontsize=12, fontweight='bold')

    ax1.set_xlabel('Tempo [s]', fontsize=12, fontstyle='italic')
    ax2.set_xlabel('Tempo [s]', fontsize=12, fontstyle='italic')
    ax3.set_xlabel('Tempo [s]', fontsize=12, fontstyle='italic')
    ax4.set_xlabel('Tempo [s]', fontsize=12, fontstyle='italic')
    ax5.set_xlabel('Tempo [s]', fontsize=12, fontstyle='italic')
    ax6.set_xlabel('Tempo [s]', fontsize=12, fontstyle='italic')
    ax7.set_xlabel('Tempo [s]', fontsize=12, fontstyle='italic')
    ax8.set_xlabel('Tempo [s]', fontsize=12, fontstyle='italic')
    ax9.set_xlabel('Tempo [s]', fontsize=12, fontstyle='italic')
    ax10.set_xlabel('Tempo [s]', fontsize=12, fontstyle='italic')

    ax1.set_ylabel('Deslocamento [m]', fontsize=12, fontstyle='italic')
    ax2.set_ylabel('Deslocamento [m]', fontsize=12, fontstyle='italic')
    ax3.set_ylabel('Deslocamento [m]', fontsize=12, fontstyle='italic')
    ax4.set_ylabel('Deslocamento [m]', fontsize=12, fontstyle='italic')
    ax5.set_ylabel('Deslocamento [m]', fontsize=12, fontstyle='italic')
    ax6.set_ylabel('Deslocamento [m]', fontsize=12, fontstyle='italic')
    ax7.set_ylabel('Deslocamento [m]', fontsize=12, fontstyle='italic')
    ax8.set_ylabel('Deslocamento [m]', fontsize=12, fontstyle='italic')
    ax9.set_ylabel('Deslocamento [m]', fontsize=12, fontstyle='italic')
    ax10.set_ylabel('Deslocamento [m]', fontsize=12, fontstyle='italic')

    plt.tight_layout()

    fig = plt.gcf()
    fig.savefig("splt " + str(X0) + ".png", format='png', dpi=200)

    plt.show()

####################################################################################################
####################################################################################################
####################################################################################################

main()