# Importamos las bibliotecas necesarias
import gymnasium as gym  # Para crear el entorno de CartPole-v1
import numpy as np       # Para trabajar con arrays y cálculos matemáticos
import matplotlib.pyplot as plt  # Para graficar los resultados

# Función para crear los bins (cajas) que discretizan el espacio de estados
def create_bins(low, high, bins):
    # Esta función crea una lista de bins, uno para cada dimensión del estado
    # Usamos np.linspace para dividir el rango de valores en partes iguales
    return [np.linspace(l, h, num + 1) for l, h, num in zip(low, high, bins)]

# Función para discretizar el estado continuo
def discretize_state(state, bins):
    # Convertimos el estado continuo en un índice discretizado
    return tuple(np.digitize(s, bins[i]) - 1 for i, s in enumerate(state))

# Función de entrenamiento principal
def train(episodes):
    # Creamos el entorno de CartPole
    env = gym.make("CartPole-v1", render_mode="human")
    
    # Definimos los límites del espacio de estados (posición y velocidad del carro, ángulo y velocidad del palo)
    low = [-4.8, -np.inf, -0.41887903, -np.inf]  # Límites inferiores
    high = [4.8, np.inf, 0.41887903, np.inf]    # Límites superiores
    
    # Definimos el número de bins por cada dimensión del estado
    num_bins = [8, 8, 8, 8]  # Usamos 8 bins para cada dimensión
    
    # Creamos los bins (cajas) para cada dimensión del espacio de estados
    bins = create_bins(low, high, num_bins)
    
    # Creamos la tabla Q, inicializada con ceros
    q_table = np.zeros((*[len(b) for b in bins], env.action_space.n))

    # Parámetros de Q-learning
    learning_rate = 0.1
    discount_factor = 0.95
    epsilon = 1.0
    epsilon_decay_rate = 0.002
    min_epsilon = 0.01
    
    # Arrays para almacenar recompensas y valores de epsilon
    rewards_per_episode = np.zeros(episodes)
    epsilon_values = np.zeros(episodes)
    
    # Entrenamiento
    for i in range(episodes):
        # Reseteamos el entorno
        state, _ = env.reset()
        state = discretize_state(state, bins)
        terminated, truncated = False, False
        total_reward = 0

        # Episodio
        while not terminated and not truncated:
            # Exploración o explotación
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            # Realizamos la acción
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_state = discretize_state(new_state, bins)

            # Actualización de la tabla Q
            q_table[state + (action,)] += learning_rate * (
                reward + discount_factor * np.max(q_table[new_state]) - q_table[state + (action,)]
            )

            state = new_state
            total_reward += reward

        # Decaimiento de epsilon
        epsilon = max(min_epsilon, epsilon - epsilon_decay_rate)
        rewards_per_episode[i] = total_reward
        epsilon_values[i] = epsilon

        # Imprimir progreso
        if (i + 1) % 50 == 0:
            print(f"Episodio: {i + 1} - Recompensa Total: {rewards_per_episode[i]} - Epsilon: {epsilon}")
    
    # Cerrar el entorno
    env.close()
    
    # Imprimir tabla Q final
    print("Tabla Q final:")
    print(q_table)
    
    # Graficar resultados
    plt.figure(figsize=(12, 6))

    # Recompensas por episodio
    plt.subplot(1, 2, 1)
    plt.plot(rewards_per_episode)
    plt.xlabel('Episodios')
    plt.ylabel('Recompensas')
    plt.title('Recompensa por Episodio')

    # Evolución de epsilon
    plt.subplot(1, 2, 2)
    plt.plot(epsilon_values)
    plt.xlabel('Episodios')
    plt.ylabel('Valor de epsilon')
    plt.title('Decay de Epsilon')

    plt.tight_layout()
    plt.show()

# Ejecutar entrenamiento
if __name__ == "__main__":
    train(500)
