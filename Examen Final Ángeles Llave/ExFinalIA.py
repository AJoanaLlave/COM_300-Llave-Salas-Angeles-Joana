import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

# Configuración del entorno GridWorld
def create_gridworld(grid_size, obstacles):
    grid = np.zeros(grid_size)
    for obstacle in obstacles:
        grid[obstacle] = -1  # Obstáculos representados como -1
    return grid

def plot_grid(grid, agent_position, visited, title="GridWorld"):
    grid_display = grid.copy()
    for cell in visited:
        grid_display[cell] = 0.5  # Representar celdas visitadas con 0.5
    grid_display[agent_position] = 2  # Representar al agente con un 2

    # Crear un mapa de colores personalizado para diferentes elementos
    cmap = sns.color_palette(["gray", "black", "red", "pink", "purple"], as_cmap=True)

    # Actualizar valores en grid_display para ajustar el rango del cmap
    # -1 -> Obstáculos (negro), 0 -> Celdas libres (blanco), 0.5 -> Celdas visitadas (lightblue)
    # 2 -> Agente (rojo), otros valores si es necesario
    grid_display[grid_display == -1] = 1  # Obstáculos
    grid_display[grid_display == 0] = 0  # Espacio libre
    grid_display[grid_display == 0.5] = 3  # Visitadas
    grid_display[grid_display == 2] = 4  # Agente

    plt.figure(figsize=(6.5, 6.5))
    sns.heatmap(grid_display, annot=False, cbar=False, cmap=cmap, linewidths=0.5, linecolor="black")
    plt.title(title)
    plt.show(block=False)
    plt.pause(0.2)
    plt.close()

def initialize_rewards(grid_size, goal_position):
    rewards = np.full(grid_size, -0.1)  # Recompensa predeterminada por cada paso
    rewards[goal_position] = 1  # Recompensa positiva al alcanzar el objetivo
    return rewards

def valid_moves(grid, position):
    moves = []
    directions = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1)
    }
    for direction, (dr, dc) in directions.items():
        new_row, new_col = position[0] + dr, position[1] + dc
        if 0 <= new_row < grid.shape[0] and 0 <= new_col < grid.shape[1] and grid[new_row, new_col] != -1:
            moves.append((direction, (new_row, new_col)))
    return moves

def choose_action(state, valid_moves, q_table, epsilon, visited_cells):
    if random.uniform(0, 1) < epsilon:
        unvisited_moves = [move for move in valid_moves if move[1] not in visited_cells]
        if unvisited_moves:
            return random.choice(unvisited_moves)
        return random.choice(valid_moves)
    else:
        best_move = max(valid_moves, key=lambda move: q_table.get((state, move[0]), 0))
        return best_move

def update_q_table(q_table, state, action, reward, next_state, alpha, gamma, grid):
    valid_next_moves = valid_moves(grid, next_state)
    current_q = q_table.get((state, action), 0)
    max_next_q = max([q_table.get((next_state, move[0]), 0) for move in valid_next_moves], default=0)
    q_table[(state, action)] = current_q + alpha * (reward + gamma * max_next_q - current_q)

def calculate_reward(position, visited_cells, goal_position, steps_taken):
    if position in visited_cells:
        return -1.0  # Penalización más fuerte por volver a una celda visitada
    elif position == goal_position:
        return 10.0 - (steps_taken * 0.1)  # Recompensa significativa ajustada por el tiempo
    else:
        return -0.2  # Penalización ligera por cada paso innecesario
    
# Parámetros del entorno
grid_size = (5, 5)
grid_obstacles = [(1, 1), (1, 2), (3, 4), (3, 3)]
goal_position = (4, 4)

grid = create_gridworld(grid_size, grid_obstacles)
rewards = initialize_rewards(grid_size, goal_position)

# Identificar todas las celdas libres
free_cells = {(r, c) for r in range(grid_size[0]) for c in range(grid_size[1]) if grid[r, c] != -1}

# Hiperparámetros de entrenamiento
alpha = 0.5  # Tasa de aprendizaje
gamma = 0.5  # Factor de descuento
epsilon = 0.5  # Probabilidad de exploración
min_epsilon = 0.01
epsilon_decay = 0.9
num_episodes = 100  # Número de episodios

# Inicialización de la tabla Q
q_table = {}

# Entrenamiento
rewards_per_episode = []
epsilon_values = []

for episode in range(num_episodes):
    agent_position = (0, 0)  # Reiniciar la posición del agente
    visited_cells = set()
    visited_cells.add(agent_position)
    steps_taken = 0
    total_reward = 0

    while True:
        state = agent_position
        moves = valid_moves(grid, agent_position)
        if not moves:
            break  # Si no hay movimientos válidos, termina

        chosen_move = choose_action(state, moves, q_table, epsilon, visited_cells)
        action, next_position = chosen_move

        # Obtener recompensa basada en tiempo
        reward = calculate_reward(next_position, visited_cells, goal_position, steps_taken)
        total_reward += reward

        # Actualizar Q-table
        update_q_table(q_table, state, action, reward, next_position, alpha, gamma, grid)

        agent_position = next_position
        visited_cells.add(agent_position)
        steps_taken += 1

        plot_grid(grid, agent_position, visited_cells, title=f"Episodio {episode+1}, Moviendo al agente ({action})")

        # Verificar si el agente ha visitado todas las celdas libres y alcanzado el objetivo
        if visited_cells == free_cells and agent_position == goal_position:
            print(f"¡Episodio {episode+1}: Todas las celdas libres visitadas y objetivo alcanzado en {steps_taken} pasos!")
            break

    rewards_per_episode.append(total_reward)
    epsilon_values.append(epsilon)

    # Reducir epsilon para disminuir la exploración
    epsilon = max(min_epsilon, epsilon * 0.95)

# Gráficas de resultados
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(rewards_per_episode, label="Recompensa por episodio")
plt.xlabel("Episodios")
plt.ylabel("Recompensa total")
plt.title("Recompensas durante el entrenamiento")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epsilon_values, label="Epsilon", color="orange")
plt.xlabel("Episodios")
plt.ylabel("Epsilon")
plt.title("Evolución de epsilon")
plt.legend()

plt.tight_layout()
plt.show()

print("Entrenamiento completado.")
