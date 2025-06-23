# Tetris Q-Learning Agent 🎮

Este proyecto implementa un agente que aprende a jugar **Tetris** mediante el algoritmo de **Q-Learning**, utilizando el entorno personalizado `tetris_gymnasium` compatible con `gymnasium`.

---

## Descripción del Proyecto

Este proyecto es una prueba experimental de aprendizaje por refuerzo **sin usar directamente el estado del entorno**. El objetivo es explorar cómo un agente puede aprender a seleccionar acciones basándose únicamente en las recompensas, usando una variante de Q-Learning estilo "multi-armed bandit".

El entorno utilizado es una versión de Tetris basada en `gymnasium`, con renderizado visual (`render_mode="human"`), donde el agente elige acciones al azar o por valor aprendido.

---

## Instalación

### 1. Clonar el repositorio (si lo subes a GitHub)
```bash
git clone https://github.com/tu_usuario/tetris-ql.git
cd tetris-ql
