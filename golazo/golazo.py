import cv2
import pygame
import mediapipe as mp
import numpy as np
import os

# Inicialización de Pygame
pygame.init()

# Configuración de la pantalla
WIDTH, HEIGHT = 1000, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("¡Golazo! Juego de Fútbol") # Título del juego

# Directorio base para encontrar los recursos (imágenes, etc.)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 150, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Fuentes
font = pygame.font.Font(None, 74)
small_font = pygame.font.Font(None, 50)

# --- Constantes de los objetos del juego ---
goal_width, goal_height = 300, 150
goalkeeper_width, goalkeeper_height = 80, 100

# --- Carga de imágenes (asegúrate de tener estas imágenes en tu carpeta) ---
# Si no tienes estas imágenes, puedes crearlas o usar formas geométricas simples
try:
    field_image = pygame.image.load(os.path.join(BASE_DIR, "campo.PNG")) # Imagen de fondo del campo
    field_image = pygame.transform.scale(field_image, (WIDTH, HEIGHT))
except pygame.error:
    print("Advertencia: 'campo.PNG' no encontrado. Usando fondo verde.")
    field_image = None

try:
    ball_image = pygame.image.load(os.path.join(BASE_DIR, "balon.png")) # Imagen de la pelota
    ball_image = pygame.transform.scale(ball_image, (50, 50))
except pygame.error:
    print("Advertencia: 'balon.png' no encontrado. Usando círculo rojo.")
    ball_image = None

try:
    goal_image = pygame.image.load(os.path.join(BASE_DIR, "porteria.png")) # Imagen de la portería
    goal_image = pygame.transform.scale(goal_image, (goal_width, goal_height))
except pygame.error:
    print("Advertencia: 'porteria.png' no encontrado. Usando rectángulo blanco.")
    goal_image = None

try:
    goalkeeper_image = pygame.image.load(os.path.join(BASE_DIR, "portero.png")) # Imagen del portero
    goalkeeper_image = pygame.transform.scale(goalkeeper_image, (goalkeeper_width, goalkeeper_height))
except pygame.error:
    print("Advertencia: 'portero.png' no encontrado. Usando rectángulo azul.")
    goalkeeper_image = None

try:
    shoe_image = pygame.image.load(os.path.join(BASE_DIR, "zapato.png")) # Imagen del zapato para patear
    shoe_image = pygame.transform.scale(shoe_image, (80, 80))
except pygame.error:
    print("Advertencia: 'zapato.png' no encontrado. No se mostrará el zapato.")
    shoe_image = None

# --- Variables del juego ---
game_state = "MENU" # Estados: "MENU", "PLAYING", "GOAL", "SAVED"

# Pelota
ball_start_pos = (WIDTH // 2, HEIGHT - 100)
ball_x, ball_y = ball_start_pos
ball_speed_x, ball_speed_y = 0, 0
ball_radius = 25 # Usado si no hay imagen de pelota
ball_kicked = False

# Portería
goal_x, goal_y = (WIDTH - goal_width) // 2, 50
goal_rect = pygame.Rect(goal_x, goal_y, goal_width, goal_height)

# Portero
goalkeeper_speed = 7
goalkeeper_direction = 1 # 1 para derecha, -1 para izquierda
goalkeeper_x = goal_x + (goal_width - goalkeeper_width) // 2 # Centrado inicialmente en la portería
goalkeeper_y = goal_y + (goal_height - goalkeeper_height) // 2
goalkeeper_rect = pygame.Rect(goalkeeper_x, goalkeeper_y, goalkeeper_width, goalkeeper_height)

# Zapato (controlado por la mano)
shoe_width, shoe_height = 80, 80
shoe_x, shoe_y = 0, 0
shoe_rect = pygame.Rect(shoe_x, shoe_y, shoe_width, shoe_height)

# Puntuación
score = 0

# --- Configuración de Mediapipe (importado pero no usado activamente en esta lógica inicial) ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0) # Inicializa la cámara, como en rompe.py
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1)

# --- Funciones del juego ---
def reset_ball():
    global ball_x, ball_y, ball_speed_x, ball_speed_y, ball_kicked
    ball_x, ball_y = ball_start_pos
    ball_speed_x, ball_speed_y = 0, 0
    ball_kicked = False

def draw_elements():
    # Dibujar fondo
    if field_image:
        screen.blit(field_image, (0, 0))
    else:
        screen.fill(GREEN)

    # Dibujar portería
    if goal_image:
        screen.blit(goal_image, (goal_x, goal_y))
    else:
        pygame.draw.rect(screen, WHITE, goal_rect, 5) # Marco de la portería

    # Dibujar portero
    if goalkeeper_image:
        screen.blit(goalkeeper_image, (goalkeeper_x, goalkeeper_y))
    else:
        pygame.draw.rect(screen, BLUE, goalkeeper_rect)

    # Dibujar pelota
    if ball_image:
        screen.blit(ball_image, (ball_x - ball_radius, ball_y - ball_radius))
    else:
        pygame.draw.circle(screen, RED, (int(ball_x), int(ball_y)), ball_radius)

    # Dibujar zapato si el juego está activo
    if shoe_image and game_state == "PLAYING":
        # Dibuja el zapato centrado en la posición de la mano
        screen.blit(shoe_image, (shoe_x - shoe_width // 2, shoe_y - shoe_height // 2))

    # Dibujar puntuación
    score_text = small_font.render(f"Goles: {score}", True, BLACK)
    screen.blit(score_text, (10, 10))

# --- Bucle principal del juego ---
running = True
clock = pygame.time.Clock()

while running:
    # --- Captura de cámara y detección de manos (se hace en cada frame) ---
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # --- Manejo de eventos de Pygame ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and game_state != "PLAYING":
                game_state = "PLAYING"
                score = 0
                reset_ball()

    # --- Lógica principal del juego ---
    if game_state == "PLAYING":
        # Mover el zapato con la mano
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Usar el punto medio de la palma como referencia
                wrist = hand_landmarks.landmark[0]
                shoe_x = int(wrist.x * WIDTH)
                shoe_y = int(wrist.y * HEIGHT)
                shoe_rect.center = (shoe_x, shoe_y)

        # Movimiento del portero
        goalkeeper_x += goalkeeper_speed * goalkeeper_direction
        if goalkeeper_x <= goal_x or goalkeeper_x + goalkeeper_width >= goal_x + goal_width:
            goalkeeper_direction *= -1
        goalkeeper_rect.x = goalkeeper_x

        # Movimiento de la pelota
        ball_rect = pygame.Rect(ball_x - ball_radius, ball_y - ball_radius, ball_radius * 2, ball_radius * 2)

        # Lógica de patada: si el zapato toca la pelota y no ha sido pateada
        if not ball_kicked and shoe_rect.colliderect(ball_rect):
            # --- Lógica de patada mejorada ---
            # Calcular la dirección del tiro basándose en el punto de impacto.
            # Esto simula "empujar" la pelota desde la posición del zapato.
            dir_x = ball_x - shoe_rect.centerx
            dir_y = ball_y - shoe_rect.centery

            # Normalizar el vector para tener una dirección consistente
            distance = np.sqrt(dir_x**2 + dir_y**2)
            if distance == 0: # Evitar división por cero
                distance = 1
            
            kick_speed = 15 # Magnitud de la velocidad del tiro
            ball_speed_x = (dir_x / distance) * kick_speed
            ball_speed_y = (dir_y / distance) * kick_speed
            
            ball_kicked = True

        if ball_kicked:
            ball_x += ball_speed_x
            ball_y += ball_speed_y

            # Colisión con los bordes laterales (para que la pelota no se salga de la pantalla)
            if ball_x - ball_radius <= 0 or ball_x + ball_radius >= WIDTH:
                ball_speed_x *= -1

            # Actualizar el rect de la pelota para las colisiones
            ball_rect = pygame.Rect(ball_x - ball_radius, ball_y - ball_radius, ball_radius * 2, ball_radius * 2)
            
            # Colisión con el portero -> PARADA
            if ball_rect.colliderect(goalkeeper_rect):
                game_state = "SAVED"
                reset_ball()

            # Colisión con la portería -> GOL (solo si no fue parada)
            elif ball_rect.colliderect(goal_rect) and ball_rect.centery < goal_rect.bottom:
                # Asegurarse de que esté dentro de los postes
                if ball_rect.centerx > goal_rect.left and ball_rect.centerx < goal_rect.right:
                    game_state = "GOAL"
                    score += 1
                    reset_ball()

            # Pelota fuera de juego (pasó la portería o se fue por abajo)
            elif ball_y < 0 or ball_y > HEIGHT:
                game_state = "SAVED" # O "MISSED"
                reset_ball()

    # --- Dibujado ---
    draw_elements()

    # Mostrar mensajes de estado
    if game_state == "MENU":
        menu_text = font.render("Presiona ESPACIO para empezar", True, BLACK)
        text_rect = menu_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(menu_text, text_rect)
    elif game_state == "GOAL":
        goal_message = font.render("¡GOLAZO!", True, RED)
        text_rect = goal_message.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(goal_message, text_rect)
    elif game_state == "SAVED":
        saved_message = font.render("¡PARADÓN!", True, BLUE)
        text_rect = saved_message.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(saved_message, text_rect)

    pygame.display.flip()
    clock.tick(60) # Limitar a 60 FPS

# --- Cierre de Pygame y OpenCV ---
pygame.quit()
cap.release()
cv2.destroyAllWindows()