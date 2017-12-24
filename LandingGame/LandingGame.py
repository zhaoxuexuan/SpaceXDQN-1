# A simple Landing Game
import pygame
from pygame.locals import *
from sys import exit

# Game stuff
pygame.init()
SCREEN_SIZE = (1600, 900)
screen_width, screen_height = SCREEN_SIZE
screen = pygame.display.set_mode(SCREEN_SIZE, RESIZABLE, 32)
myfont = pygame.font.SysFont("Consolas", 40)
pygame.event.set_allowed([QUIT, VIDEORESIZE, KEYDOWN, KEYUP, MOUSEBUTTONDOWN])

# Scales
unitspeed = screen_width / 5
unitaccle = unitspeed
gravity = 0.5

# Game settings
board = ((screen_width / 5 * 2, screen_height - 10), (screen_width / 5, 10))
landing_veltol = 0.5 * unitspeed
x, y = 50, 50
u_x, u_y = 0, screen_height / 5

# Game propeties
gamewin = gamehit = gameover = False
clock = pygame.time.Clock()
past_time = 0
acc_left, acc_right, acc_up = 0, 0, 0
acc_x, acc_y = 0, 0
trace = [(x, y), (x, y)]

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            exit()

        if event.type == VIDEORESIZE:
            SCREEN_SIZE = event.size
            screen = pygame.display.set_mode(SCREEN_SIZE, RESIZABLE, 32)
            screen_width, screen_height = SCREEN_SIZE
            board = ((screen_width / 5 * 2, screen_height - 10), (screen_width / 5, 10))

        if event.type == KEYDOWN:
            if event.key == K_LEFT:
                acc_left = -1
            elif event.key == K_RIGHT:
                acc_right = 1
            elif event.key == K_UP:
                acc_up = -1
            elif event.key == K_DOWN or event.key == K_SPACE:
                x, y = 0, 0
                acc_left, acc_right, acc_up = 0, 0, 0
                acc_x, acc_y = 0, 0
                u_x, u_y = 0, screen_height / 5
                trace = [(x, y), (x, y)]
                gamehit = gameover = gamewin = False
                clock.tick()
        elif event.type == KEYUP:
            if event.key == K_LEFT:
                acc_left = 0
            elif event.key == K_RIGHT:
                acc_right = 0
            elif event.key == K_UP:
                acc_up = 0

    if gamehit or gameover or gamewin:
        continue

    dt = clock.tick() / 1000.0
    u_x = u_x + (acc_left + acc_right) * unitaccle * dt
    u_y = u_y + (acc_up + gravity) * unitaccle * dt
    x += u_x * dt
    y += u_y * dt

    past_time += dt
    if past_time > 0.01:
        past_time = 0
        trace.append((x, y))

    if y >= board[0][1]:
        if x >= board[0][0] and x <= board[0][0] + board[1][0]:
            if abs(u_x) <= 0.5 * unitspeed and abs(u_y) <= 0.1 * unitspeed:
                gamewin = True
            else:
                gamehit = True
        else:
            gameover = True
    elif x < 0 or x > screen_width:
        gameover = True

    # background
    screen.fill((255,255,255))
    # masspoint
    pygame.draw.circle(screen, (255, 0, 0), (int(x), int(y)), 16)
    # landing board
    pygame.draw.rect(screen, (0, 0, 0), board)
    # motion trace
    pygame.draw.aalines(screen, (112, 128, 144), False, trace, 100)
    if gameover:
        text_surface = myfont.render(str("Game over"), True, (0,0,0))
        screen.blit(text_surface, (0, screen_height / 2))
    elif gamehit:
        text_surface = myfont.render(str("You gota hit with Ux : ") + str(u_x / unitspeed)
        + str(" and Uy : ") + str(u_y / unitspeed), True, (0,0,0))
        screen.blit(text_surface, (0, screen_height / 2))
    elif gamewin:
        text_surface = myfont.render(str("You win!"), True, (0,0,0))
        screen.blit(text_surface, (0, screen_height / 2))
    
    pygame.display.update()