import pygame
import numpy as np
from tanchisnake import GameEnvironment
from model import QNetwork, get_network_input
import torch
import sys

gridsize = 15  # 13
framerate = 10
block_size = 20

model = QNetwork(input_dim=10, hidden_dim=20, output_dim=5)
## model.load_state_dict(torch.load('./dir_chk_lr0.0001/Snake_40000'))
model.load_state_dict(torch.load('./dir_chk_len/Snake_16750'))

board = GameEnvironment(gridsize, 0., -100., 100.)
windowwidth = gridsize * block_size * 2
windowheight = gridsize * block_size

pygame.init()#pygame 初始化
win = pygame.display.set_mode((windowwidth, windowheight))#设置pygame窗口
pygame.display.set_caption("snake")
font = pygame.font.SysFont('arial', 18)
clock = pygame.time.Clock()


def drawboard(snake, apple):#通过pygame绘制可视化贪吃蛇运动的
    win.fill((0, 0, 0))
    for pos in snake.prevpos:#逐个绘制贪吃蛇的身体（贪吃蛇身体由不同的小block组成）
        pygame.draw.rect(win, (0, 255, 0), (pos[0] * block_size, pos[1] * block_size, block_size, block_size))
    pygame.draw.rect(win, (255, 0, 0), (apple.pos[0] * block_size, apple.pos[1] * block_size, block_size, block_size))#绘制苹果


runGame = True

prev_len_of_snake = 0

while runGame:
    clock.tick(framerate)

    state_0 = get_network_input(board.snake, board.apple)
    state = model(state_0)

    action = torch.argmax(state)

    reward, done, len_of_snake = board.update_boardstate(action)
    drawboard(board.snake, board.apple)

    lensnaketext = font.render('          LEN OF SNAKE: ' + str(len_of_snake), False, (255, 255, 255))
    rewardtext = font.render('          REWARD: ' + str(int(reward)), False, (255, 255, 255))
    prevlensnaketext = font.render('          LEN OF PREVIOUS SNAKE: ' + str(prev_len_of_snake), False, (255, 255, 255))

    win.blit(lensnaketext, (windowwidth // 2, 40))
    win.blit(rewardtext, (windowwidth // 2, 80))
    win.blit(prevlensnaketext, (windowwidth // 2, 120))

    for event in pygame.event.get():#pygame 推出
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            runGame = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_r]:
        paused = True
        while paused == True:
            clock.tick(10)
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    paused = False

    pygame.display.update()

    if board.game_over == True:
        prev_len_of_snake = len_of_snake
        board.resetgame()

pygame.quit()