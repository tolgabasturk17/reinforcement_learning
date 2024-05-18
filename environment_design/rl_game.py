"""
SPRITE
"""

import pygame
import random
import numpy as np
from collections import deque
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# window size
WIDTH = 360
HEIGHT = 360
FPS = 30  # frame per seconds

# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class Player(pygame.sprite.Sprite):
    # sprite for the player
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20,20))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.centerx = WIDTH/2
        self.rect.bottom = HEIGHT -1
        self.radius = 10  # Çember yarıçapı
        pygame.draw.circle(self.image, RED, self.rect.center, self.radius)
        self.speedx = 0


    def update(self):
        self.speedx = 0
        keystate = pygame.key.get_pressed()

        if keystate[pygame.K_LEFT]:
            self.speedx = -4
        elif keystate[pygame.K_RIGHT]:
            self.speedx = 4
        else:
            self.speedx = 0

        self.rect.x +=self.speedx

        if self.rect.right > WIDTH:
            self.rect.right = WIDTH;
        if self.rect.left < 0:
            self.rect.left = 0

    def getCoordinate(self):
        return self.rect.x, self.rect.y


class Enemy(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,10))
        self.image.fill(RED)
        self.rect = self.image.get_rect()

        self.rect.x = random.randrange(0,WIDTH-self.rect.width)
        self.rect.y = random.randrange(2,6)

        self.radius = 5  # Çember yarıçapı
        pygame.draw.circle(self.image, WHITE, self.rect.center, self.radius)

        self.speedx = 0
        self.speedy = 3
        #self.radius = 5  # Çember yarıçapı

    def update(self, *args, **kwargs):
        self.rect.x += self.speedx
        self.rect.y += self.speedy

        if self.rect.top > HEIGHT + 10:
            self.rect.x = random.randrange(0, WIDTH - self.rect.width)
            self.rect.y = random.randrange(2, 6)
    def getCoordinate(self):
        return self.rect.x, self.rect.y


class DQLAgent:

    def __init__(self, env):
        # parameters / hyper parameters
        self.state_size = 4
        self.action_size = 3
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=1000)
        self.model = self.build_model()

    def build_model(self):
        # neural network for deep q learning
        model = keras.Sequential()
        model.add(Dense(48, input_dim = self.state_size, activation="tanh"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer = Adam(learning_rate = self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # storage
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        pass

    def replay(self, batch_size):
        # training
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            trained_target = self.model.predict(state)
            trained_target[0][action] = target
            self.model.fit(state, trained_target, verbose=0)

    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay


# initialize pygame and create window
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("BASTURK'S FIRST REINFORCEMENT LEARNING GAME")
clock = pygame.time.Clock()

# sprite
all_sprite = pygame.sprite.Group()
enemy = pygame.sprite.Group()
player = Player()
enemy1 = Enemy()
enemy2 = Enemy()
all_sprite.add(player)
all_sprite.add(enemy1)
all_sprite.add(enemy2)
enemy.add(enemy1)
enemy.add(enemy2)

# game loop
running = True
while running:
    # keep loop running at the right speed
    clock.tick(FPS)

    # process input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # update
    all_sprite.update()

    hits = pygame.sprite.spritecollide(player, enemy, False, pygame.sprite.collide_circle)
    if hits:
        running = False
        print("Game Over!")

    # draw / render(show)
    screen.fill(GREEN)
    all_sprite.draw(screen)

    # flip display after drawing
    pygame.display.flip()

pygame.quit()