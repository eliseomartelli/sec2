import pandas as pd
import pygame
import numpy as np
from PIL import Image


def initialize():
    """
    Initialize Pygame and other components.

    Returns:
    - screen: Pygame screen object.
    - font: Pygame font object for rendering text.
    - models: Dictionary of trained classifiers.
    - constants: Dictionary containing colors and brush properties.
    """

    pygame.init()

    WIDTH, HEIGHT = 280, 280
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Draw a Digit")

    font = pygame.font.Font(None, 36)

    colors = {
        "WHITE": (255, 255, 255),
        "BLACK": (0, 0, 0),
        "BLUE": (0, 0, 255),
    }

    brush_radius = 10

    constants = {
        "WIDTH": WIDTH,
        "HEIGHT": HEIGHT,
        "BRUSH_RADIUS": brush_radius,
        **colors,
    }

    return screen, font, constants


def preprocess_canvas(screen):
    """
    Capture the canvas, preprocess it, and return it as a DataFrame with the
    same columns as the training data (784 features for 28x28 image).
    """
    pixel_data = pygame.surfarray.array3d(screen)
    grayscale = np.mean(pixel_data, axis=2)

    grayscale = 255 - grayscale

    image = Image.fromarray(grayscale).convert("L")
    image = image.resize((28, 28))

    image_array = np.array(image) / 255.0

    flattened_array = image_array.flatten().reshape(
        1, -1)

    column_names = [f"pixel{i+1}" for i in range(flattened_array.shape[1])]

    return pd.DataFrame(flattened_array, columns=column_names)


def predict_digit(models, digit_array):
    """
    Predict the digit using the provided classifiers.

    Parameters:
    - models: Dictionary of trained classifiers.
    - digit_array: Preprocessed flattened image array.

    Returns:
    - predictions: Dictionary of predictions from each classifier.
    """
    predictions = {}
    for name, clf in models.items():
        predictions[name] = clf.predict(digit_array)[0]
    return predictions


def draw_interface(models):
    """
    Main loop for the Pygame drawing interface.

    Parameters:
    - models: Dictionary of trained classifiers.
    """
    screen, font, constants = initialize()
    running = True
    screen.fill(constants['BLACK'])
    font = pygame.font.Font(None, 36)
    predictions = {}

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if pygame.mouse.get_pressed()[0]:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                pygame.draw.circle(
                    screen, constants['WHITE'], (mouse_x, mouse_y),
                    constants['BRUSH_RADIUS'])

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    screen.fill(constants['BLACK'])
                    predictions = {}

                if event.key == pygame.K_p:
                    digit_array = preprocess_canvas(screen)
                    predictions = predict_digit(models, digit_array)

        y_offset = 10
        for model_name, prediction in predictions.items():
            text_surface = font.render(
                f"{model_name}: {prediction}", True, constants['BLUE'])
            screen.blit(text_surface, (10, y_offset))
            y_offset += 40

        pygame.display.flip()

    pygame.quit()

