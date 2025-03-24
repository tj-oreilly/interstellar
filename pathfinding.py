"""
This is a basic prototype implementation with the following goals:
    - Generate a set of points in 2D space (the stars)
    - Select a start point and an end point goal
    - Calculate the shortest (time) path for some given "boost" to the velocity per star visited

It is missing a realistic boost function for gravitational slingshot, or indeed the velocity the
ship may lose by changing direction.

You can click on the screen to recalculate for a new set of points.
"""

import pygame
import numpy as np
import heapq

CANVAS_SIZE = 800.0
POINT_COUNT = 100


class BaseObject:
    def __init__(self, pos, mass):
        self.pos = pos
        self.mass = mass


class TrivialObject(BaseObject):
    def __init__(self, pos, mass):
        super().__init__(pos, mass)

    def calculate_boost(self, v_in):
        """
        Calculates the boost the spacecraft receives from this object.

        Params: fuel used, initial velocity, angle from entry
        Returns: final velocity
        """

        return v_in * 1.5  # Very basic multiplier for now


def generate_2d_point(low, high):
    """
    Generates a random 2D vector given some min/max x and y
    """
    return np.random.uniform(low=low, high=high, size=2)


def calculate_cost(from_vertex, to_vertex, v_in):
    """
    Calculates the time cost to go from one vertex to another
    """

    diff = to_vertex.pos - from_vertex.pos
    sq_dist = np.dot(diff, diff)

    v = from_vertex.calculate_boost(v_in)

    return sq_dist / v


def minimise_path(vertices, start_index, end_index, initial_v):
    """
    Calculates the shortest path between a given start and end vertex.

    Slightly different to Dijkstra's algorithm as a given edge cost is dependent on the path taken,
    every vertex is assumed to be connected by an edge.
    """

    max_cost = calculate_cost(vertices[start_index], vertices[end_index], initial_v)
    best_path = [start_index]

    paths_queue = [
        (best_path, 0.0, initial_v)
    ]  # Potential paths (path, cost, velocity) - the end node is implied

    print(f"Initial cost: {max_cost}")

    while paths_queue:
        path, cost, velocity = heapq.heappop(paths_queue)
        final_v_index = path[len(path) - 1]

        # Introduce a new point
        for i, vertex in enumerate(vertices):
            if i == final_v_index or i == end_index:
                continue

            new_cost = cost + calculate_cost(vertices[final_v_index], vertex, velocity)
            end_cost = calculate_cost(
                vertex, vertices[end_index], vertex.calculate_boost(velocity)
            )
            if new_cost + end_cost >= max_cost:
                continue

            new_path = path + [i]
            new_velocity = vertex.calculate_boost(velocity)

            max_cost = new_cost + end_cost  # Update maximum cost
            best_path = new_path + [end_index]

            print(max_cost)

            heapq.heappush(paths_queue, (new_path, new_cost, new_velocity))

    print(f"Optimal cost: {max_cost}")
    print(best_path)

    return best_path, max_cost


def draw_points(vertices, start_index, end_index, screen):
    """
    Draw the vertices on the screen, highlighting the start and end vertex.
    """

    for i, vertex in enumerate(vertices):
        if i == start_index or i == end_index:
            col = [255, 0, 0]
            size = 5.0
        else:
            col = [255, 255, 255]
            size = 2.0

        pygame.draw.circle(screen, col, vertex.pos, size)

    pygame.display.flip()


def draw_path(vertices, path, screen):
    """
    Draws a given path of vertices
    """

    for i, vertex_index in enumerate(path):
        if i < len(path) - 1:
            pygame.draw.line(
                screen,
                [0, 255, 0],
                vertices[vertex_index].pos,
                vertices[path[i + 1]].pos,
            )

    pygame.display.flip()


def execute(screen):
    screen.fill([0, 0, 0])

    # These are the vertices
    objects = [
        TrivialObject(generate_2d_point(0.0, CANVAS_SIZE), 1.0)
        for i in range(POINT_COUNT)
    ]

    start_index = 0
    end_index = np.random.randint(1, len(objects))

    draw_points(objects, start_index, end_index, screen)

    # Assume starting at vertex 0
    # Apply Dijkstra's algorithm where the cost is calculated dynamically given previous conditions
    # and the boost function
    path, cost = minimise_path(objects, start_index, end_index, 1000.0)

    draw_path(objects, path, screen)


def main():
    # Set up window
    pygame.init()
    screen = pygame.display.set_mode([CANVAS_SIZE, CANVAS_SIZE])
    pygame.display.set_caption("Pathfinding")

    execute(screen)

    # Window event loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                execute(screen)


if __name__ == "__main__":
    main()
