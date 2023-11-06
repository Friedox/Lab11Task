from __future__ import annotations
from abc import ABC
from itertools import product
from time import sleep
import math
import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LineString
from descartes import PolygonPatch
from typing import List, Tuple
import random
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

from shapely.geometry import Point
import numpy as np


class MyPoint:
    def __init__(self, *args):
        self.point = Point(*args)
        self.x, self.y = self.point.x, self.point.y

    def __add__(self, other):
        return MyPoint(self.x + other.x, self.y + other.y)

    def scale(self, ratio):
        return MyPoint(self.x * ratio, self.y * ratio)

    def get_xy(self):
        return self.x, self.y

    def rotate(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        r = np.array([[c, -s], [s, c]])
        new_xy = list(np.matmul(r, self.get_xy()))
        return MyPoint(new_xy[0], new_xy[1])


class MyLineString(LineString):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def bearing(self):
        return math.atan2(self.coords[1][1] - self.coords[0][1],
                          self.coords[1][0] - self.coords[0][0])

    def get_angle(self, other):
        return math.fabs(self.bearing() - other.bearing()) % math.pi


class Obstacle:
    def __init__(self, center_point, size=1.0):
        corners = [MyPoint(-1, -1), MyPoint(-1, 1), MyPoint(1, 1), MyPoint(1, -1)]
        scaled_corners = [p.scale(size) + center_point for p in corners]
        self.points = scaled_corners
        self.center = center_point

    def get_drawable(self, color):
        return plt.Polygon([(p.x, p.y) for p in self.points], color=color)

    def get_center(self):
        return self.center


class Robot:
    _points: List[Point]
    lines: List[MyLineString]

    def __init__(self, start_point, end_point, grid_num, obstacles):
        self.start: MyPoint = start_point
        self.stop: MyPoint = end_point
        self._point_num: int = grid_num
        self.obstacles: List[Obstacle] = obstacles
        self._create_st_line()

    def _create_st_line(self):
        # Line that connects start and stop points
        self.st_line = MyLineString(
            [self.start.get_xy(), self.stop.get_xy()])
        self.theta = math.atan2(self.stop.y - self.start.y,
                                self.stop.x - self.start.x)
        self.__x_prime_array = np.arange(
            0, self.st_line.length + self.st_line.length / self._point_num,
               self.st_line.length / self._point_num)

        self._points = [MyPoint(x, 0) for x in self.__x_prime_array]
        self.lines = []

    def set_start_stop_point(self, s_point, t_point):
        self.start = s_point
        self.stop = t_point
        self._create_st_line()

    def update_points(self, points):
        points = [0] + points + [0]
        self._points = [MyPoint(x, y).rotate(self.theta) for x, y in zip(self.__x_prime_array, points)]
        self._points = [MyPoint(p.x, p.y) + self.start for p in self._points]
        self.lines = [
            MyLineString([p1.get_xy(), p2.get_xy()]) for
            p1, p2 in zip([self.start] + self._points, self._points + [self.stop])]

    def get_cost(self, points=None) -> float:
        if points is not None:
            self.update_points(points)

        # Define coefficients for different cost components
        length_coeff = 0.1  # Adjust coefficients as needed
        angle_coeff = 0.2
        clearance_coeff = 0.3
        intersection_coeff = 0.4

        # Calculate cost components
        length_penalty = length_coeff * self.get_length_penalty(self.lines)
        angle_penalty = angle_coeff * self.get_angle_penalty(self.lines)
        clearance_penalty = clearance_coeff * self.get_clearance_penalty(self.lines, self.obstacles)
        intersection_penalty = intersection_coeff * self.get_intersection_penalty(self.lines, self.obstacles)

        # Sum up the cost components
        total_cost = length_penalty + angle_penalty + clearance_penalty + intersection_penalty

        return total_cost

    def get_intersection_penalty(self, lines, obstacles) -> float:
        intersection_penalty = 0.0
        for i in range(len(lines) - 1):
            line1 = lines[i]
            line2 = lines[i + 1]
            for obstacle in obstacles:
                if self.do_lines_intersect(line1, obstacle) or self.do_lines_intersect(line2, obstacle):
                    intersection_penalty += 1.0  # You can adjust the penalty value as needed
        return intersection_penalty

    def do_lines_intersect(self, line, obstacle):
        # Check if the input line is a LineString object and convert it to a tuple if necessary
        if isinstance(line, LineString):
            line_coords = tuple(line.coords)

            if len(line_coords) == 2:
                x1, y1 = line_coords[0]
                x2, y2 = line_coords[1]
            else:
                # Handle the case where LineString has more than 2 coordinates (e.g., a multi-segment line)
                # You can choose how to handle such cases, such as considering the first and last points as endpoints
                x1, y1 = line_coords[0]
                x2, y2 = line_coords[-1]
        else:
            # Handle other cases where line is not a LineString object
            x1, y1, x2, y2 = line  # Assuming line is already in the format (x1, y1, x2, y2)

        # Get the vertices of the obstacle polygon
        obstacle_vertices = [(p.x, p.y) for p in obstacle.points]

        # Check for intersection by iterating through the vertices of the obstacle polygon
        for i in range(len(obstacle_vertices)):
            x3, y3 = obstacle_vertices[i]
            x4, y4 = obstacle_vertices[(i + 1) % len(obstacle_vertices)]  # Wrap around to the first vertex

            # Check for intersection between the line segment and each edge of the obstacle polygon
            if self.do_line_segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
                return True  # Intersection detected

        return False  # No intersection

    def do_line_segments_intersect(self, x1, y1, x2, y2, x3, y3, x4, y4):
        # Check for intersection between two line segments (x1, y1) - (x2, y2) and (x3, y3) - (x4, y4)

        # Calculate the direction vectors of the line segments
        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3

        # Calculate determinants to check if the segments are parallel
        determinant = dx1 * dy2 - dx2 * dy1

        if determinant == 0:
            # Line segments are parallel; check if they overlap
            if (x1 == x3 and y1 == y3) or (x1 == x4 and y1 == y4):
                return True
            if (x2 == x3 and y2 == y3) or (x2 == x4 and y2 == y4):
                return True

            # Check if the endpoints of one segment are on the other segment
            if (
                    min(x1, x2) <= x3 <= max(x1, x2) and min(y1, y2) <= y3 <= max(y1, y2) or
                    min(x1, x2) <= x4 <= max(x1, x2) and min(y1, y2) <= y4 <= max(y1, y2) or
                    min(x3, x4) <= x1 <= max(x3, x4) and min(y3, y4) <= y1 <= max(y3, y4) or
                    min(x3, x4) <= x2 <= max(x3, x4) and min(y3, y4) <= y2 <= max(y3, y4)
            ):
                return True
            return False

        # Calculate the parameters for the line equations
        t1 = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / determinant
        t2 = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / determinant

        # Check if the line segments intersect within their parameter ranges (0 to 1)
        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            return True

        return False

    def get_length_penalty(self, lines):
        if lines is not None:
            length_penalty = sum(self.calculate_segment_length(segment) for segment in lines)
            return length_penalty
        else:
            return 0.0  # Default value if lines are not available

    def get_angle_penalty(self, lines):
        if lines is not None:
            angle_penalty = self.calculate_angle_penalty(lines)
            return angle_penalty if angle_penalty is not None else 0.0  # Return 0.0 if no valid angle penalty
        else:
            return 0.0  # Default value if lines are not available

    def calculate_angle_penalty(self, lines):
        if len(lines) < 3:
            return 0.0  # Not enough line segments to calculate angles

        total_angle = 0.0

        for i in range(1, len(lines) - 1):
            line1 = lines[i - 1]
            line2 = lines[i]
            line3 = lines[i + 1]

            # Calculate the angle between line2 and the average direction of line1 and line3
            angle = self.calculate_angle(line1, line2, line3)

            total_angle += angle

        # Calculate the average angle per line segment
        average_angle = total_angle / (len(lines) - 2)

        # Define a threshold for "sharp" turns and penalize them
        sharp_turn_threshold = 45.0  # You can adjust this threshold as needed
        angle_penalty = max(0.0, average_angle - sharp_turn_threshold)

        return angle_penalty

    from shapely.geometry import LineString

    def calculate_angle(self, line1, line2, line3):
        # Check if the input lines are LineString objects and convert them to tuples if necessary
        if isinstance(line1, LineString):
            line1 = tuple(line1.coords[0]) + tuple(line1.coords[-1])
        if isinstance(line2, LineString):
            line2 = tuple(line2.coords[0]) + tuple(line2.coords[-1])
        if isinstance(line3, LineString):
            line3 = tuple(line3.coords[0]) + tuple(line3.coords[-1])

        # Extract the endpoints of the line segments
        x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6 = line1 + line2 + line3

        # Calculate vectors (x1, y1) -> (x2, y2) and (x3, y3) -> (x4, y4)
        v1 = (x2 - x1, y2 - y1)
        v2 = (x4 - x3, y4 - y3)

        # Calculate vectors (x3, y3) -> (x4, y4) and (x5, y5) -> (x6, y6)
        v3 = (x4 - x3, y4 - y3)
        v4 = (x6 - x5, y6 - y5)

        # Calculate the dot product of the vectors
        dot_product = v1[0] * v3[0] + v1[1] * v3[1]

        # Calculate the magnitudes of the vectors
        magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        magnitude_v3 = math.sqrt(v3[0] ** 2 + v3[1] ** 2)

        if magnitude_v1 == 0.0 or magnitude_v3 == 0.0:
            return 0.0  # Avoid division by zero

        # Calculate the angle in radians
        angle_radians = math.acos(dot_product / (magnitude_v1 * magnitude_v3))

        # Convert the angle to degrees
        angle_degrees = math.degrees(angle_radians)

        return angle_degrees

    def get_clearance_penalty(self, lines, obstacles):
        if lines is not None and obstacles is not None:
            clearance_penalty = self.calculate_clearance_penalty(lines, obstacles)
            return clearance_penalty if clearance_penalty is not None else 0.0  # Return 0.0 if no valid clearance penalty
        else:
            return 0.0  # Default value if lines or obstacles are not available

    def calculate_clearance_penalty(self, lines, obstacles):
        min_clearance = float('inf')  # Initialize with a large value

        for line in lines:
            for obstacle in obstacles:
                # Calculate the distance between the line segment and the obstacle
                clearance = self.calculate_clearance(line, obstacle)
                min_clearance = min(min_clearance, clearance)

        # Define a threshold for "too close" to obstacles and penalize accordingly
        too_close_threshold = 1.0  # You can adjust this threshold as needed
        clearance_penalty = max(0.0, too_close_threshold - min_clearance)

        return clearance_penalty

    def calculate_clearance(self, line, obstacle):
        # Check if the input line is a LineString object and convert it to a tuple if necessary
        if isinstance(line, LineString):
            line = tuple(line.coords[0]) + tuple(line.coords[-1])

        # Get the center of the obstacle
        obstacle_center_x = obstacle.center.x
        obstacle_center_y = obstacle.center.y

        # Extract the endpoints of the line segment
        x1, y1, x2, y2 = line

        # Calculate the closest point on the line segment to the center of the obstacle
        closest_x, closest_y = self.closest_point_on_line(x1, y1, x2, y2, obstacle_center_x, obstacle_center_y)

        # Calculate the Euclidean distance between the closest point and the center of the obstacle
        distance = math.sqrt((closest_x - obstacle_center_x) ** 2 + (closest_y - obstacle_center_y) ** 2)

        return distance

    def closest_point_on_line(self, x1, y1, x2, y2, x3, y3):
        # Calculate the closest point (x, y) on the line (x1, y1) - (x2, y2) to the point (x3, y3)
        # Based on the formula for projecting a point onto a line segment

        # Calculate the direction vector of the line segment
        dx = x2 - x1
        dy = y2 - y1

        # Calculate the vector from (x1, y1) to (x3, y3)
        qx = x3 - x1
        qy = y3 - y1

        # Calculate the dot product of the direction vector and the vector to the point
        dot_product = dx * qx + dy * qy

        if dot_product <= 0:
            return x1, y1  # The closest point is at the start of the line segment

        squared_length = dx * dx + dy * dy

        if dot_product >= squared_length:
            return x2, y2  # The closest point is at the end of the line segment

        t = dot_product / squared_length

        x = x1 + t * dx
        y = y1 + t * dy

        return x, y

    def get_path(self):
        return LineString([p.get_xy() for p in self._points])

    def calculate_segment_length(self, segment):
        if isinstance(segment, LineString):
            coordinates = list(segment.coords)
            if len(coordinates) == 2:
                (x1, y1), (x2, y2) = coordinates
                # Calculate the Euclidean distance between the two endpoints
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                return length

        # Return 0.0 for invalid or non-LineString segments
        return 0.0


class Chromosome:
    genes: np.ndarray  # Use np.ndarray, not np.NDArray

    def __init__(self, genes_len=10, gene_pool_min=-5.0, gene_pool_max=5.0, genes=None):
        if genes is None:
            self.genes = np.random.uniform(gene_pool_min, gene_pool_max, genes_len)
        else:
            self.genes = genes

    def mutate(self, gene_pool_min: float, gene_pool_max: float) -> Chromosome:
        mutated_genes = self.genes.copy()  # Create a copy of the current genes

        # Choose a random gene to mutate
        gene_to_mutate_index = random.randint(0, len(self.genes) - 1)

        # Generate a new value for the selected gene within the specified range
        new_gene_value = np.random.uniform(gene_pool_min, gene_pool_max)

        # Apply the mutation by updating the selected gene
        mutated_genes[gene_to_mutate_index] = new_gene_value

        # Create a new chromosome with the mutated genes
        mutated_chromosome = Chromosome(mutated_genes)

        return mutated_chromosome

    def crossover(self, other: "Chromosome") -> Tuple["Chromosome", "Chromosome"]:
        # Determine the crossover point
        crossover_point = random.randrange(len(self.genes))

        # Perform one-point crossover
        child1_genes = np.concatenate((self.genes[:crossover_point], other.genes[crossover_point:]))
        child2_genes = np.concatenate((other.genes[:crossover_point], self.genes[crossover_point:]))

        # Create child chromosomes with the new genes
        child1 = Chromosome(genes=child1_genes)
        child2 = Chromosome(genes=child2_genes)

        return child1, child2

    def get_genes(self):
        return list(self.genes).copy()


class GA:
    # get size of population and chromosome and talent size at the first
    def __init__(self, chr_size, talent_size):
        self._chr_size = chr_size
        self._talentSize = talent_size
        self.population = []
        self.top = {"cost_value": float('Inf'), "chr": []}

    def reset_top(self):
        self.top = {"cost_value": float('Inf'), "chr": []}

    def reset(self, pop_size):
        self.population = []
        self.gen_population(gene_pool_min=-3, gene_pool_max=3, pop_size=pop_size)

    def append_population(self, population):
        if population is not None:
            self.population = self.population + population

    def change_population(self, pop):
        del (self.population[int(len(self.population) / 2):])
        self.append_population(pop)

    def gen_population(self, gene_pool_max, gene_pool_min, pop_size):
        for p in range(pop_size):
            self.population.append(
                Chromosome(self._chr_size, gene_pool_min, gene_pool_max))
        return self.population

    def mutation(self, num, gene_pool_min, gene_pool_max):
        def mutation(self, num, gene_pool_min, gene_pool_max):
            if num > len(self.population):
                num = len(self.population)
            mutated = []
            mutate_indexes = np.random.randint(0, len(self.population), num)
            for mutate_index in mutate_indexes:
                mutated += [self.population[mutate_index].mutate(int(gene_pool_min), int(gene_pool_max))]
            return mutated

    def crossover(self, num):
        if len(self.population) <= 0:
            return []  # Handle the case where the population is empty

        crossovered = []
        for _ in range(num):
            s = list(np.random.randint(0, len(self.population), 2))
            # Rest of your crossover logic goes here
        return crossovered

    def calc_fitness(self, func, pop=None):
        if pop is None:
            pop = self.population

        fitness_list = [func(chr_.get_genes()) for chr_ in pop if func(chr_.get_genes()) is not None]

        if not fitness_list:
            return [], None  # Handle the case where all fitness values are None

        sorted_list = sorted(zip(fitness_list, pop), key=lambda f: f[0])
        # print("chromosome with fitness =",[(a[0], a[1].getGenes()) for a in sorted_list])
        sorted_chromosome = [s[1] for s in sorted_list]

        top_fitness = sorted_list[0][0]

        if self.top["cost_value"] > top_fitness:
            self.top["cost_value"] = top_fitness
            self.top["chr_"] = sorted_list[0][1]
        return sorted_chromosome, top_fitness


# create robot object
run_index = 1
flag = True
grid_size = 15
pop_size = 20
r = Robot(MyPoint(0, 0), MyPoint(10, 10), grid_size + 1, None)
ga = GA(chr_size=grid_size, talent_size=3)
g = ga.gen_population(gene_pool_min=-5.0, gene_pool_max=5.0, pop_size=pop_size)


def ga_iterate(num, mutate_chance=0.8, mutate_min=-15.0, mutate_max=15.0):
    global flag
    cost = []
    for i in range(num):
        best_path, most_fit = ga.calc_fitness(r.get_cost)
        cost.append(most_fit)
        ga.population = best_path
        crossovered = ga.crossover(int(pop_size / 2))
        # best_crossovered_path = ga.calPopFitness(r.getFitness, pop=crossovered)
        # print("len ga", len(ga.getPopulation()))
        if flag:
            ga.append_population(crossovered)
            flag = False
        else:
            ga.change_population(crossovered)

        a = np.random.uniform(0, 1, 1)
        if a < mutate_chance:
            mutated = ga.mutation(pop_size, mutate_min, mutate_max)
            ga.change_population(mutated)
    return best_path, cost


ITERATION_NUMBER = 10
RUN_NUMBER = 5

START = (1, 1)
END = (10, 10)


def run():
    for _ in range(RUN_NUMBER):
        fig, ax = plt.subplots()
        ax.clear()
        best_path, score = ga_iterate(num=ITERATION_NUMBER)
        if best_path and len(best_path) > 0:
            r.update_points(list(best_path[0].get_genes()))

            p = r.get_path()

            ax.grid(b=None, which='both', axis='both')
            draw_start_stop_points(ax, r.start, r.stop)
            draw_obstacles(ax, r.obstacles)
            ax.autoscale(enable=True, axis='both', tight=None)

            draw_path(ax, p)
            print("Fit:", r.get_cost())
            print("Length penalty: {}".format(r.get_length_penalty()))
            print("Angle penalty: {}".format(r.get_angle_penalty()))
            print("Clearance penalty: {}".format(r.get_clearance_penalty()))
            print("Intersection penalty: {}".format(r.get_intersection_penalty()))
            plt.show()
        else:
            # Handle the case where no valid path is found
            print("No valid path found.")


# some function for better viewing
def draw_start_stop_points(ax, start, end):
    ax.plot([start.x], [start.y], 'ro', color="blue"),
    ax.annotate("start", xy=(start.x, start.y),
                xytext=(start.x, start.y + 0.2))
    ax.plot([end.x], [end.y], 'ro', color="blue")
    ax.annotate("end", xy=(end.x, end.y),
                xytext=(end.x, end.y + 0.2))


def draw_obstacles(ax, obstacles, color="red"):
    for obs in obstacles:
        ax.add_patch(obs.get_drawable(color))


def draw_path(ax, p):
    ax.add_line(
        mlines.Line2D([p.coords[i][0] for i in range(len(p.coords))],
                      [p.coords[i][1] for i in range(len(p.coords))],
                      color="green"))


def reset_obstacle(ax):
    global pop_size
    ga.reset_top()
    ga.reset(pop_size)
    ax.clear()
    ax.grid(True, which='both', axis='both')
    obstacles = [
        Obstacle(MyPoint(random.randint(1, 15), random.randint(1, 15)), 0.5)
        for i in range(10)]
    r.obstacles = obstacles
    draw_obstacles(ax, obstacles)
    ax.autoscale(enable=True, axis='both', tight=None)


fig, axis = plt.subplots()
reset_obstacle(axis)
plt.show()
run()
