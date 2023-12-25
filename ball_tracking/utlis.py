import matplotlib.pyplot as plt

height = 640
width = 384

# list = [[x, y, x, y], [x, y, x, y], [x, y, x, y]]

def Coordinatates_to_midpoint(Coordinates):
    trajectory_list = []

    for i in range(len(Coordinates)):
        x = (abs(Coordinates[0] - Coordinates[2]))/2
        y = (abs(Coordinates[1] - Coordinates[3]))/2
        trajectory_list.append([x, y])

    return trajectory_list


def plot_trajectory(trajectory_list):
    plt.figure(figsize=(width, height))
    
    for i in trajectory_list:

        plt.scatter(i[0], i[1])
        plt.pause(0.05)
        plt.show()