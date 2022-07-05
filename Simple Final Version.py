# Candidate Num: 237707
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

GLOBAL_OPTIMUM = 4  # For limit up to 2, 2
# Definition of Simple landscape


def SimpleLandscape(x, y):
    return np.where(1-np.abs(2*x) > 0, 1-np.abs(2*x)+x+y, x+y)

# Definition of gradient of Simple landscape


def SimpleLandscapeGrad(x, y):
    g = np.zeros(2)
    if 1 - np.abs(2 * x) > 0:
        if x < 0:
            g[0] = 3
        elif x == 0:
            g[0] = 0
        else:
            g[0] = -1
    else:
        g[0] = 1
    g[1] = 1
    return g

# Function to draw a surface (equivalent to ezmesh in Matlab)
# See argument cmap of plot_surface instruction to adjust color map (if so desired)


def DrawSurface(fig, varxrange, varyrange, function):
    """Function to draw a surface given x,y ranges and a function."""
    ax = fig.gca(projection='3d')
    xx, yy = np.meshgrid(varxrange, varyrange, sparse=False)
    z = function(xx, yy)
    # color map can be adjusted, or removed!
    ax.plot_surface(xx, yy, z, cmap='RdBu')
    fig.canvas.draw()
    return ax


# Function implementing gradient ascent
def GradAscent(StartPt, NumSteps, LRate):
    PauseFlag = 1
    for i in range(NumSteps):

        # Ensure StartPt is within the specified bounds (un/comment relevant lines)
        StartPt = np.maximum(StartPt, [-2, -2])
        StartPt = np.minimum(StartPt, [2, 2])
        # StartPt = np.maximum(StartPt, [-3, -3])
        # StartPt = np.minimum(StartPt, [7, 7])
        print(f"Iteration: {i}")
        print(StartPt)
        # TO =DO: Calculate the 'height' at StartPt using SimpleLandscape or ComplexLandscape
        height = SimpleLandscape(StartPt[0], StartPt[1])
        print(height)

        # TO DO: Plot point on the landscape
        # Use a markersize that you can see well enough (e.g., * in size 10)
        plt.plot(StartPt[0], StartPt[1], height, 'r*')

        # TO DO: Calculate the gradient at StartPt using SimpleLandscapeGrad or ComplexLandscapeGrad
        gradient = SimpleLandscapeGrad(StartPt[0], StartPt[1])
        # print(gradient)

        # TO DO: Calculate the new point and update StartPt
        StartPt = StartPt + LRate * gradient

        # Check if the global optimum is reached or not
        if height == GLOBAL_OPTIMUM:
            return 1, i
            break
        # Return 0 if all the iterations are completed
        if i == NumSteps-1:
            return 0, i
        # Pause to view output
        if PauseFlag:
            x = plt.waitforbuttonpress()


# TO DO: Mutation function
# Returns a mutated point given the old point and the range of mutation
def Mutate(OldPt, MaxMutate):
    # TO DO: Select a random distance MutDist to mutate in the range (-MaxMutate,MaxMutate)
    MutDist = np.random.uniform(-MaxMutate, MaxMutate)
    # TO DO: Randomly choose which element of OldPt to mutate and mutate by MutDist
    randomElement = np.random.choice([0, 1])
    OldPt[randomElement] = OldPt[randomElement] + MutDist
    return OldPt
    # return MutatedPt


# Function implementing hill climbing
def HillClimb(StartPt, NumSteps, MaxMutate):
    PauseFlag = 1
    for i in range(NumSteps):
        print(f"Iteration: {i}")
        print(StartPt)
        # TO DO: Calculate the 'height' at StartPt using SimpleLandscape or ComplexLandscape
        height = SimpleLandscape(StartPt[0], StartPt[1])
        print(height)
        # TO DO: Plot point on the landscape
        # Use a markersize that you can see well enough (e.g., * in size 10)
        plt.plot(StartPt[0], StartPt[1], height, 'r*')

        # Mutate StartPt into NewPt
        # Use copy because Python passes variables by references (see Mutate function)
        NewPt = Mutate(np.copy(StartPt), MaxMutate)

        # Ensure NewPt is within the specified bounds (un/comment relevant lines)
        NewPt = np.maximum(NewPt, [-2, -2])
        NewPt = np.minimum(NewPt, [2, 2])
        #NewPt = np.maximum(NewPt,[-3,-3])
        #NewPt = np.minimum(NewPt,[7,7])

        # TO DO: Calculate the height of the new point
        newHeight = SimpleLandscape(NewPt[0], NewPt[1])
        # TO DO: Decide whether to update StartPt or not
        if newHeight > height:
            StartPt = NewPt

        # Check if the global optimum is reached or not
        if height == GLOBAL_OPTIMUM:
            return (1, i)
        # Return 0 if all the iterations are completed
        if i == NumSteps-1:
            return (0, i)
        # Pause to view output
        if PauseFlag:
            x = plt.waitforbuttonpress()


def GridTest(NumSteps, LRate, MaxMutate):
    possiblePoints = np.linspace(-2, 2, 50)
    x, y = np.meshgrid(possiblePoints, possiblePoints)
    successList = []
    iterationsList = []
    for i in range(50):
        for j in range(50):
            success, iterations = GradAscent(
                np.array([x[i][j], y[i][j]]), NumSteps, LRate)
            # success, iterations = HillClimb(
            #     np.array([x[i][j], y[i][j]]), NumSteps, MaxMutate)
            successList.append(success)
            iterationsList.append(iterations)

    print(
        f"Number of points which reached global maxima: {successList.count(1)}")
    # To calculate mean number of iterations for those points which reached global maxima
    totalIterationCount = 0
    for i, success in enumerate(successList):
        if success == 1:
            totalIterationCount += iterationsList[i]
    print(
        f"Mean iteration for succesful count: {totalIterationCount/successList.count(1)}")

    successArray = np.array(successList).reshape(x.shape)
    plt.pcolormesh(x, y, successArray)
    plt.colorbar()
    plt.waitforbuttonpress()


# Template
# Plot the landscape (un/comment relevant line)
plt.ion()
fig = plt.figure()
ax = DrawSurface(fig, np.arange(-2, 2.025, 0.025),
                 np.arange(-2, 2.025, 0.025), SimpleLandscape)

# Enter maximum number of iterations of the algorithm, learning rate and mutation range
NumSteps = 50
LRate = 0.05
MaxMutate = 0.4
# StartPt = np.array([-1.2, 0.5])
StartPt = np.array([np.random.uniform(-2, 2), np.random.uniform(-2, 2)])
# TO DO: choose a random starting point with x and y in the range (-2, 2) for simple landscape, (-3,7) for complex landscape

# Find maximum (un/comment relevant line)
# GradAscent(StartPt, NumSteps, LRate)
HillClimb(StartPt, NumSteps, MaxMutate)

# GridTest(NumSteps, LRate, MaxMutate)
