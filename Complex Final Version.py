#Candidate Num: 237707

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Definition of Complex landscape


def ComplexLandscape(x, y):
    return 4 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - \
        15 * (x / 5 - x ** 3 - y ** 5) * np.exp(-x ** 2 - y ** 2) - \
        (1./3)*np.exp(-(x+1)**2 - y**2)-1*(2*(x-3)**7 -
                                           0.3*(y-4)**5+(y-3)**9)*np.exp(-(x-3)**2-(y-3)**2)

# Definition of gradient of Complex landscape


def ComplexLandscapeGrad(x, y):
    g = np.zeros(2)
    g[0] = -8 * np.exp(-(x**2)-(y+1)**2)*((1-x)+x*(1-x)**2)-15 * np.exp(-x**2-y**2)*((0.2-3*x**2) - 2*x*(x/5 - x**3 - y**5)) + (
        2./3)*(x+1) * np.exp(-(x+1)**2 - y**2)-1 * np.exp(-(x-3)**2-(y-3)**2)*(14*(x-3)**6-2*(x-3)*(2*(x-3)**7-0.3*(y-4)**5+(y-3)**9))
    g[1] = -8*(y+1)*(1-x)**2 * np.exp(-(x**2)-(y+1)**2) - 15 * np.exp(-x**2-y**2)*(-5*y**4 - 2*y*(x/5 - x**3 - y**5)) + (2./3)*y * \
        np.exp(-(x+1)**2 - y**2)-1 * np.exp(-(x-3)**2-(y-3)**2) * \
        ((-1.5*(y-4)**4+9*(y-3)**8)-2*(y-3)*(2*(x-3)**7-0.3*(y-4)**5+(y-3)**9))
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
        print(f"Iteration: {i}")
        # Ensure StartPt is within the specified bounds (un/comment relevant lines)
        # StartPt = np.maximum(StartPt, [-2, -2])
        # StartPt = np.minimum(StartPt, [2, 2])
        StartPt = np.maximum(StartPt, [-3, -3])
        StartPt = np.minimum(StartPt, [7, 7])

        # TO =DO: Calculate the 'height' at StartPt using SimpleLandscape or ComplexLandscape
        height = ComplexLandscape(StartPt[0], StartPt[1])
        # print(height)

        # TO DO: Plot point on the landscape
        # Use a markersize that you can see well enough (e.g., * in size 10)
        plt.plot(StartPt[0], StartPt[1], height, 'r*')

        # TO DO: Calculate the gradient at StartPt using SimpleLandscapeGrad or ComplexLandscapeGrad
        gradient = ComplexLandscapeGrad(StartPt[0], StartPt[1])
        # print(gradient)

        # TO DO: Calculate the new point and update StartPt
        StartPt = StartPt + LRate * gradient
        print(StartPt)

        # Return 0 if all the iterations are completed
        if i == NumSteps-1:
            return height
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
        # print(StartPt)
        # TO DO: Calculate the 'height' at StartPt using SimpleLandscape or ComplexLandscape
        height = ComplexLandscape(StartPt[0], StartPt[1])
        # print(height)
        # TO DO: Plot point on the landscape
        # Use a markersize that you can see well enough (e.g., * in size 10)
        plt.plot(StartPt[0], StartPt[1], height, 'r*')

        # Mutate StartPt into NewPt
        # Use copy because Python passes variables by references (see Mutate function)
        NewPt = Mutate(np.copy(StartPt), MaxMutate)

        # Ensure NewPt is within the specified bounds (un/comment relevant lines)
        # NewPt = np.maximum(NewPt, [-2, -2])
        # NewPt = np.minimum(NewPt, [2, 2])
        NewPt = np.maximum(NewPt, [-3, -3])
        NewPt = np.minimum(NewPt, [7, 7])

        # TO DO: Calculate the height of the new point
        newHeight = ComplexLandscape(NewPt[0], NewPt[1])
        # TO DO: Decide whether to update StartPt or not
        if newHeight > height:
            StartPt = NewPt

        if i == NumSteps-1:
            return height
        # Pause to view output
        if PauseFlag:
            x = plt.waitforbuttonpress()


def GridTest(NumSteps, LRate, MaxMutate):
    possiblePoints = np.linspace(-3, 7, 100)
    x, y = np.meshgrid(possiblePoints, possiblePoints)
    heightList = []
    for i in range(100):
        for j in range(100):
            # height = GradAscent(
            #     np.array([x[i][j], y[i][j]]), NumSteps, LRate)
            height = HillClimb(
                np.array([x[i][j], y[i][j]]), NumSteps, MaxMutate)
            heightList.append(height)

    heightArray = np.array(heightList).reshape(x.shape)
    plt.pcolormesh(x, y, heightArray)
    plt.colorbar()
    plt.waitforbuttonpress()


# Template
# Plot the landscape (un/comment relevant line)
plt.ion()
fig = plt.figure()


ax = DrawSurface(fig, np.arange(-3, 7.025, 0.025),
                 np.arange(-3, 7.025, 0.025), ComplexLandscape)
# Enter maximum number of iterations of the algorithm, learning rate and mutation range
NumSteps = 50
LRate = 0.1
MaxMutate = 1
# StartPt = np.array([-0.5, 0])
StartPt = np.array([np.random.uniform(-3, 7), np.random.uniform(-3, 7)])

# TO DO: choose a random starting point with x and y in the range (-2, 2) for simple landscape, (-3,7) for complex landscape

# Find maximum (un/comment relevant line)
# GradAscent(StartPt, NumSteps, LRate)
HillClimb(StartPt, NumSteps, MaxMutate)
# GridTest(NumSteps, LRate, MaxMutate)
