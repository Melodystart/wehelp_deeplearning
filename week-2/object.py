import math

def distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

class Line:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    def vector(self):
        return (self.point2[0] - self.point1[0], self.point2[1] - self.point1[1])
    
    def slope(self):
        return self.vector()[0] / self.vector()[1]
    
    def isParallel(self, line):
        return self.slope() == line.slope()
    
    def isPerpendicular(self, line):
        return self.vector()[0] * line.vector()[0] == -1 *self.vector()[1] * line.vector()[1]

class Circle:
    def __init__(self, center, r):
        self.center = center
        self.r = r
    
    def area(self):
        return math.pi * self.r * self.r
    
    def intersects(self, circle):
        return distance(self.center, circle.center) < self.r + circle.r
    
    def isInside(self, point):
        return distance(self.center, point) <= self.r

class Polygon:
    def __init__(self,point1, point2, point3, point4):
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3
        self.point4 = point4

    def perimeter(self):
        return distance(self.point1, self.point2) + distance(self.point2, self.point3) + distance(self.point3, self.point4) + distance(self.point4, self.point1)

# Task 1:
lineA = Line((-6, 1), (2, 4))
lineB = Line((-6,-1), (2, 2))
lineC = Line((-4,-4),(-1, 6))
circleA = Circle((6, 3), 2)
circleB = Circle((8, 1), 1)
polygonA = Polygon((2, 0), (5, -1), (4, -4), (-1, -2))

print("Are Line A and Line B parallel?", lineA.isParallel(lineB))
print("Are Line C and Line A perpendicular?", lineC.isPerpendicular(lineA))  
print("Print the area of Circle A.", circleA.area())
print("Do Circle A and Circle B intersect?", circleA.intersects(circleB))
print("Print the perimeter of Polygon A.", polygonA.perimeter())

# Task1 Answers:
# True
# False
# 12.566370614359172
# True
# 15.315271402935252

print("====================")

# Task 2:
class Enemy:
    def __init__(self, label, position, life, move):
        self.label = label
        self.position = position
        self.life = life
        self.move = move

def attack(enemy, damage):
    if enemy.life > 0:
        enemy.life -= damage
       
T1 = Circle((-3,2),2)
T2 = Circle((-1, -2),2)
T3 = Circle((4, 2),2)
T4 = Circle((7, 0),2)
A1 = Circle((1, 1), 4)
A2 = Circle((4, -3), 4)
E1 = Enemy("E1", (-10, 2), 10, (2, -1))
E2 = Enemy("E2", (-8, 0), 10, (3, 1))
E3 = Enemy("E3", (-9, -1), 10, (3, 0))

BasicTower = [T1, T2, T3, T4]
AdvancedTower = [A1, A2]
Enemies = [E1, E2, E3]

for i in range(1, 11):
    for E in Enemies:
        if E.life > 0:
            E.position = (E.position[0] + E.move[0], E.position[1] + E.move[1])
            for T in BasicTower:
                if T.isInside(E.position):
                    attack(E, 1)
            for A in AdvancedTower:
                if A.isInside(E.position):
                    attack(E, 2)

for E in Enemies:
    print(E.label, E.position[0], E.position[1], E.life)

# Task2 Answers:
# E1 6 -6 0
# E2 22 10 4
# E3 6 -1 0



