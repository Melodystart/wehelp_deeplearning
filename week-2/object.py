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
class BasicTower(Circle):
    def __init__(self, center):
        super().__init__(center, 2)
        self.damage = 1

    def attack(self, enemy):
        if enemy.life > 0:
            enemy.life -= self.damage

class AdvancedTower(BasicTower):
    def __init__(self, center):
        super().__init__(center)
        self.r = 4
        self.damage = 2

class Enemy:
    def __init__(self, label, position, life, move):
        self.label = label
        self.position = position
        self.life = life
        self.move = move

       
T1 = BasicTower((-3,2))
T2 = BasicTower((-1, -2))
T3 = BasicTower((4, 2))
T4 = BasicTower((7, 0))
A1 = AdvancedTower((1, 1))
A2 = AdvancedTower((4, -3))
E1 = Enemy("E1", (-10, 2), 10, (2, -1))
E2 = Enemy("E2", (-8, 0), 10, (3, 1))
E3 = Enemy("E3", (-9, -1), 10, (3, 0))

Towers = [T1, T2, T3, T4, A1, A2]
Enemies = [E1, E2, E3]

for i in range(1, 11):
    for E in Enemies:
        if E.life > 0:
            E.position = (E.position[0] + E.move[0], E.position[1] + E.move[1])
            for T in Towers:
                if T.isInside(E.position):
                    T.attack(E)

for E in Enemies:
    print(E.label, E.position[0], E.position[1], E.life)

# Task2 Answers:
# E1 6 -6 0
# E2 22 10 4
# E3 6 -1 0



