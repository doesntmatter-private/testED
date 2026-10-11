"""Pure Python geometry helpers. No third party dependencies so tests run anywhere."""
from __future__ import annotations

Point = tuple[float, float]


def point_in_polygon(pt: Point, poly: list[Point]) -> bool:
    """Ray casting test. Polygon is a list of (x, y) vertices in order."""
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def normalize(v: Point) -> Point:
    mag = (v[0] ** 2 + v[1] ** 2) ** 0.5
    if mag == 0:
        return (0.0, 0.0)
    return (v[0] / mag, v[1] / mag)
