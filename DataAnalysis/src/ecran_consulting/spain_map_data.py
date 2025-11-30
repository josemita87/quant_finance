"""
Spain regional coordinates for map visualization.
Approximate coordinates for major Spanish cities/regions.
"""

# Approximate coordinates for Spanish regions (latitude, longitude)
SPAIN_REGIONS = {
    'Madrid': {'lat': 40.4168, 'lon': -3.7038, 'size': 'large'},
    'Barcelona': {'lat': 41.3851, 'lon': 2.1734, 'size': 'large'},
    'Valencia': {'lat': 39.4699, 'lon': -0.3763, 'size': 'medium'},
    'Seville': {'lat': 37.3891, 'lon': -5.9845, 'size': 'medium'},
    'Málaga': {'lat': 36.7213, 'lon': -4.4214, 'size': 'medium'},
    'Bilbao': {'lat': 43.2630, 'lon': -2.9340, 'size': 'medium'},
}

# Spain country boundaries (approximate polygon)
SPAIN_BOUNDARY = [
    (-9.3, 43.8),   # Northwest
    (-7.0, 43.8),   # North
    (-1.8, 43.4),   # North
    (3.3, 42.4),    # Northeast
    (3.3, 39.0),    # East
    (0.3, 38.0),    # Southeast
    (-0.5, 36.0),   # South
    (-5.6, 36.0),   # Southwest
    (-9.3, 37.0),   # West
    (-9.3, 43.8),   # Back to start
]
