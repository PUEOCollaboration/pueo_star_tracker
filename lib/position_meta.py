"""
Position Metadata Definition for Astrometry Telemetry

This module defines the metadata structure for astronomical position and orientation
telemetry data. The metadata provides descriptions, units, and type information for
all fields in the position telemetry format.

The metadata follows the JSON Schema-like structure for self-documenting telemetry data.
"""

position_meta = {
    'timestamp': {
        'description': 'Timestamp of the position measurement in UTC',
        'units': 'ISO 8601 datetime string (YYYY-MM-DDTHH:MM:SS.ssssss)',
        'type': 'string',
        'required': True
    },
    'solver': {
        'description': 'Name of the astrometry solver algorithm used for star identification',
        'units': 'none',
        'type': 'string',
        'required': True,
        'examples': ['solver1', 'astrometry-net', 'custom-solver']
    },
    'astro_position': {
        'description': 'Astronomical orientation coordinates representing spacecraft attitude',
        'units': 'degrees',
        'type': 'array[3]',
        'required': True,
        'elements': [
            {
                'index': 0,
                'name': 'RA',
                'description': 'Right Ascension - celestial longitude coordinate',
                'units': 'degrees',
                'range': [0, 360]
            },
            {
                'index': 1,
                'name': 'Dec',
                'description': 'Declination - celestial latitude coordinate',
                'units': 'degrees',
                'range': [-90, 90]
            },
            {
                'index': 2,
                'name': 'Roll',
                'description': 'Roll angle - rotation around the pointing axis',
                'units': 'degrees',
                'range': [0, 360]
            }
        ]
    },
    'FOV': {
        'description': 'Field of View - angular extent of the observable world',
        'units': 'degrees',
        'type': 'float',
        'required': True,
        'range': [0.1, 180.0]
    },
    'RMSE': {
        'description': 'Root Mean Square Error - overall quality metric of the astrometric solution',
        'units': 'arcseconds',
        'type': 'float',
        'required': True,
        'range': [0, 1000]
    },
    'RMS': {
        'description': 'Per-axis Root Mean Square errors for each orientation component',
        'units': 'arcseconds',
        'type': 'array[3]',
        'required': True,
        'elements': [
            {
                'index': 0,
                'name': 'RA_RMS',
                'description': 'Right Ascension axis RMS error',
                'units': 'arcseconds',
                'range': [0, 1000]
            },
            {
                'index': 1,
                'name': 'Dec_RMS',
                'description': 'Declination axis RMS error',
                'units': 'arcseconds',
                'range': [0, 1000]
            },
            {
                'index': 2,
                'name': 'Roll_RMS',
                'description': 'Roll axis RMS error',
                'units': 'arcseconds',
                'range': [0, 1000]
            }
        ]
    },
    'sources': {
        'description': 'Number of star sources detected in the image',
        'units': 'count',
        'type': 'integer',
        'required': True,
        'range': [0, 1000]
    },
    'matched_stars': {
        'description': 'Number of stars successfully matched to reference catalog',
        'units': 'count',
        'type': 'integer',
        'required': True,
        'range': [0, 1000]
    },
    'probability': {
        'description': 'Probability estimate of the solution being correct',
        'units': 'none (dimensionless probability 0-1)',
        'type': 'float',
        'required': True,
        'range': [0, 1]
    },
    'angular_velocity': {
        'description': 'Angular velocity components in the local spacecraft frame',
        'units': 'degrees per second',
        'type': 'array[3]',
        'required': False,
        'elements': [
            {
                'index': 0,
                'name': 'roll_rate',
                'description': 'Angular velocity around the roll (pointing) axis',
                'units': 'deg/s',
                'range': [-100, 100]
            },
            {
                'index': 1,
                'name': 'az_rate',
                'description': 'Angular velocity around the azimuth (right) axis',
                'units': 'deg/s',
                'range': [-100, 100]
            },
            {
                'index': 2,
                'name': 'el_rate',
                'description': 'Angular velocity around the elevation (up) axis',
                'units': 'deg/s',
                'range': [-100, 100]
            }
        ]
    }
}

# Additional metadata for the overall telemetry structure
telemetry_structure_meta = {
    'position': {
        'description': 'Main telemetry container for astrometric position data',
        'type': 'object',
        'fields': {
            'timestamp': {
                'description': 'Timestamp of the telemetry packet creation',
                'units': 'ISO 8601 datetime string',
                'type': 'string'
            },
            'size': {
                'description': 'Number of position records in the data array',
                'units': 'count',
                'type': 'integer'
            },
            'data': {
                'description': 'Array of position measurement records',
                'type': 'array',
                'items': position_meta
            }
        }
    },
    'metadata': {
        'description': 'Schema metadata describing the telemetry format',
        'type': 'object',
        'fields': {
            'position': {
                'description': 'Metadata for the position data structure',
                'type': 'object'
            }
        }
    }
}


# Utility function to get field description
def get_field_description(field_name):
    """
    Get the description for a specific field in the position metadata.

    Args:
        field_name (str): Name of the field to look up

    Returns:
        str: Field description or None if not found
    """
    return position_meta.get(field_name, {}).get('description')


# Utility function to get field units
def get_field_units(field_name):
    """
    Get the units for a specific field in the position metadata.

    Args:
        field_name (str): Name of the field to look up

    Returns:
        str: Field units or None if not found
    """
    return position_meta.get(field_name, {}).get('units')