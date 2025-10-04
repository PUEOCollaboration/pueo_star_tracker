"""
Position/Telemetry/Filesystem metadata module + builder.

Preserves the original position_meta you supplied (no semantic changes),
and refines telemetry & filesystem metadata. Provides a builder helper that
creates the server JSON exactly like your sample, and only adds the 'metadata'
keys when requested (from CLI or API).
"""

from typing import Any, Dict, List, Optional
import time
import copy

# -------------------------
# position_meta
# -------------------------
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

# -------------------------
# telemetry_meta
# dynamic headers are documented explicitly
# -------------------------
telemetry_meta = {
    'description': 'Metadata describing telemetry container structure. Telemetry records carry dynamic headers + parallel data lists.',
    'fields': {
        'timestamp': {
            'description': 'Container-level timestamp (ISO8601)',
            'type': 'string'
        },
        'size': {
            'description': 'Number of telemetry records in data array',
            'type': 'integer'
        },
        'data_record': {
            'description': 'A telemetry row: headers[] and data[] are parallel arrays; headers determine semantics',
            'type': 'object',
            'fields': {
                'timestamp': {
                    'description': 'Row capture time (string)',
                    'type': 'string'
                },
                'headers': {
                    'description': 'Array of header names (dynamic depending on attached sensors and system probes)',
                    'type': 'array[string]'
                },
                'data': {
                    'description': 'Array of string values matching headers by index. Parsing rules are external.',
                    'type': 'array[string]'
                }
            },
            'notes': 'Header "S1..SN" generally represent Arduino or custom sensor channels; system sensors use descriptive names (coretemp_, drivetemp_, coreX_load, etc.)'
        }
    }
}

# -------------------------
# filesystem_meta
# -------------------------
filesystem_meta = {
    'description': 'Metadata for filesystem container (root and device entries like ssd, sd_card)',
    'fields': {
        'timestamp': {
            'description': 'Container-level timestamp (ISO8601)',
            'type': 'string'
        },
        'size': {
            'description': 'Number of filesystem records in data array',
            'type': 'integer'
        },
        'data_record': {
            'description': 'Device-level object (status, levels, current, trend, forecast)',
            'type': 'object',
            'fields': {
                'name': {'description': 'logical name of device (root, ssd, sd_card)', 'type': 'string'},
                'status': {'description': 'health string: normal|warning|critical', 'type': 'string'},
                'is_critical': {'description': 'boolean flag', 'type': 'boolean'},
                'levels': {'description': 'threshold percents', 'type': 'object'},
                'current': {'description': 'current snapshot object (ts, folders, files, size_mb, used_mb, total_mb, used_pct, free_pct)', 'type': 'object'},
                'trend': {'description': 'trend windows (1h,6h,24h,week,month,all)', 'type': 'object'},
                'forecast': {'description': 'estimated time until warning/critical or null', 'type': 'object'}
            }
        }
    }
}

# -------------------------
# Helper to optionally include metadata keys in containers
# -------------------------
def maybe_meta(meta_obj: Dict[str, Any], include: bool) -> Optional[Dict[str, Any]]:
    """
    Return a deep copy of meta_obj if include==True, otherwise return None.
    The builder will only insert the metadata key if this returns a dict.
    """
    if include:
        return copy.deepcopy(meta_obj)
    return None


# -------------------------
# Main builder: constructs the JSON exactly like your server code intended
# - only includes 'metadata' keys when include_metadata == True
# - expects "position_elements" and "telemetry_elements" already assembled lists
# -------------------------
def build_flight_telemetry(
    mode: str,
    position_elements: List[Dict[str, Any]],
    telemetry_elements: List[Dict[str, Any]],
    filesystem_entries: List[Dict[str, Any]],
    ts_iso: Optional[str] = None,
    include_metadata: bool = False
) -> Dict[str, Any]:
    """
    Build flight telemetry top-level payload.

    Args:
        mode: e.g. 'flight'
        position_elements: list of position records (each matches position_meta)
        telemetry_elements: list of telemetry records (each contains 'headers' and 'data')
        filesystem_entries: list (usually one) of filesystem device objects (monitor.status())
        ts_iso: optional ISO8601 timestamp for container-level timestamp; if None uses current time
        include_metadata: if True, attach metadata dicts; otherwise omit metadata keys

    Returns:
        dict shaped like your example server response (no empty metadata when include_metadata=False)
    """
    if ts_iso is None:
        ts_iso = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())

    # container 'position'
    position_container = {
        'timestamp': ts_iso,
        'size': len(position_elements),
        'data': position_elements
    }
    # only add 'metadata' key if requested
    if include_metadata:
        position_container['metadata'] = maybe_meta(position_meta, True)

    # container 'telemetry'
    telemetry_container = {
        'timestamp': ts_iso,
        'size': len(telemetry_elements),
        'data': telemetry_elements
    }
    if include_metadata:
        telemetry_container['metadata'] = maybe_meta(telemetry_meta, True)

    # container 'filesystem'
    filesystem_container = {
        'timestamp': ts_iso,
        'size': len(filesystem_entries),
        'data': filesystem_entries
    }
    if include_metadata:
        filesystem_container['metadata'] = maybe_meta(filesystem_meta, True)

    payload = {
        'mode': mode,
        'position': position_container,
        'telemetry': telemetry_container,
        'filesystem': filesystem_container
    }
    return payload


# -------------------------
# Example CLI-style usage (how to wire into your pueo_cli.py)
# -------------------------
# Usage: pueo_cli.py get_flight_telemetry [limit] [metadata]
# where `metadata` is a boolean flag (True/False)
#
# Example construction (pseudo-code used inside server handler):
#
#   ts = datetime.utcnow().isoformat()
#   position_elements = [...]       # assembled by your solver loop
#   telemetry_elements = [...]      # assembled by sensors (headers + data lists)
#   filesystem_entries = [monitor.status()]
#
#   payload = build_flight_telemetry(
#       mode=server.flight_mode,
#       position_elements=position_elements,
#       telemetry_elements=telemetry_elements,
#       filesystem_entries=filesystem_entries,
#       ts_iso=ts,
#       include_metadata=args.metadata
#   )
#
# This ensures metadata keys are only present in the JSON when the CLI requested them.
# -------------------------
