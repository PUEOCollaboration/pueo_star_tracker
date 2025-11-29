import time
import glob
import numpy as np
from astropy.table import Table
from astropy.io import fits
import subprocess
import shutil
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union, Any
import logging
import re
import os
from contextlib import suppress
import json

class AstrometryNet:
    """
    A class to handle conversion of centroid data to FITS format and
    interface with astrometry.net's solve-field command.

    Attributes:
        cfg: Configuration object with an_ prefixed properties
        log: Logger instance for tracking operations

    Returns Example:
    {
        'solved': True,
        'RA': 230.559089,
        'Dec': 26.655801,
        'Roll': 113.112,
        'FOV': 9.107335,
        'RMSE': 2.5,  # RMS error in arcseconds
        'Matches': 6,
        'pixel_scale_arcsec': 9.40814,
        'index_used': 'index-4115.fits',

        # Star data
        'matched_centroids': [[x1, y1], [x2, y2], ...],  # Image coordinates of matched stars
        'matched_stars': [[ra1, dec1, None, None], ...],  # Sky coordinates of matched stars
        'reference_stars': [[ra1, dec1], [ra2, dec2], ...],  # All catalog stars in field
        'detected_stars': [[x1, y1], [x2, y2], ...],  # All stars detected in image
        'detected_stars_count': 25,

        # Complete WCS transformation
        'wcs_data': {
            'CRPIX1': 2072.0, 'CRPIX2': 1411.0,
            'CRVAL1': 230.559089, 'CRVAL2': 26.655801,
            'CD1_1': -0.002612, 'CD1_2': -0.000123,
            'CD2_1': 0.000145, 'CD2_2': -0.002611,
            'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN'
        },

        # File paths
        'output_files': {
            'solved': '/path/to/file.solved',
            'wcs': '/path/to/file.wcs',
            'corr': '/path/to/file.corr',
            'match': '/path/to/file.match',
            'rdls': '/path/to/file.rdls',
            'axy': '/path/to/file.axy',
            'indx_xyls': '/path/to/file-indx.xyls'
        }
    }
    """

    def __init__(self, config: Optional[Any] = None, log=None):
        """
        Initialize the AstrometryNet processor with configuration.

        Args:
            config: Configuration object with an_ prefixed properties
        """
        self.cfg = config
        self.log = log

    def _get_config_value(self, param_name: str, default: Any = None) -> Any:
        """
        Get configuration value with an_ prefix, handling various naming patterns.

        Args:
            param_name: Base parameter name (e.g., 'scale_units')
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        if self.cfg is None:
            return default

        # Try different naming patterns
        # We actually have all naming as an_*** 1 to 1 MATCHING!
        patterns = [
            f"an_{param_name}",  # an_scale_units
            # f"an_{param_name.replace('_', '')}",  # an_scaleunits
            # param_name,  # scale_units (direct)
        ]

        for pattern in patterns:
            if hasattr(self.cfg, pattern):
                value = getattr(self.cfg, pattern)
                if value is not None:
                    # Log only used params and their config.ini value.
                    self.log.debug(f'param: {param_name}: patterns: {patterns} value: {value}')
                    return value
        # Do not log skipped params
        # self.log.debug(f'param: {param_name}: patterns: {patterns} value: None~ default')
        return default

    def _build_param_mapping(self) -> Dict[str, str]:
        """
        Create mapping from solve-field parameters to config property names.
        """
        return {
            'scale-units': 'scale_units',
            'scale-low': 'scale_low',
            'scale-high': 'scale_high',
            'downsample': 'downsample',
            'overwrite': 'overwrite',
            'no-plots': 'no_plots',
            'corr': 'corr',
            'new-fits': 'new_fits',
            'solved': 'solved',
            'cpulimit': 'cpulimit',
            'radius': 'radius',
            'depth': 'depth',
            'parity': 'parity',
            'tweak-order': 'tweak_order',
            'crpix-center': 'crpix_center',
            'no-verify': 'no_verify',
            'no-background-subtraction': 'no_background_subtraction',
            'uniformize': 'uniformize',
            'no-remove-lines': 'no_remove_lines',
            'keep-xylist': 'keep_xylist',
            'width': 'width',
            'height': 'height',
            'x': 'x',
            'y': 'y',
            'ra': 'ra',
            'dec': 'dec',
            'sort-column': 'sort_column',
            'sort-ascending': 'sort_ascending',
            'no-fits2fits': 'no_fits2fits',
            'dir': 'dir',
            'temp-dir': 'temp_dir',
            'config': 'config',
            'index': 'index',
            'cancel': 'cancel',
            'continue': 'continue',
            'no-tweak': 'no_tweak',
            'resort': 'resort',
            'depth-method': 'depth_method',
            'objs': 'objs',
            'invert': 'invert',
            'd2': 'd2',
            'sigma': 'sigma',
            'code-tol': 'code_tol',
            'pixel-error': 'pixel_error',
            'quad-size-min': 'quad_size_min',
            'quad-size-max': 'quad_size_max',
            'odds-to-solve': 'odds_to_solve',
            'odds-to-tune': 'odds_to_tune',
            'tune-up': 'tune_up',
            'wcs': 'wcs',
            'match': 'match',
            'rdls': 'rdls',
            'axy': 'axy',
            'kmz': 'kmz',
            'temp-axy': 'temp_axy',
            'backend-config': 'backend_config',
            'no-plot': 'no_plot',
            'plot-scale': 'plot_scale',
            'stretch': 'stretch',
            'grayplot': 'grayplot',
            'image': 'image',
            'fits-image': 'fits_image',
            # 'port': 'port',
            # 'ip': 'ip'
        }

    def ndarray2fits(self, centroids: np.ndarray, filename: str) -> str:
        """
        Convert a numpy ndarray of centroids to FITS format for astrometry.net.

        Args:
            centroids: NumPy array with columns [y, x, flux, std, fwhm]
            filename: Output FITS filename

        Returns:
            Path to the created FITS file

        Raises:
            ValueError: If centroids array doesn't have expected structure
        """
        if centroids.ndim != 2 or centroids.shape[1] < 2:
            raise ValueError("Centroids array must be 2D with at least 2 columns (x, y)")

        # Create structured array for astropy Table
        dtype = [('x', 'f8'), ('y', 'f8')]
        data = np.empty(centroids.shape[0], dtype=dtype)

        # Assuming centroids columns: [y, x, flux, std, fwhm]
        # astrometry.net expects x, y coordinates
        data['x'] = centroids[:, 1]  # x coordinate from second column
        data['y'] = centroids[:, 0]  # y coordinate from first column

        # Create astropy table
        table = Table(data)

        # Add metadata for astrometry.net compatibility
        table.meta['EXTNAME'] = 'XYLS'
        table.meta['CTYPE1'] = 'X'
        table.meta['CTYPE2'] = 'Y'

        # Save to FITS
        output_path = Path(filename)
        table.write(str(output_path), format='fits', overwrite=True)

        self.log.info(f"Created FITS file with {len(table)} sources: {output_path}")
        return str(output_path)

    def build_solve_field_command(self, fits_filename: str,
                                  image_size: Tuple[int, int] = None,  # (height, width)
                                  output_dir: Optional[str] = None) -> List[str]:
        """
        Build the solve-field command with configured parameters.

        Args:
            fits_filename: Input FITS file to solve
            image_size (Tuple[int, int], optional): Image size as a tuple of (height, width).
            output_dir: Directory for output files (default: same as input)

        Returns:
            List of command arguments for subprocess
        """
        cmd = ["solve-field"]
        param_mapping = self._build_param_mapping()

        # Add configured parameters
        for solve_param, config_param in param_mapping.items():
            value = self._get_config_value(config_param)

            if value is None:
                continue

            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{solve_param}")
            else:
                cmd.append(f"--{solve_param}")
                cmd.append(str(value))

        # Add image size
        if image_size:
            height, width = image_size
            cmd.extend(["--width", str(width), "--height", str(height)])

        # Add output directory if specified
        if output_dir:
            cmd.extend(["--dir", output_dir])
        else:
            # Check if directory is configured
            dir_value = self._get_config_value('dir')
            if dir_value:
                cmd.extend(["--dir", str(dir_value)])

        # Add the input file
        cmd.append(fits_filename)

        return cmd

    def solve_field(self, fits_filename: str,
                    image_size: Tuple[int, int] = None,  # (height, width)
                    output_dir: Optional[str] = None,
                    timeout: int = 300) -> Dict:
        """
        Execute solve-field command and capture results.

        Args:
            fits_filename: Input FITS file to solve
            image_size (Tuple[int, int], optional): Image size as a tuple of (height, width).
            output_dir: Directory for output files
            timeout: Maximum execution time in seconds

        Returns:
            Dictionary containing process results and output files

        Raises:
            FileNotFoundError: If solve-field command is not found
            subprocess.TimeoutExpired: If process exceeds timeout
            subprocess.CalledProcessError: If process returns non-zero exit code
        """
        # Check if solve-field is available
        if not shutil.which("solve-field"):
            self.log.error('solve-field not available')
            raise FileNotFoundError("solve-field command not found. Install astrometry.net first.")

        self.log.info('solve-field command available')

        # Build command
        cmd = self.build_solve_field_command(fits_filename, image_size, output_dir)

        self.log.info(f"Executing: {' '.join(cmd)}")

        # Execute command
        try:
            t0 = time.monotonic()
            env = os.environ.copy()
            # Modify env if necessary, e.g., env["PATH"] = "/custom/path:" + env["PATH"]
            cwd = os.getcwd()
            self.log.debug(f'  working dir: {cwd}')
            # self.log.debug(f'  env: {env}')
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,  # Raise exception if a non-zero exit code
                env=env
            )
            solve_time = time.monotonic() - t0

            # Get base filename for output files
            base_filename = os.path.splitext(fits_filename)[0]
            if output_dir:
                base_filename = os.path.join(output_dir, os.path.basename(base_filename))

            # Parse the complete solution
            parser = AstrometryNetParser(self.log)
            parsed_data = parser.parse_solution(result.stdout, result.stderr, base_filename)
            parsed_data['T_solve'] = solve_time

            # Check what files were successfully parsed
            if 'detected_stars' in parsed_data:
                self.log.info(f"Detected {parsed_data['detected_stars_count']} stars in image")
            if 'reference_stars' in parsed_data:
                with suppress(TypeError):
                    self.log.info(f"Found {len(parsed_data.get('reference_stars', []))} reference stars in catalog")
            if 'matched_centroids' in parsed_data:
                self.log.info(f"Successfully matched {parsed_data['Matches']} stars")

            # Build final output
            output = parsed_data
            output |= {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': parsed_data['solved'],  # Use .solved file existence as success
                'command': ' '.join(cmd),
            }

            self.log.info(f"solve-field completed with return code {result.returncode}")
            if result.stdout:
                self.log.debug(f"stdout: {result.stdout[:500]}...")  # First 500 chars
            if result.stderr:
                self.log.warning(f"stderr: {result.stderr[:500]}...")

            if parsed_data['solved']:
                self.log.info(f"Solution: RA={parsed_data['RA']}, Dec={parsed_data['Dec']}, "
                              f"Matches={parsed_data['Matches']}")

            self.log.info(f"astrometry.net completed: solved={parsed_data['solved']} in {solve_time}s.")
            return output

        except subprocess.TimeoutExpired:
            self.log.error(f"solve-field timed out after {timeout} seconds")
            raise
        except subprocess.CalledProcessError as e:
            self.log.error(f"solve-field failed with return code {e.returncode}")
            # Even if process failed, check if we got a partial solution
            try:
                base_filename = os.path.splitext(fits_filename)[0]
                if output_dir:
                    base_filename = os.path.join(output_dir, os.path.basename(base_filename))

                parser = AstrometryNetParser(self.log)
                parsed_data = parser.parse_solution(e.stdout, e.stderr, base_filename)
                if parsed_data['solved']:
                    self.log.warning("Process failed but .solved file exists - partial success")
            except:
                pass
            raise

    def process_centroids(self, centroids: np.ndarray,
                          image_size: Tuple[int, int] = None,  # (height, width)
                          output_base: str = "centroids",
                          output_dir: str = ".",
                          pre_cleanup: bool = True,
                          cleanup: bool = False) -> Dict:
        """
        Complete pipeline: convert centroids to FITS and solve field.

        Args:
            centroids: NumPy array with centroid data
            image_size (Tuple[int, int], optional): Image size as a tuple of (height, width).
            output_base: Base name for output files
            output_dir: Directory for output files
            pre_cleanup: Whether to remove files at initialisation
            cleanup: Whether to remove intermediate files

        Returns:
            Dictionary with processing results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Delete any existing files with the output_base pattern
        if pre_cleanup:
            pattern = str(output_path / f"{output_base}*")
            for file_path in glob.glob(pattern):
                try:
                    os.remove(file_path)
                except OSError as e:
                    self.log.warning(f"Warning: Could not delete {file_path}: {e}")

        # Convert to FITS
        fits_filename = str(output_path / f"{output_base}.xyls.fits")
        self.ndarray2fits(centroids, fits_filename)

        # Solve field
        result = self.solve_field(fits_filename, image_size, output_dir=str(output_path))

        # Check for solution files
        solution_files = {}
        for ext in ['.solved', '.wcs', '.rdls', '.axy', '.corr']:
            candidate = output_path / f"{output_base}.xyls{ext}"
            if candidate.exists():
                solution_files[ext[1:]] = str(candidate)

        result['solution_files'] = solution_files

        # Cleanup intermediate files if requested
        if cleanup:
            intermediate_files = [
                output_path / f"{output_base}.xyls.axy",
                output_path / f"{output_base}.xyls.corr",
                output_path / f"{output_base}.xyls.match",
                output_path / f"{output_base}.xyls.temp"
            ]
            for file in intermediate_files:
                if file.exists():
                    file.unlink()

        filename =  output_path / f"{output_base}.pueo.json"
        # Save the dictionary in a readable JSON format
        with open(filename, 'w') as f:
            json.dump(result, f, indent=4, sort_keys=True)

        return result


import os
import re
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from astropy.io import fits
import logging


class AstrometryNetParser:
    """Complete parser for all astrometry.net output files with FITS support"""

    def __init__(self, log=None):
        """Initialize parser with optional logger"""
        self.log = log
        self.log.debug('Initialised AstrometryNetParser.')

    def parse_solution(self, stdout: str, stderr: str, base_filename: str) -> Dict[str, Any]:
        """
        Parse all astrometry.net output files into comprehensive solution.

        Args:
            stdout: Standard output from solve-field command
            stderr: Standard error from solve-field command
            base_filename: Base path of the input file (without extension)

        Returns:
            Dictionary with complete astrometry solution including all file data
        """
        result = {
            'RA': None,
            'Dec': None,
            'Roll': None,
            'FOV': None,
            'RMSE': None,
            'Matches': None,
            'Prob': None,
            'T_solve': None,
            # 'solver': 'astrometry.net',
            'solved': False,
            'wcs_data': None,
            'matched_centroids': None,
            'matched_stars': None,
            'reference_stars': None,
            'detected_stars': None,
            'raw_output': stdout,
            'error_output': stderr,
            'output_files': {},
            'index_used': None,
            'pixel_scale_arcsec': None,
            'FOV_width': None,
            'FOV_height': None
        }

        # Check all available files
        files_to_check = {
            'solved': base_filename + '.solved',
            'wcs': base_filename + '.wcs',
            'corr': base_filename + '.corr',
            'match': base_filename + '.match',
            'rdls': base_filename + '.rdls',
            'axy': base_filename + '.axy',
            'indx_xyls': base_filename + '-indx.xyls'
        }

        # Convert base_path string to Path and list all files
        base_path_obj = Path(base_filename)
        all_files = list(base_path_obj.glob('*'))
        self.log.debug(f"All files in {base_filename}: \n {[f.name for f in all_files]}")

        # Record which files exist
        for file_type, file_path in files_to_check.items():
            # Convert relative path to absolute path
            file_path = os.path.abspath(file_path)

            if os.path.exists(file_path):
                result['output_files'][file_type] = file_path
                self.log.debug(f'Adding file: {file_type:>10} {file_path}')
                if file_type == 'solved':
                    result['solved'] = True
            else:
                self.log.warning(f'Skipping file: {file_type:>10} {file_path} - does not exist.')

        # Parse console output first
        self._parse_console_output(stdout, result)

        # Parse files in order of importance
        if 'wcs' in result['output_files']:
            self._parse_wcs_file(result['output_files']['wcs'], result)

        if 'corr' in result['output_files']:
            self._parse_corr_file(result['output_files']['corr'], result)

        if 'match' in result['output_files']:
            self._parse_match_file(result['output_files']['match'], result)

        if 'rdls' in result['output_files']:
            self._parse_rdls_file(result['output_files']['rdls'], result)

        if 'axy' in result['output_files']:
            self._parse_axy_file(result['output_files']['axy'], result)

        # NEW: Parse index reference stars
        if 'indx_xyls' in result['output_files']:
            self._parse_indx_xyls_file(result['output_files']['indx_xyls'], result)

        # Parse index information from console
        self._parse_index_info(stdout, result)

        return result

    def _parse_console_output(self, stdout: str, result: Dict[str, Any]):
        """Parse the console output text"""
        # Parse log-odds ratio and calculate probability
        """Parse console output for astrometry solution details"""
        # Parse log-odds ratio and calculate FALSE POSITIVE probability
        logodds_match = re.search(r'log-odds ratio ([\d.]+) \(([\d.]+e?[+-]?\d+)\)', stdout)
        if logodds_match:
            logodds = float(logodds_match.group(1))
            odds = float(logodds_match.group(2))
            result['logodds'] = logodds
            result['odds'] = odds

            # Convert to probability of FALSE POSITIVE (what you want)
            # P(false_positive) = 1 / (1 + odds) = 1 / (1 + exp(logodds))
            result['Prob'] = np.float64(1.0 / (1.0 + odds))

            self.log.info(f"Found log-odds: {logodds}, odds: {odds}, False Positive Prob: {result['Prob']}")

        # Parse field center (RA, Dec)
        ra_dec_match = re.search(r'Field center: \(RA,Dec\) = \(([\d.]+), ([\d.]+)\) deg', stdout)
        if ra_dec_match:
            result['RA'] = np.float64(ra_dec_match.group(1))
            result['Dec'] = np.float64(ra_dec_match.group(2))

        # Parse field size
        fov_match = re.search(r'Field size: ([\d.]+) x ([\d.]+) degrees', stdout)
        if fov_match:
            fov1 = float(fov_match.group(1))
            fov2 = float(fov_match.group(2))
            result['FOV'] = np.float64(fov1) # Returning the HORIZONTAL
            result['h_FOV'] = np.float64(fov1)
            result['v_FOV'] = np.float64(fov2)

        # Parse rotation angle
        rotation_match = re.search(r'Field rotation angle: up is ([\d.]+) degrees E of N', stdout)
        if rotation_match:
            result['Roll'] = np.float64(rotation_match.group(1))

        # Parse matches
        matches_match = re.search(r'log-odds ratio [\d.]+ \([\d.e+]+\), ([\d]+) match', stdout)
        if matches_match:
            result['Matches'] = int(matches_match.group(1))

        # Parse pixel scale
        pixel_scale_match = re.search(r'pixel scale ([\d.]+) arcsec/pix', stdout)
        if pixel_scale_match:
            result['pixel_scale_arcsec'] = np.float64(pixel_scale_match.group(1))

        # Parse parity
        parity_match = re.search(r'Field parity: (\w+)', stdout)
        if parity_match:
            result['parity'] = parity_match.group(1)

    def _parse_index_info(self, stdout: str, result: Dict[str, Any]):
        """Parse which index was used from console output"""
        index_match = re.search(r'solved with index (index-\d+\.fits)', stdout)
        if index_match:
            result['index_used'] = index_match.group(1)

    def _parse_wcs_file_old(self, wcs_file: str, result: Dict[str, Any]):
        """Parse the WCS file for precise astrometric solution"""
        try:
            with fits.open(wcs_file) as hdul:
                header = hdul[0].header

                # Extract WCS parameters
                wcs_data = {
                    'CRPIX1': header.get('CRPIX1'),  # Reference pixel X
                    'CRPIX2': header.get('CRPIX2'),  # Reference pixel Y
                    'CRVAL1': header.get('CRVAL1'),  # RA at reference pixel
                    'CRVAL2': header.get('CRVAL2'),  # Dec at reference pixel
                    'CD1_1': header.get('CD1_1'),  # Transformation matrix
                    'CD1_2': header.get('CD1_2'),
                    'CD2_1': header.get('CD2_1'),
                    'CD2_2': header.get('CD2_2'),
                    'CTYPE1': header.get('CTYPE1'),  # Coordinate type
                    'CTYPE2': header.get('CTYPE2'),
                    'EQUINOX': header.get('EQUINOX'),  # Equinox
                    'RADESYS': header.get('RADESYS')  # Reference system
                }

                result['wcs_data'] = wcs_data

                # Update RA/Dec from WCS if more precise
                if wcs_data['CRVAL1'] is not None:
                    result['RA'] = np.float64(wcs_data['CRVAL1'])
                if wcs_data['CRVAL2'] is not None:
                    result['Dec'] = np.float64(wcs_data['CRVAL2'])
            self.log.info(f"Parsed WCS file {wcs_file}.")
        except Exception as e:
            self.log.warning(f"Failed to parse WCS file {wcs_file}: {e}")

    def _parse_wcs_file(self, wcs_file: str, result: Dict[str, Any]):
        """Parse the WCS file for precise astrometric solution with complete metadata"""
        try:
            with fits.open(wcs_file) as hdul:
                header = hdul[0].header

                # Extract basic WCS parameters
                wcs_data = {
                    'CRPIX1': header.get('CRPIX1'),  # Reference pixel X
                    'CRPIX2': header.get('CRPIX2'),  # Reference pixel Y
                    'CRVAL1': header.get('CRVAL1'),  # RA at reference pixel
                    'CRVAL2': header.get('CRVAL2'),  # Dec at reference pixel
                    'CD1_1': header.get('CD1_1'),  # Transformation matrix
                    'CD1_2': header.get('CD1_2'),
                    'CD2_1': header.get('CD2_1'),
                    'CD2_2': header.get('CD2_2'),
                    'CTYPE1': header.get('CTYPE1'),  # Coordinate type
                    'CTYPE2': header.get('CTYPE2'),
                    'EQUINOX': header.get('EQUINOX'),  # Equinox
                    'RADESYS': header.get('RADESYS'),  # Reference system
                    'CUNIT1': header.get('CUNIT1'),  # Units
                    'CUNIT2': header.get('CUNIT2'),
                    'LONPOLE': header.get('LONPOLE'),  # Pole coordinates
                    'LATPOLE': header.get('LATPOLE')
                }

                # Calculate Roll from CD matrix - CORRECTED
                if all(key in wcs_data for key in ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']):
                    cd1_1, cd1_2, cd2_1, cd2_2 = wcs_data['CD1_1'], wcs_data['CD1_2'], wcs_data['CD2_1'], wcs_data[
                        'CD2_2']

                    # Calculate the position angle (up direction)
                    # This matches what astrometry.net reports
                    roll = np.degrees(np.arctan2(-cd1_2, -cd2_2))


                    # Alternative: calculate from both CD matrix components
                    # roll = np.degrees(np.arctan2(cd2_1, cd2_2))

                    if 'Roll' in result and result['Roll'] is None:
                        result['Roll'] = np.float64(roll)
                        wcs_data['ROLL'] = roll

                # Calculate pixel scale
                if all(key in wcs_data for key in ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']):
                    pixel_scale_x = np.sqrt(cd1_1 ** 2 + cd2_1 ** 2) * 3600  # arcsec/pixel
                    pixel_scale_y = np.sqrt(cd1_2 ** 2 + cd2_2 ** 2) * 3600  # arcsec/pixel
                    result['pixel_scale_arcsec'] = np.float64((pixel_scale_x + pixel_scale_y) / 2)
                    wcs_data['PIXSCALE'] = result['pixel_scale_arcsec']

                # Calculate FOV from image dimensions and pixel scale
                if 'IMAGEW' in header and 'IMAGEH' in header and 'pixel_scale_arcsec' in result:
                    width_pixels = header['IMAGEW']
                    height_pixels = header['IMAGEH']

                    # Calculate width and height in degrees
                    fov_width_deg = np.float64(width_pixels * result['pixel_scale_arcsec'] / 3600)
                    fov_height_deg = np.float64(height_pixels * result['pixel_scale_arcsec'] / 3600)

                    # Calculate DIAGONAL FOV (like tetra3 and your camera specification)
                    fov_diagonal_deg = np.sqrt(fov_width_deg ** 2 + fov_height_deg ** 2)

                    result['FOV_width'] = fov_width_deg
                    result['FOV_height'] = fov_height_deg
                    result['FOV'] = fov_diagonal_deg  # Diagonal FOV in degrees

                    wcs_data['FOV_WIDTH'] = fov_width_deg
                    wcs_data['FOV_HEIGHT'] = fov_height_deg
                    wcs_data['FOV_DIAGONAL'] = fov_diagonal_deg

                # Extract SIP distortion coefficients
                sip_data = {}
                for key in header:
                    if key.startswith(('A_', 'B_', 'AP_', 'BP_')):
                        sip_data[key] = header[key]
                if sip_data:
                    wcs_data['SIP_COEFFS'] = sip_data

                # Extract astrometry.net specific metadata
                astrometry_meta = {}
                history_lines = []
                comment_lines = []

                # Parse HISTORY and COMMENT cards
                for card in header.cards:
                    if card.keyword == 'HISTORY':
                        history_lines.append(str(card.value))
                    elif card.keyword == 'COMMENT':
                        comment_lines.append(str(card.value))

                # Extract index files used
                index_files = []
                for line in history_lines + comment_lines:
                    if 'index-' in line and '.fits' in line:
                        index_files.append(line.strip())

                if index_files:
                    wcs_data['INDEX_FILES'] = index_files
                    # Get the actual index used (usually the last one mentioned)
                    result['index_used'] = index_files[-1].split('/')[-1] if index_files else None

                # Extract solution quality metrics from comments
                for line in comment_lines:
                    if 'log odds:' in line:
                        try:
                            result['logodds'] = np.float64(line.split('log odds:')[-1].strip())
                        except:
                            pass
                    elif 'code error:' in line:
                        try:
                            result['RMSE'] = np.float64(line.split('code error:')[-1].strip())
                        except:
                            pass
                    elif 'nmatch:' in line:
                        try:
                            result['Matches'] = int(line.split('nmatch:')[-1].strip())
                        except:
                            pass
                    elif 'scale:' in line and 'arcsec/pix' in line:
                        try:
                            scale_str = line.split('scale:')[-1].split('arcsec/pix')[0].strip()
                            result['pixel_scale_arcsec'] = np.float64(scale_str)
                        except:
                            pass

                result['wcs_data'] = wcs_data

                # Update RA/Dec from WCS if available
                if wcs_data['CRVAL1'] is not None:
                    result['RA'] = np.float64(wcs_data['CRVAL1'])
                if wcs_data['CRVAL2'] is not None:
                    result['Dec'] = np.float64(wcs_data['CRVAL2'])

                self.log.info(f"Parsed WCS file {wcs_file} with roll: {result.get('Roll', 'unknown')}°")

        except Exception as e:
            self.log.warning(f"Failed to parse WCS file {wcs_file}: {e}")

    def _parse_axy_file(self, axy_file: str, result: Dict[str, Any]):
        """Parse the detected stars file (could be FITS or text format)"""
        try:
            # First try to read as FITS file
            try:
                self._parse_axy_fits(axy_file, result)
                self.log.info(f"Parsed AXY fits file {axy_file}.")
                return
            except (OSError, Exception) as fits_error:
                # If FITS reading fails, try as text file
                self.log.debug(f"AXY file not in FITS format, trying text: {fits_error}")
                self._parse_axy_text(axy_file, result)
                return

        except Exception as e:
            self.log.warning(f"Failed to parse AXY file {axy_file}: {e}")

    def _parse_axy_fits(self, axy_file: str, result: Dict[str, Any]):
        """Parse AXY file in FITS format"""
        try:
            with fits.open(axy_file) as hdul:
                # AXY files typically have the star coordinates in the first extension
                if len(hdul) > 1:
                    data = hdul[1].data
                    if hasattr(data, 'field') and 'x' in data.names and 'y' in data.names:
                        # Structured array format
                        x_coords = data['x']
                        y_coords = data['y']
                        detected_stars = [[float(y), float(x)] for x, y in zip(x_coords, y_coords)]
                    else:
                        # Assume first two columns are X, Y coordinates
                        detected_stars = [[row[1], row[0]] for row in data]

                    result['detected_stars'] = detected_stars
                    result['detected_stars_count'] = len(detected_stars)
                    self.log.info(f"Parsed {len(detected_stars)} stars from FITS AXY file")

        except Exception as e:
            self.log.warning(f"Failed to parse AXY FITS file {axy_file}: {e}")
            raise

    def _parse_axy_text(self, axy_file: str, result: Dict[str, Any]):
        """Parse AXY file in text format"""
        try:
            detected_stars = []

            with open(axy_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            detected_stars.append([
                                float(parts[1]),  # x coordinate
                                float(parts[0])  # y coordinate
                            ])
                        except ValueError:
                            continue

            if detected_stars:
                result['detected_stars'] = detected_stars
                result['detected_stars_count'] = len(detected_stars)
                self.log.info(f"Parsed {len(detected_stars)} stars from text AXY file")

        except Exception as e:
            self.log.warning(f"Failed to parse AXY text file {axy_file}: {e}")
            raise

    def _parse_corr_file(self, corr_file: str, result: Dict[str, Any]):
        """Parse the correspondence file (usually text, but could be FITS)"""
        try:
            # Try text format first (most common for .corr files)
            try:
                self._parse_corr_text(corr_file, result)
                self.log.info(f"Parsed CORR file {corr_file}.")
                return
            except (UnicodeDecodeError, Exception):
                # Fall back to FITS format if text parsing fails
                self._parse_corr_fits(corr_file, result)
                self.log.info(f"Parsed CORR fits file {corr_file}.")
                return

        except Exception as e:
            self.log.warning(f"Failed to parse CORR file {corr_file}: {e}")

    def _parse_corr_text(self, corr_file: str, result: Dict[str, Any]):
        """Parse CORR file in text format"""
        try:
            matched_centroids = []  # Image coordinates
            matched_stars = []  # Sky coordinates (RA, Dec)

            with open(corr_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        matched_centroids.append([
                            float(parts[0]),  # x_image
                            float(parts[1])  # y_image
                        ])
                        matched_stars.append([
                            float(parts[4]),  # RA (degrees)
                            float(parts[5]),  # Dec (degrees)
                            None,  # Magnitude
                            None  # Star ID
                        ])

            if matched_centroids:
                result['matched_centroids'] = matched_centroids
                result['matched_stars'] = matched_stars
                result['Matches'] = len(matched_centroids)
                self.log.info(f"Parsed {len(matched_centroids)} matches from text CORR file")

        except Exception as e:
            self.log.warning(f"Failed to parse CORR text file {corr_file}: {e}")
            raise

    def _parse_corr_fits(self, corr_file: str, result: Dict[str, Any]):
        """Parse CORR file in FITS format"""
        try:
            with fits.open(corr_file) as hdul:
                if len(hdul) > 1:
                    data = hdul[1].data
                    matched_centroids = []
                    matched_stars = []

                    # Use the actual field names from your CORR file
                    if (hasattr(data, 'field') and
                            'field_x' in data.names and 'field_y' in data.names and
                            'field_ra' in data.names and 'field_dec' in data.names):

                        x_coords = data['field_x']
                        y_coords = data['field_y']
                        ras = data['field_ra']
                        decs = data['field_dec']

                        for x, y, ra, dec in zip(x_coords, y_coords, ras, decs):
                            matched_centroids.append([float(y), float(x)])  # Note  returning y, x for consistency with precomputed_centroids
                            matched_stars.append([float(ra), float(dec), None, None])

                    if matched_centroids:
                        result['matched_centroids'] = matched_centroids
                        result['matched_stars'] = matched_stars
                        result['Matches'] = len(matched_centroids)
                        self.log.info(f"Parsed {len(matched_centroids)} matches from FITS CORR file")
                    else:
                        self.log.warning(f"Parsed 0 matches from FITS CORR file - check field names")
                        self.log.debug(f"Available field names: {data.names}")
                else:
                    self.log.warning(f"Hdul data len: {len(hdul)}, expected > 1")
        except Exception as e:
            self.log.warning(f"Failed to parse CORR FITS file {corr_file}: {e}")
            raise

    def _parse_match_file(self, match_file: str, result: Dict[str, Any]):
        """Parse the match file (usually text, but could be FITS)"""
        try:
            # Try text format first (most common for .match files)
            try:
                self._parse_match_text(match_file, result)
                # Check if text parsing actually found anything
                if 'RMSE' not in result and 'Matches' not in result:
                    raise ValueError("Text parsing found no valid data - likely FITS format")

                self.log.info(f"Parsed MATCH file {match_file}.")
                return
            except (UnicodeDecodeError, ValueError, Exception) as e:
                # Fall back to FITS format if text parsing fails or finds nothing
                self.log.debug(f"Text parsing failed, trying FITS: {e}")
                self._parse_match_fits(match_file, result)
                self.log.info(f"Parsed MATCH fits file {match_file}.")
                return

        except Exception as e:
            self.log.warning(f"Failed to parse MATCH file {match_file}: {e}")

    def _parse_match_text(self, match_file: str, result: Dict[str, Any]):
        """Parse text format match file - raises exception if file is FITS format"""
        # Quick check for FITS file by reading first few bytes

        with open(match_file, 'rb') as f:
            magic = f.read(4)
            if magic == b'SIMP':  # FITS magic number
                raise UnicodeDecodeError("FITS file detected", "binary", 0, 4, "Not a text file")

        # If we get here, it's not a FITS file, proceed with text parsing
        with open(match_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Additional check: if content looks like binary data (mostly unprintable chars)
        printable_ratio = sum(1 for c in content if c.isprintable() or c.isspace()) / len(content)
        if printable_ratio < 0.7:  # Less than 70% printable characters
            raise ValueError("File appears to be binary data, not text")

        # Parse RMS error
        rms_match = re.search(r'RMS: ([\d.]+) arcsec', content)
        if rms_match:
            result['RMSE'] = np.float64(rms_match.group(1))

        # Parse number of matches (more reliable than console)
        matches_match = re.search(r'Matched (\d+) objects', content)
        if matches_match:
            result['Matches'] = int(matches_match.group(1))

        # If we found no valid data, raise exception to trigger FITS fallback
        if 'RMSE' not in result and 'Matches' not in result:
            raise ValueError("No match data found in text content")

    def _parse_match_fits(self, match_file: str, result: Dict[str, Any]):
        """Parse FITS format match file - data is stored in table fields, not header"""
        with fits.open(match_file) as hdul:
            if len(hdul) > 1:
                data = hdul[1].data

                # Extract key metrics from the data table (not header)
                if len(data) > 0:
                    # Single row table - extract all metrics from first row
                    row = data[0]

                    # Extract all available metrics
                    if 'NMATCH' in data.names:
                        result['Matches'] = int(row['NMATCH'])
                    if 'LOGODDS' in data.names:
                        result['logodds'] = np.float64(row['LOGODDS'])
                    if 'WORSTLOGODDS' in data.names:
                        result['worst_logodds'] = np.float64(row['WORSTLOGODDS'])
                    if 'CODEERR' in data.names:
                        result['RMSE'] = np.float64(row['CODEERR'])  # This is likely the RMS error
                    if 'NDISTRACT' in data.names:
                        result['distractors'] = int(row['NDISTRACT'])
                    if 'NCONFLICT' in data.names:
                        result['conflicts'] = int(row['NCONFLICT'])
                    if 'NFIELD' in data.names:
                        result['field_stars'] = int(row['NFIELD'])
                    if 'NINDEX' in data.names:
                        result['index_stars'] = int(row['NINDEX'])
                    if 'NAGREE' in data.names:
                        result['agreements'] = int(row['NAGREE'])
                    if 'TIMEUSED' in data.names:
                        result['time_used'] = np.float64(row['TIMEUSED'])
                    if 'NVERIFIED' in data.names:
                        result['verified'] = int(row['NVERIFIED'])

                    # Extract WCS information
                    if 'CRVAL' in data.names:
                        result['crval'] = [float(x) for x in row['CRVAL']]
                    if 'CRPIX' in data.names:
                        result['crpix'] = [float(x) for x in row['CRPIX']]
                    if 'CD' in data.names:
                        result['cd_matrix'] = [float(x) for x in row['CD']]
                    if 'WCS_VALID' in data.names:
                        result['wcs_valid'] = bool(row['WCS_VALID'])

                    # Extract field information
                    if 'FIELDNAME' in data.names:
                        result['field_name'] = str(row['FIELDNAME']).strip()
                    if 'HEALPIX' in data.names:
                        result['healpix'] = int(row['HEALPIX'])
                    if 'HPNSIDE' in data.names:
                        result['healpix_nside'] = int(row['HPNSIDE'])
                    if 'PARITY' in data.names:
                        result['parity'] = bool(row['PARITY'])
                    if 'RADIUS' in data.names:
                        result['radius_deg'] = np.float64(row['RADIUS'])

                    # Quad information
                    if 'QTRIED' in data.names:
                        result['quads_tried'] = int(row['QTRIED'])
                    if 'QMATCHED' in data.names:
                        result['quads_matched'] = int(row['QMATCHED'])
                    if 'QSCALEOK' in data.names:
                        result['quads_scale_ok'] = int(row['QSCALEOK'])

                    # Log successful parsing
                    self.log.info(f"Parsed FITS MATCH file with {result.get('Matches', 0)} matches")

                else:
                    self.log.warning("MATCH FITS table has no rows")

            else:
                self.log.warning(f"FITS MATCH file has only {len(hdul)} HDUs, expected at least 2")

    def _parse_rdls_file(self, rdls_file: str, result: Dict[str, Any]):
        """Parse the reference star list file (could be FITS or text)"""
        try:
            # Try FITS format first
            try:
                self._parse_rdls_fits(rdls_file, result)
                self.log.info(f"Parsed RDLS file {rdls_file}.")
                return
            except (OSError, Exception):
                # Fall back to text format
                self._parse_rdls_text(rdls_file, result)
                self.log.info(f"Parsed RDLS file {rdls_file}.")
                return

        except Exception as e:
            self.log.warning(f"Failed to parse RDLS file {rdls_file}: {e}")

    def _parse_rdls_fits(self, rdls_file: str, result: Dict[str, Any]):
        """Parse RDLS file in FITS format"""
        try:
            with fits.open(rdls_file) as hdul:
                if len(hdul) > 1:
                    data = hdul[1].data
                    reference_stars = []

                    if hasattr(data, 'field') and 'RA' in data.names and 'DEC' in data.names:
                        # Structured array with RA, DEC fields
                        ras = data['RA']
                        decs = data['DEC']
                        reference_stars = [[float(ra), float(dec)] for ra, dec in zip(ras, decs)]
                    else:
                        # Assume first two columns are RA, Dec
                        for row in data:
                            if len(row) >= 2:
                                reference_stars.append([float(row[0]), float(row[1])])

                    if reference_stars:
                        result['reference_stars'] = reference_stars
                        self.log.info(f"Parsed {len(reference_stars)} reference stars from FITS RDLS file")

        except Exception as e:
            self.log.warning(f"Failed to parse RDLS FITS file {rdls_file}: {e}")
            raise

    def _parse_rdls_text(self, rdls_file: str, result: Dict[str, Any]):
        """Parse RDLS file in text format"""
        try:
            reference_stars = []

            with open(rdls_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            reference_stars.append([
                                float(parts[0]),  # RA
                                float(parts[1])  # Dec
                            ])
                        except ValueError:
                            continue

            if reference_stars:
                result['reference_stars'] = reference_stars
                self.log.info(f"Parsed {len(reference_stars)} reference stars from text RDLS file")

        except Exception as e:
            self.log.warning(f"Failed to parse RDLS text file {rdls_file}: {e}")
            raise

    def _parse_indx_xyls_file(self, indx_file: str, result: Dict[str, Any]):
        """Parse the index reference stars file (FITS format) - pixel coordinates version"""
        try:
            with fits.open(indx_file) as hdul:
                if len(hdul) > 1:
                    data = hdul[1].data
                    reference_pixels = []

                    # Only X and Y columns available - these are pixel coordinates
                    if (hasattr(data, 'field') and 'X' in data.names and 'Y' in data.names):

                        x_coords = data['X']
                        y_coords = data['Y']

                        for x, y in zip(x_coords, y_coords):
                            # Create list format: [x, y]
                            reference_pixels.append([float(x), float(y)])

                    if reference_pixels:
                        result['reference_pixels'] = reference_pixels
                        result['reference_star_count'] = len(reference_pixels)
                        self.log.info(f"Parsed {len(reference_pixels)} reference star pixel positions from index file")
                    else:
                        self.log.warning(f"No reference star pixels found in {indx_file}")
            self.log.info(f"Parsed INDX fits file {indx_file}.")
        except Exception as e:
            self.log.warning(f"Failed to parse index xyls file {indx_file}: {e}")

# Example usage
if __name__ == "__main__":
    # Configure logging

    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger('pueo')

    # Create sample configuration
    from lib.config import Config
    cfg = Config(config_file="../conf/config.ini", dynamic_file="../conf/dynamic.ini")

    # Create sample centroids data [y, x, flux, std, fwhm]
    # Source: Test Image 1: 0.1-20240420_starcamera-0 - Input Image.png
    # Pueo Detect - precomputed_star_centroids
    precomputed_star_centroids = np.array([
        [50.35772478013797, 832.842807400621, 1393732.3230738079, 3.9782820782863686, 33.01534652709961],
        [24.323750007200452, 3624.1191628331426, 1385927.8224663357, 4.183178366499924, 35.5819206237793],
        [2367.5728033543232, 2526.0537084540074, 1319670.2065404456, 4.392644871676284, 35.58159255981445],
        [936.415634275481, 3381.8929727467466, 1214650.0460666222, 3.7997142435182134, 33.01534652709961],
        [489.9962345388205, 948.6425350632817, 1139248.0940569008, 3.6677104932197846, 33.01534652709961],
        [1496.8508616583842, 4087.9703675759297, 1060500.907411159, 3.8000283473236594, 33.01534652709961],
        [804.5745828157088, 3044.1996994163655, 1040783.2487091227, 3.759154822487739, 33.01534652709961],
        [1240.7813522869185, 195.38505628754098, 989050.6884681566, 3.5703719792994755, 33.01534652709961],
        [374.2970819257794, 1493.485490629251, 983765.8294522628, 3.5721369083597616, 33.01534652709961],
        [558.8539794079618, 1076.2231738888024, 973223.6276197231, 3.571208464335348, 32.5271110534668],
        [1466.9610588025942, 3121.100530197024, 955092.6650298676, 3.993867187652087, 32.58671951293945],
        [437.9450801988132, 2702.55971787717, 951352.3534474038, 3.634624810525788, 33.01534652709961],
        [2072.812413506098, 1714.9783444320076, 731239.7473929332, 3.5433540445108327, 30.8870906829834],
        [459.44918236619367, 3406.750453301172, 689902.386099018, 3.4565895066300154, 30.280736923217773],
        [1578.3917116383734, 1438.9346802935295, 639672.0104788896, 3.475331783334556, 29.968122482299805],
        [575.4774697467614, 3728.1359523304923, 618693.1391110654, 3.604475267751281, 32.5271110534668],
        [319.1934176850595, 4054.7441994574597, 598332.7576693327, 3.495814137089522, 29.83306884765625],
        [764.4523252078069, 247.807666885608, 584760.4905335637, 3.439662792680999, 32.586524963378906],
        [29.457408432995017, 2269.8649163426203, 528099.0331072189, 3.3632795436489062, 29.15496063232422],
        [1581.1116485341183, 1717.2774246529716, 522029.2664776763, 3.2540045285318397, 27.87146759033203],
        [45.95619108854863, 1254.8065190409893, 501168.8111268603, 3.2094003313188235, 30.8870906829834],
        [1262.347203564979, 483.9229751778145, 494501.8251999598, 3.3862800393966235, 30.280803680419922],
        [29.445001865019698, 1598.7002680640958, 461595.7342816644, 3.9616233469061135, 30.8870906829834],
        [1078.9998437869754, 2007.2477873126982, 449489.78434747417, 3.3380573121335892, 29.83306884765625],
        [151.19713002790556, 2612.688365161704, 438977.41849751957, 3.1601040114121104, 29.851760864257812],
        [301.86253889895113, 1633.5965262203365, 421670.300192366, 2.995292771087609, 26.870258331298828],
        [191.70569297224478, 1476.1273731178671, 378343.74253315805, 3.281080639730296, 27.459260940551758],
        [692.0349830815952, 950.7323107617458, 375004.075377139, 3.1579769138157503, 27.459260940551758],
        [1096.5722562378203, 2391.4864947446144, 369912.3294016407, 3.3933909046102078, 27.459260940551758],
        [1047.721643840157, 1497.248514347113, 332218.02723498875, 3.3783629650762146, 29.83306884765625],
        [660.7878722470165, 3110.019563812419, 323487.79032094753, 2.981001149393974, 26.870258331298828],
        [808.4687136864574, 122.81165842546909, 313258.04545914725, 3.0223326979425984, 26.870258331298828],
        [263.67739886588095, 495.7606661086708, 234218.63749114124, 2.9366033226104733, 27.459260940551758],
    ])
    max_c, _ = precomputed_star_centroids.shape
    image_size = (2822, 4144)

    # Process the centroids with configuration
    an_solver = AstrometryNet(cfg, log)

    output_base = "6.0 - astrometry.net-solve-field-centroids"
    output_dir = "../partial_results"

    # Test end to end process_centroids
    if False:
        try:
            astrometry, solver_exec_time = an_solver.process_centroids(
                precomputed_star_centroids[:max_c],
                image_size,
                output_base=output_base,
                output_dir=cfg.partial_results_path  # "./astrometry_results"
            )

            print(f"Processing successful: {astrometry['success']}")
            if astrometry['success']:
                print("Solution files created:")
                for name, path in astrometry['solution_files'].items():
                    print(f"  {name}: {path}")
        except Exception as e:
            print(f"Error processing centroids: {e}")

    # Parsing
    if True:
        solve_time = 0.0
        class result:
            stdout = ''
            stderr = ''

      # Get base filename for output files
        output_path = Path(output_dir)
        fits_filename = output_path / f"{output_base}.xyls.fits"
        base_filename = os.path.splitext(fits_filename)[0]
        if output_dir:
            base_filename = os.path.join(output_dir, os.path.basename(base_filename))

        log.info(f'base_filename: {base_filename}')
        # Parse the complete solution
        parser = AstrometryNetParser(log)
        parsed_data = parser.parse_solution(result.stdout, result.stderr, base_filename)
        parsed_data['T_solve'] = solve_time

        # Check what files were successfully parsed
        if 'detected_stars' in parsed_data:
            log.info(f"Detected {parsed_data['detected_stars_count']} stars in image")
        if 'reference_stars' in parsed_data:
            with suppress(TypeError):
                log.info(f"Found {len(parsed_data.get('reference_stars', []))} reference stars in catalog")
        if 'matched_centroids' in parsed_data:
            log.info(f"Successfully matched {parsed_data['Matches']} stars")

        log.info(f'{parsed_data}')