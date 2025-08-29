import numpy as np
from astropy.table import Table
from astropy.io import fits
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
import re


class AstrometryNet:
    """
    A class to handle conversion of centroid data to FITS format and
    interface with astrometry.net's solve-field command.

    Attributes:
        cfg: Configuration object with an_ prefixed properties
        log: Logger instance for tracking operations
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
        patterns = [
            f"an_{param_name}",  # an_scale_units
            f"an_{param_name.replace('_', '')}",  # an_scaleunits
            param_name,  # scale_units (direct)
        ]

        for pattern in patterns:
            if hasattr(self.cfg, pattern):
                value = getattr(self.cfg, pattern)
                if value is not None:
                    return value

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
            'corr': 'corr',
            'solved': 'solved',
            'axy': 'axy',
            'new-fits': 'new_fits',
            'kmz': 'kmz',
            'temp-axy': 'temp_axy',
            'backend-config': 'backend_config',
            'no-plot': 'no_plot',
            'plot-scale': 'plot_scale',
            'stretch': 'stretch',
            'grayplot': 'grayplot',
            'image': 'image',
            'fits-image': 'fits_image',
            'port': 'port',
            'ip': 'ip'
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
                                  output_dir: Optional[str] = None) -> List[str]:
        """
        Build the solve-field command with configured parameters.

        Args:
            fits_filename: Input FITS file to solve
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
                    output_dir: Optional[str] = None,
                    timeout: int = 300) -> Dict:
        """
        Execute solve-field command and capture results.

        Args:
            fits_filename: Input FITS file to solve
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
            raise FileNotFoundError("solve-field command not found. Install astrometry.net first.")

        # Build command
        cmd = self.build_solve_field_command(fits_filename, output_dir)

        self.log.info(f"Executing: {' '.join(cmd)}")

        # Execute command
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True
            )

            # Parse output for useful information
            output = {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0,
                'command': ' '.join(cmd)
            }

            self.log.info(f"solve-field completed with return code {result.returncode}")
            if result.stdout:
                self.log.debug(f"stdout: {result.stdout[:500]}...")  # First 500 chars
            if result.stderr:
                self.log.warning(f"stderr: {result.stderr[:500]}...")

            return output

        except subprocess.TimeoutExpired:
            self.log.error(f"solve-field timed out after {timeout} seconds")
            raise
        except subprocess.CalledProcessError as e:
            self.log.error(f"solve-field failed with return code {e.returncode}")
            self.log.error(f"Error output: {e.stderr}")
            raise

    def process_centroids(self, centroids: np.ndarray,
                          output_base: str = "centroids",
                          output_dir: str = ".",
                          cleanup: bool = True) -> Dict:
        """
        Complete pipeline: convert centroids to FITS and solve field.

        Args:
            centroids: NumPy array with centroid data
            output_base: Base name for output files
            output_dir: Directory for output files
            cleanup: Whether to remove intermediate files

        Returns:
            Dictionary with processing results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Convert to FITS
        fits_filename = str(output_path / f"{output_base}.xyls.fits")
        self.ndarray2fits(centroids, fits_filename)

        # Solve field
        result = self.solve_field(fits_filename, output_dir=str(output_path))

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

        return result


# Example configuration class that mimics your setup
class Config:
    """Example configuration class with an_ prefixed properties"""

    def __init__(self):
        # These would be loaded from your config.ini [ASTROMETRY.NET] section
        self.an_scale_units = 'degwidth'
        self.an_scale_low = 0.1
        self.an_scale_high = 1.0
        self.an_downsample = 2
        self.an_overwrite = True
        self.an_no_plots = True
        self.an_cpulimit = 30
        self.an_depth = 20


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create sample configuration
    cfg = Config()

    # Create sample centroids data [y, x, flux, std, fwhm]
    centroids = np.array([
        [100.5, 200.3, 5000.0, 0.8, 2.5],
        [150.2, 180.7, 3000.0, 0.6, 2.2],
        [120.8, 220.1, 4500.0, 0.7, 2.4]
    ])

    # Process the centroids with configuration
    solver = AstrometryNet(cfg)
    try:
        result = solver.process_centroids(
            centroids,
            output_base= "4.0 - astrometry.net-centroids",
            output_dir= "../partial_results"
        )

        print(f"Processing successful: {result['success']}")
        if result['success']:
            print("Solution files created:")
            for name, path in result['solution_files'].items():
                print(f"  {name}: {path}")

    except Exception as e:
        print(f"Error processing centroids: {e}")